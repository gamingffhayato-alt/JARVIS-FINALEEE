"""
EduBot - AI-Powered Education Telegram Bot
Uses Groq API (Llama 4 Scout) for AI, supports text/image/audio input,
generates PDFs with rendered math/physics, fetches Wikimedia diagrams.
Errors are forwarded to a separate error-reporting bot.
"""

import os
import io
import re
import sys
import json
import logging
import asyncio
import tempfile
import traceback
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

import httpx
import aiofiles
import aiohttp
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.mathtext as mathtext
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from telegram import Update, InputFile
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from telegram.constants import ParseMode, ChatAction
from groq import AsyncGroq

# ─────────────────────────── Configuration ───────────────────────────

BOT_TOKEN        = os.environ["TELEGRAM_BOT_TOKEN"]
ERROR_BOT_TOKEN  = os.environ["TELEGRAM_ERROR_BOT_TOKEN"]
ERROR_CHAT_ID    = os.environ["TELEGRAM_ERROR_CHAT_ID"]   # your personal chat/group id
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]

GROQ_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
WHISPER_MODEL    = "whisper-large-v3"
MAX_TOKENS       = 4096
WIKIMEDIA_API    = "https://en.wikipedia.org/w/api.php"
WIKIMEDIA_COMMONS = "https://commons.wikimedia.org/w/api.php"

# ─────────────────────────── Logging ────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EduBot")

# ─────────────────────────── Error Reporter ─────────────────────────

async def report_error(error: Exception, context_info: str = ""):
    """Send detailed error info to the separate error bot."""
    tb = traceback.format_exc()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    message = (
        f"🚨 <b>EduBot Error</b>\n"
        f"<b>Time:</b> {timestamp}\n"
        f"<b>Context:</b> {context_info or 'N/A'}\n"
        f"<b>Error:</b> <code>{type(error).__name__}: {str(error)[:300]}</code>\n\n"
        f"<b>Traceback:</b>\n<pre>{tb[:2000]}</pre>"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{ERROR_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": ERROR_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
    except Exception as e:
        logger.error(f"Failed to send error report: {e}")


def error_guard(context_label: str = ""):
    """Decorator that catches exceptions, reports them, and replies gracefully."""
    def decorator(func):
        async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            try:
                return await func(update, ctx, *args, **kwargs)
            except Exception as exc:
                user = update.effective_user
                info = f"{context_label} | user={user.id if user else '?'}"
                logger.exception(f"Error in {func.__name__}: {exc}")
                await report_error(exc, info)
                try:
                    await update.message.reply_text(
                        "⚠️ An error occurred while processing your request. "
                        "The dev team has been notified automatically!"
                    )
                except Exception:
                    pass
        return wrapper
    return decorator

# ─────────────────────────── Groq Client ────────────────────────────

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are EduBot — an expert AI tutor specialising in Physics, Mathematics, Chemistry, Biology, Computer Science, and all academic subjects.

Guidelines:
- Give clear, step-by-step explanations suited to the student's level.
- For maths/physics, always show full working. Use LaTeX notation wrapped in $...$ for inline and $$...$$ for display equations.
- When asked for diagrams, suggest a Wikimedia search term in the format: [DIAGRAM: <search term>]
- When asked to generate a PDF, respond with your full answer and append: [GENERATE_PDF]
- Be encouraging, precise, and academically rigorous.
- If an image is shared, analyse it and help the student understand it or solve problems in it.
- Keep answers well-structured with headings where appropriate.
"""

# ─────────────────────────── Math Renderer ──────────────────────────

def render_latex_to_image(latex: str, dpi: int = 150) -> Optional[bytes]:
    """Render a LaTeX expression to a PNG image using matplotlib."""
    try:
        fig = plt.figure(figsize=(8, 1.5))
        fig.patch.set_facecolor("white")
        # Wrap in display math if not already
        expr = latex.strip()
        if not expr.startswith("$"):
            expr = f"${expr}$"
        fig.text(0.5, 0.5, expr, fontsize=16, ha="center", va="center",
                 color="black", usetex=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.2, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.warning(f"LaTeX render failed for '{latex[:60]}': {e}")
        plt.close("all")
        return None


def extract_latex_blocks(text: str):
    """
    Extract display math blocks ($$...$$) from text.
    Returns list of (before_text, latex_expr, after_text) tuples.
    """
    parts = re.split(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)
    return parts  # alternating: text, latex, text, latex, ...

# ─────────────────────────── Wikimedia ──────────────────────────────

async def fetch_wikimedia_image(search_term: str) -> Optional[tuple[bytes, str]]:
    """Search Wikimedia Commons and return (image_bytes, description)."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Search Commons
            params = {
                "action": "query", "list": "search",
                "srsearch": search_term, "srnamespace": "6",
                "srlimit": "3", "format": "json"
            }
            r = await client.get(WIKIMEDIA_COMMONS, params=params)
            data = r.json()
            results = data.get("query", {}).get("search", [])
            if not results:
                # Fallback: search Wikipedia for diagrams
                params2 = {
                    "action": "query", "prop": "pageimages",
                    "titles": search_term, "pithumbsize": "600",
                    "format": "json", "pilimit": "1"
                }
                r2 = await client.get(WIKIMEDIA_API, params=params2)
                d2 = r2.json()
                pages = d2.get("query", {}).get("pages", {})
                for page in pages.values():
                    thumb = page.get("thumbnail", {}).get("source")
                    if thumb:
                        img_r = await client.get(thumb)
                        return img_r.content, search_term
                return None

            # Get the first result's image URL
            title = results[0]["title"]  # e.g. "File:Diagram.svg"
            params3 = {
                "action": "query", "titles": title,
                "prop": "imageinfo", "iiprop": "url|mime",
                "iiurlwidth": "600", "format": "json"
            }
            r3 = await client.get(WIKIMEDIA_COMMONS, params=params3)
            d3 = r3.json()
            pages = d3.get("query", {}).get("pages", {})
            for page in pages.values():
                info_list = page.get("imageinfo", [])
                if info_list:
                    url = info_list[0].get("thumburl") or info_list[0].get("url", "")
                    if url:
                        img_r = await client.get(url)
                        if img_r.status_code == 200:
                            return img_r.content, title.replace("File:", "")
    except Exception as e:
        logger.warning(f"Wikimedia fetch failed: {e}")
    return None

# ─────────────────────────── PDF Generator ──────────────────────────

def build_pdf(title: str, content: str, diagram_images: list[tuple[bytes, str]] = None) -> bytes:
    """
    Build a polished PDF from markdown-like content.
    Renders $$...$$ blocks as images. Appends diagram images.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
        title=title
    )

    styles = getSampleStyleSheet()
    style_title  = ParagraphStyle("EduTitle",  parent=styles["Title"],
                                   fontSize=22, spaceAfter=14, textColor=colors.HexColor("#1a237e"))
    style_h1     = ParagraphStyle("EduH1",     parent=styles["Heading1"],
                                   fontSize=16, spaceBefore=12, spaceAfter=6,
                                   textColor=colors.HexColor("#283593"))
    style_h2     = ParagraphStyle("EduH2",     parent=styles["Heading2"],
                                   fontSize=13, spaceBefore=8, spaceAfter=4,
                                   textColor=colors.HexColor("#3949ab"))
    style_body   = ParagraphStyle("EduBody",   parent=styles["Normal"],
                                   fontSize=11, leading=16, alignment=TA_JUSTIFY)
    style_code   = ParagraphStyle("EduCode",   parent=styles["Code"],
                                   fontSize=9,  leading=13, backColor=colors.HexColor("#f5f5f5"),
                                   borderPadding=4)
    style_caption = ParagraphStyle("EduCaption", parent=styles["Normal"],
                                    fontSize=9, textColor=colors.grey, alignment=TA_CENTER)

    story = []

    # Title
    story.append(Paragraph(title, style_title))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3949ab")))
    story.append(Spacer(1, 10))

    # Process content line by line
    parts = extract_latex_blocks(content)
    # parts = [text, latex, text, latex, ...]
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # LaTeX display block — render to image
            img_bytes = render_latex_to_image(part)
            if img_bytes:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_bytes)
                tmp.flush()
                tmp_path = tmp.name
                tmp.close()
                try:
                    rl_img = RLImage(tmp_path, width=13*cm, height=None)
                    rl_img.hAlign = "CENTER"
                    story.append(rl_img)
                    story.append(Spacer(1, 4))
                finally:
                    os.unlink(tmp_path)
            else:
                # Fallback: show as monospace
                story.append(Paragraph(f"<font name='Courier'>{part}</font>", style_code))
        else:
            # Regular text — parse simple markdown
            lines = part.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 4))
                    continue
                if line.startswith("### "):
                    story.append(Paragraph(line[4:], style_h2))
                elif line.startswith("## "):
                    story.append(Paragraph(line[3:], style_h1))
                elif line.startswith("# "):
                    story.append(Paragraph(line[2:], style_h1))
                elif line.startswith("**") and line.endswith("**"):
                    story.append(Paragraph(f"<b>{line[2:-2]}</b>", style_body))
                elif line.startswith("- ") or line.startswith("* "):
                    story.append(Paragraph(f"• {line[2:]}", style_body))
                elif re.match(r"^\d+\. ", line):
                    story.append(Paragraph(line, style_body))
                elif line.startswith("`") and line.endswith("`"):
                    story.append(Paragraph(f"<font name='Courier'>{line[1:-1]}</font>", style_code))
                else:
                    # Inline math $...$ — render inline by showing in code font
                    processed = re.sub(r"\$([^$]+)\$",
                                       lambda m: f"<font name='Courier' color='#1a237e'>{m.group(1)}</font>",
                                       line)
                    story.append(Paragraph(processed, style_body))

    # Diagrams section
    if diagram_images:
        story.append(PageBreak())
        story.append(Paragraph("📊 Diagrams", style_h1))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3949ab")))
        story.append(Spacer(1, 8))
        for img_bytes, caption in diagram_images:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            # Convert to PNG if needed
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                pil_img.save(tmp.name, "PNG")
            except Exception:
                tmp.write(img_bytes)
                tmp.flush()
            tmp.close()
            try:
                rl_img = RLImage(tmp.name, width=14*cm, height=None)
                rl_img.hAlign = "CENTER"
                story.append(KeepTogether([
                    rl_img,
                    Spacer(1, 4),
                    Paragraph(caption, style_caption),
                    Spacer(1, 12),
                ]))
            finally:
                os.unlink(tmp.name)

    # Footer timestamp
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(
        f"Generated by EduBot on {datetime.utcnow().strftime('%B %d, %Y at %H:%M UTC')}",
        style_caption
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ─────────────────────────── Groq Helpers ───────────────────────────

async def ask_groq_text(messages: list) -> str:
    response = await groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
    )
    return response.choices[0].message.content


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    """Transcribe audio using Groq Whisper."""
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".ogg", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as af:
            transcription = await groq_client.audio.transcriptions.create(
                file=(filename, af),
                model=WHISPER_MODEL,
                response_format="text",
            )
        return transcription
    finally:
        os.unlink(tmp_path)


async def ask_groq_vision(image_bytes: bytes, prompt: str, mime: str = "image/jpeg") -> str:
    """Send an image + prompt to Llama 4 Scout (vision)."""
    b64 = base64.b64encode(image_bytes).decode()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt or "Please analyse this image and explain the educational content."}
        ]}
    ]
    response = await groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content

# ─────────────────────────── Conversation Store ─────────────────────

# Simple in-memory conversation history per user
conversation_history: dict[int, list] = {}

def get_history(user_id: int) -> list:
    if user_id not in conversation_history:
        conversation_history[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return conversation_history[user_id]

def add_message(user_id: int, role: str, content):
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    # Keep last 20 exchanges (40 messages) + system
    if len(history) > 41:
        conversation_history[user_id] = [history[0]] + history[-40:]

# ─────────────────────────── Handlers ───────────────────────────────

@error_guard("start")
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "Student"
    await update.message.reply_text(
        f"👋 Hello *{name}*\\! I'm *EduBot* — your AI tutor\\.\n\n"
        f"I can help you with:\n"
        f"📐 *Maths & Physics* — step\\-by\\-step solutions\n"
        f"🧪 *Chemistry & Biology*\n"
        f"💻 *Computer Science*\n"
        f"📊 *Diagrams* from Wikimedia\n"
        f"📄 *PDF notes* with rendered equations\n\n"
        f"Send me a *text question*, an *image*, or a *voice message*\\!\n\n"
        f"Commands:\n"
        f"/pdf — convert last answer to PDF\n"
        f"/diagram \\<topic\\> — fetch a diagram\n"
        f"/clear — clear chat history\n"
        f"/help — show this message",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


@error_guard("help")
async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, ctx)


@error_guard("clear")
async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    conversation_history.pop(uid, None)
    await update.message.reply_text("🗑️ Conversation history cleared. Fresh start!")


@error_guard("pdf_command")
async def cmd_pdf(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    history = get_history(uid)
    # Find the last assistant message
    last_answer = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), None)
    if not last_answer:
        await update.message.reply_text("No previous answer found. Ask me something first!")
        return
    await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
    question = next((m["content"] for m in reversed(history) if m["role"] == "user"), "Answer")
    title = (question[:60] + "...") if len(question) > 60 else question
    # Check for diagram tags
    diagram_data = []
    for tag in re.findall(r"\[DIAGRAM:\s*(.+?)\]", last_answer):
        result = await fetch_wikimedia_image(tag)
        if result:
            diagram_data.append(result)
    pdf_bytes = build_pdf(title, last_answer, diagram_data)
    await update.message.reply_document(
        document=InputFile(io.BytesIO(pdf_bytes), filename="EduBot_Notes.pdf"),
        caption="📄 Here are your notes as a PDF!"
    )


@error_guard("diagram_command")
async def cmd_diagram(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = " ".join(ctx.args) if ctx.args else ""
    if not query:
        await update.message.reply_text("Usage: /diagram <topic>\nExample: /diagram Newton's laws of motion")
        return
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    result = await fetch_wikimedia_image(query)
    if result:
        img_bytes, caption = result
        await update.message.reply_photo(
            photo=InputFile(io.BytesIO(img_bytes), filename="diagram.png"),
            caption=f"📊 *{caption}*\nSource: Wikimedia Commons",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await update.message.reply_text(f"❌ No diagram found for '{query}'. Try a different search term.")


@error_guard("text_message")
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_text = update.message.text

    await update.message.chat.send_action(ChatAction.TYPING)

    add_message(uid, "user", user_text)
    history = get_history(uid)

    response_text = await ask_groq_text(history)
    add_message(uid, "assistant", response_text)

    # Check if PDF generation was requested
    generate_pdf = "[GENERATE_PDF]" in response_text
    clean_response = response_text.replace("[GENERATE_PDF]", "").strip()

    # Check for diagram tags
    diagram_tags = re.findall(r"\[DIAGRAM:\s*(.+?)\]", clean_response)
    display_text = re.sub(r"\[DIAGRAM:\s*.+?\]", "", clean_response).strip()

    # Send text response (split if > 4096 chars)
    chunks = [display_text[i:i+4000] for i in range(0, len(display_text), 4000)]
    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    # Fetch and send diagrams
    for tag in diagram_tags:
        await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
        result = await fetch_wikimedia_image(tag)
        if result:
            img_bytes, caption = result
            await update.message.reply_photo(
                photo=InputFile(io.BytesIO(img_bytes), filename="diagram.png"),
                caption=f"📊 *{caption}* (Wikimedia Commons)",
                parse_mode=ParseMode.MARKDOWN,
            )

    # Auto-generate PDF if requested
    if generate_pdf:
        await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
        diagram_data = []
        for tag in diagram_tags:
            result = await fetch_wikimedia_image(tag)
            if result:
                diagram_data.append(result)
        title = (user_text[:60] + "...") if len(user_text) > 60 else user_text
        pdf_bytes = build_pdf(title, clean_response, diagram_data)
        await update.message.reply_document(
            document=InputFile(io.BytesIO(pdf_bytes), filename="EduBot_Notes.pdf"),
            caption="📄 Your PDF notes are ready!"
        )


@error_guard("photo_message")
async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    caption = update.message.caption or ""

    await update.message.chat.send_action(ChatAction.TYPING)

    # Get highest resolution photo
    photo = update.message.photo[-1]
    photo_file = await photo.get_file()
    img_bytes = await photo_file.download_as_bytearray()

    prompt = caption if caption else "Analyse this image. If it contains a problem, solve it step by step. If it's a diagram, explain it."
    response_text = await ask_groq_vision(bytes(img_bytes), prompt)
    add_message(uid, "assistant", response_text)

    clean_response = response_text.replace("[GENERATE_PDF]", "").strip()
    diagram_tags = re.findall(r"\[DIAGRAM:\s*(.+?)\]", clean_response)
    display_text = re.sub(r"\[DIAGRAM:\s*.+?\]", "", clean_response).strip()

    chunks = [display_text[i:i+4000] for i in range(0, len(display_text), 4000)]
    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    for tag in diagram_tags:
        result = await fetch_wikimedia_image(tag)
        if result:
            img_bytes2, caption2 = result
            await update.message.reply_photo(
                photo=InputFile(io.BytesIO(img_bytes2), filename="diagram.png"),
                caption=f"📊 {caption2}",
            )

    if "[GENERATE_PDF]" in response_text:
        title = (caption[:60] + "...") if caption else "Image Analysis"
        pdf_bytes = build_pdf(title, clean_response, [])
        await update.message.reply_document(
            document=InputFile(io.BytesIO(pdf_bytes), filename="EduBot_Notes.pdf"),
            caption="📄 PDF notes generated!"
        )


@error_guard("audio_message")
async def handle_audio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id

    # Get audio file (voice note or audio file)
    audio = update.message.voice or update.message.audio
    if not audio:
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    audio_file = await audio.get_file()
    audio_bytes = await audio_file.download_as_bytearray()
    ext = ".ogg" if update.message.voice else ".mp3"

    await update.message.reply_text("🎙️ Transcribing your audio...")

    transcript = await transcribe_audio(bytes(audio_bytes), f"audio{ext}")
    if not transcript:
        await update.message.reply_text("❌ Could not transcribe the audio. Please try again.")
        return

    await update.message.reply_text(f"📝 *Transcribed:* {transcript}", parse_mode=ParseMode.MARKDOWN)
    await update.message.chat.send_action(ChatAction.TYPING)

    add_message(uid, "user", transcript)
    history = get_history(uid)
    response_text = await ask_groq_text(history)
    add_message(uid, "assistant", response_text)

    clean_response = response_text.replace("[GENERATE_PDF]", "").strip()
    diagram_tags = re.findall(r"\[DIAGRAM:\s*(.+?)\]", clean_response)
    display_text = re.sub(r"\[DIAGRAM:\s*.+?\]", "", clean_response).strip()

    chunks = [display_text[i:i+4000] for i in range(0, len(display_text), 4000)]
    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    for tag in diagram_tags:
        result = await fetch_wikimedia_image(tag)
        if result:
            img_bytes2, caption2 = result
            await update.message.reply_photo(
                photo=InputFile(io.BytesIO(img_bytes2), filename="diagram.png"),
                caption=f"📊 {caption2}",
            )

    if "[GENERATE_PDF]" in response_text:
        title = (transcript[:60] + "...") if len(transcript) > 60 else transcript
        pdf_bytes = build_pdf(title, clean_response, [])
        await update.message.reply_document(
            document=InputFile(io.BytesIO(pdf_bytes), filename="EduBot_Notes.pdf"),
            caption="📄 PDF notes ready!"
        )


@error_guard("document_message")
async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    mime = doc.mime_type or ""

    if "image" in mime:
        # Treat as image
        await update.message.chat.send_action(ChatAction.TYPING)
        doc_file = await doc.get_file()
        img_bytes = await doc_file.download_as_bytearray()
        caption = update.message.caption or "Analyse this image."
        uid = update.effective_user.id
        response_text = await ask_groq_vision(bytes(img_bytes), caption, mime)
        add_message(uid, "assistant", response_text)
        clean = response_text.replace("[GENERATE_PDF]", "").strip()
        for chunk in [clean[i:i+4000] for i in range(0, len(clean), 4000)]:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(
            "📁 I can process image files. For PDFs or text files, "
            "please paste the content as text."
        )


async def global_error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    """Catch-all Telegram error handler."""
    exc = ctx.error
    logger.error(f"Unhandled error: {exc}")
    await report_error(exc, f"global_error_handler | update={type(update).__name__}")

# ─────────────────────────── Main ───────────────────────────────────

def main():
    logger.info("Starting EduBot...")
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("pdf",   cmd_pdf))
    app.add_handler(CommandHandler("diagram", cmd_diagram))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    app.add_error_handler(global_error_handler)

    logger.info("EduBot is running. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
