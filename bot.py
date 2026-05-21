"""
EduBot - AI-Powered Education Telegram Bot
Uses Groq API (Llama 4 Scout) for AI, supports text/image/audio input,
generates PDFs with rendered math/physics, generates Nano Banana diagrams.
Errors are forwarded to a separate error-reporting bot.
"""

import os
import io
import re
import sys
import json
import html
import logging
import asyncio
import tempfile
import traceback
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional

import httpx
from PIL import Image
import matplotlib
matplotlib.use("Agg")

# Use Matplotlib's built-in math renderer (requires ZERO external TeX installations)
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'

import matplotlib.pyplot as plt
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

# Import our new Nano Banana generator
from image_generator import generate_diagram

# ─────────────────────────── Configuration ───────────────────────────

BOT_TOKEN        = os.environ["TELEGRAM_BOT_TOKEN"]
ERROR_BOT_TOKEN  = os.environ["TELEGRAM_ERROR_BOT_TOKEN"]
ERROR_CHAT_ID    = os.environ["TELEGRAM_ERROR_CHAT_ID"]
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]

GROQ_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
WHISPER_MODEL    = "whisper-large-v3"
MAX_TOKENS       = 4096

# ─────────────────────────── Logging ────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EduBot")

# ─────────────────────────── Error Reporter ─────────────────────────

async def report_error(error: Exception, context_info: str = ""):
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

# ─────────────────────────── Formatting Helpers ─────────────────────

def escape_and_format_html(text: str) -> str:
    escaped = html.escape(text, quote=False)
    formatted = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
    return formatted

def sanitize_latex_for_pdf(text: str) -> str:
    # Safely convert double backslashes before letters to single backslashes
    text = re.sub(r"\\\\([a-zA-Z])", r"\\\1", text)
    
    text = text.replace(r"\$", "$")
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    
    # Strip unsupported LaTeX environments that break mathtext
    text = re.sub(r"\\begin\{[a-zA-Z*]+\}", "", text)
    text = re.sub(r"\\end\{[a-zA-Z*]+\}", "", text)
    
    # Strip unsupported \boxed command and convert \text to \mathrm
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\text\{(.*?)\}", r"\\mathrm{\1}", text)
    
    return text

# ─────────────────────────── Groq Client ────────────────────────────

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are EduBot — an expert AI tutor.

CRITICAL MATH FORMATTING & REASONING RULES:
1. NEVER escape dollar signs. Write $5 \Omega$, NOT \$5 \Omega\$.
2. ALL equations, numbers, and variables MUST be wrapped in $...$ (inline) or $$...$$ (display math). 
3. NEVER write naked equations. Every single equation must have a delimiter.
4. DO NOT use complex LaTeX environments like \begin{vmatrix}, \begin{matrix}, \begin{array}, \begin{align}, etc.
5. For cross products and determinants, DO NOT draw a matrix. Write the algebraic expansion linearly. Example: $\vec{A} \times \vec{B} = (A_y B_z - A_z B_y)\hat{i} - ...$
6. DO NOT use \boxed{} or \text{} as they break the renderer. Use simple, basic LaTeX equations. Use \mathrm{} instead of \text{}.
7. Think step-by-step and DOUBLE-CHECK algebraic manipulations. ALWAYS convert units to standard SI (e.g., mA to A) before solving equations.
8. When asked for diagrams, suggest a detailed visual prompt for an AI image generator: [DIAGRAM: <detailed visual prompt>]
9. To generate a PDF, append: [GENERATE_PDF]
"""

# ─────────────────────────── Math Renderer ──────────────────────────

def render_math_to_image(latex: str, dpi: int = 150, inline: bool = False) -> Optional[tuple[bytes, float, float]]:
    """Renders LaTeX to PNG using built-in mathtext. Returns (bytes, width_pt, height_pt)."""
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_facecolor("white")
        
        expr = latex.strip()
        if not expr.startswith("$"):
            expr = f"${expr}$"
            
        fontsize = 12 if inline else 16
        t = fig.text(0, 0, expr, fontsize=fontsize, color="black")
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        
        img_bytes = buf.read()
        with Image.open(io.BytesIO(img_bytes)) as img:
            w_px, h_px = img.size
            # Convert pixels at current DPI to ReportLab points (72 pts per inch)
            scale = 1.05 if inline else 1.2
            w_pt = (w_px / dpi) * 72 * scale
            h_pt = (h_px / dpi) * 72 * scale
            
        return img_bytes, w_pt, h_pt
    except Exception as e:
        logger.warning(f"Mathtext render failed for '{latex[:60]}': {e}")
        plt.close("all")
        return None

def extract_latex_blocks(text: str):
    parts = re.split(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)
    return parts

# ─────────────────────────── PDF Generator ──────────────────────────

def build_pdf(title: str, content: str, diagram_images: list[tuple[bytes, str]] = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm, title=title
    )

    styles = getSampleStyleSheet()
    style_title  = ParagraphStyle("EduTitle", parent=styles["Title"], fontSize=22, spaceAfter=14, textColor=colors.HexColor("#1a237e"))
    style_h1     = ParagraphStyle("EduH1", parent=styles["Heading1"], fontSize=16, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#283593"))
    style_h2     = ParagraphStyle("EduH2", parent=styles["Heading2"], fontSize=13, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#3949ab"))
    style_body   = ParagraphStyle("EduBody", parent=styles["Normal"], fontSize=11, leading=18, alignment=TA_JUSTIFY)
    style_code   = ParagraphStyle("EduCode", parent=styles["Code"], fontSize=9, backColor=colors.HexColor("#f5f5f5"), borderPadding=4)
    style_caption = ParagraphStyle("EduCaption", parent=styles["Normal"], fontSize=9, textColor=colors.grey, alignment=TA_CENTER)

    story = []
    temp_files = []

    def process_text_line(text, style):
        math_store = []
        def store_math(m):
            math_store.append(m.group(1))
            return f"__MATH_{len(math_store)-1}__"
            
        # Isolate math so it doesn't get ruined by html.escape
        text = re.sub(r"\$([^$]+)\$", store_math, text)
        text = html.escape(text)
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        
        def inject_math(m):
            idx = int(m.group(1))
            latex = math_store[idx]
            res = render_math_to_image(latex, inline=True)
            if res:
                img_b, w, h = res
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_b)
                tmp.flush()
                temp_files.append(tmp.name)
                tmp.close()
                # Use negative valign to align image nicely with text baseline
                return f'<img src="{tmp.name}" width="{w}" height="{h}" valign="{-h*0.2}"/>'
            else:
                return f"<font name='Courier' color='#1a237e'>{html.escape(latex)}</font>"
                
        text = re.sub(r"__MATH_(\d+)__", inject_math, text)
        return Paragraph(text, style)

    try:
        story.append(Paragraph(title, style_title))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3949ab")))
        story.append(Spacer(1, 10))

        clean_content = sanitize_latex_for_pdf(content)
        parts = extract_latex_blocks(clean_content)
        
        for i, part in enumerate(parts):
            if i % 2 == 1:
                # Display block math
                res = render_math_to_image(part, inline=False)
                if res:
                    img_b, w, h = res
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    tmp.write(img_b)
                    tmp.flush()
                    temp_files.append(tmp.name)
                    tmp.close()
                    rl_img = RLImage(tmp.name, width=w, height=h)
                    rl_img.hAlign = "CENTER"
                    story.append(rl_img)
                    story.append(Spacer(1, 4))
                else:
                    story.append(Paragraph(f"<font name='Courier'>{html.escape(part)}</font>", style_code))
            else:
                # Text processing
                lines = part.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        story.append(Spacer(1, 4))
                        continue
                    
                    if line.startswith("### "):
                        story.append(process_text_line(line[4:], style_h2))
                    elif line.startswith("## "):
                        story.append(process_text_line(line[3:], style_h1))
                    elif line.startswith("# "):
                        story.append(process_text_line(line[2:], style_h1))
                    elif line.startswith("- ") or line.startswith("* "):
                        story.append(process_text_line(f"• {line[2:]}", style_body))
                    elif line.startswith("`") and line.endswith("`"):
                        story.append(Paragraph(f"<font name='Courier'>{html.escape(line[1:-1])}</font>", style_code))
                    else:
                        story.append(process_text_line(line, style_body))

        if diagram_images:
            story.append(PageBreak())
            story.append(Paragraph("📊 Diagrams", style_h1))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3949ab")))
            story.append(Spacer(1, 8))
            for img_bytes, caption in diagram_images:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_files.append(tmp.name)
                try:
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_img.save(tmp.name, "PNG")
                except Exception:
                    tmp.write(img_bytes)
                    tmp.flush()
                tmp.close()
                rl_img = RLImage(tmp.name, width=14*cm, height=None)
                rl_img.hAlign = "CENTER"
                story.append(KeepTogether([
                    rl_img, Spacer(1, 4),
                    Paragraph(html.escape(caption), style_caption),
                    Spacer(1, 12),
                ]))

        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Paragraph(
            f"Generated by EduBot on {datetime.utcnow().strftime('%B %d, %Y at %H:%M UTC')}",
            style_caption
        ))

        doc.build(story)
        buf.seek(0)
        return buf.read()
    
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

# ─────────────────────────── Groq Helpers ───────────────────────────

async def ask_groq_text(messages: list) -> str:
    response = await groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, max_tokens=MAX_TOKENS, temperature=0.7,
    )
    return response.choices[0].message.content

async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".ogg", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as af:
            transcription = await groq_client.audio.transcriptions.create(
                file=(filename, af), model=WHISPER_MODEL, response_format="text",
            )
        return transcription
    finally:
        os.unlink(tmp_path)

async def ask_groq_vision(image_bytes: bytes, prompt: str, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt or "Please analyse this image and explain."}
        ]}
    ]
    response = await groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content

# ─────────────────────────── Conversation Store ─────────────────────

conversation_history: dict[int, list] = {}

def get_history(user_id: int) -> list:
    if user_id not in conversation_history:
        conversation_history[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return conversation_history[user_id]

def add_message(user_id: int, role: str, content):
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > 41:
        conversation_history[user_id] = [history[0]] + history[-40:]

# ─────────────────────────── Handlers ───────────────────────────────

@error_guard("start")
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "Student"
    msg = (
        f"👋 Hello <b>{html.escape(name)}</b>! I'm <b>EduBot</b> — your AI tutor.\n\n"
        f"I can help you with:\n"
        f"📐 <b>Maths & Physics</b> — step-by-step solutions\n"
        f"🧪 <b>Chemistry & Biology</b>\n"
        f"💻 <b>Computer Science</b>\n"
        f"📊 <b>Diagrams</b> generated by Nano Banana AI\n"
        f"📄 <b>PDF notes</b> with rendered equations\n\n"
        f"Send me a <b>text question</b>, an <b>image</b>, or a <b>voice message</b>!\n\n"
        f"Commands:\n"
        f"/pdf — convert last answer to PDF\n"
        f"/diagram &lt;prompt&gt; — generate an AI diagram\n"
        f"/clear — clear chat history\n"
        f"/help — show this message"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


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
    last_answer = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), None)
    if not last_answer:
        await update.message.reply_text("No previous answer found. Ask me something first!")
        return
        
    await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
    question = next((m["content"] for m in reversed(history) if m["role"] == "user"), "Answer")
    title = (question[:60] + "...") if len(question) > 60 else question
    
    diagram_data = []
    for tag in re.findall(r"\[DIAGRAM:\s*(.+?)\]", last_answer):
        result = await generate_diagram(tag)
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
        await update.message.reply_text("Usage: /diagram &lt;prompt&gt;\nExample: /diagram A detailed cross-section of a plant cell", parse_mode=ParseMode.HTML)
        return
        
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    result = await generate_diagram(query)
    if result:
        img_bytes, caption = result
        await update.message.reply_photo(
            photo=InputFile(io.BytesIO(img_bytes), filename="diagram.png"),
            caption=f"📊 <b>{escape_and_format_html(caption[:100])}</b>...\nGenerated by Nano Banana AI",
            parse_mode=ParseMode.HTML,
        )
    else:
        await update.message.reply_text(f"❌ Failed to generate diagram for '{escape_and_format_html(query)}'.", parse_mode=ParseMode.HTML)


@error_guard("text_message")
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_text = update.message.text

    await update.message.chat.send_action(ChatAction.TYPING)

    add_message(uid, "user", user_text)
    history = get_history(uid)

    response_text = await ask_groq_text(history)
    add_message(uid, "assistant", response_text)

    generate_pdf = "[GENERATE_PDF]" in response_text
    clean_response = response_text.replace("[GENERATE_PDF]", "").strip()

    diagram_tags = re.findall(r"\[DIAGRAM:\s*(.+?)\]", clean_response)
    display_text = re.sub(r"\[DIAGRAM:\s*.+?\]", "", clean_response).strip()

    formatted_text = escape_and_format_html(display_text)
    chunks = [formatted_text[i:i+4000] for i in range(0, len(formatted_text), 4000)]
    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)

    for tag in diagram_tags:
        await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
        result = await generate_diagram(tag)
        if result:
            img_bytes, caption = result
            await update.message.reply_photo(
                photo=InputFile(io.BytesIO(img_bytes), filename="diagram.png"),
                caption=f"📊 <b>{escape_and_format_html(caption)}</b> (Nano Banana AI)",
                parse_mode=ParseMode.HTML,
            )

    if generate_pdf:
        await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
        diagram_data = []
        for tag in diagram_tags:
            result = await generate_diagram(tag)
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

    photo = update.message.photo[-1]
    photo_file = await photo.get_file()
    img_bytes = await photo_file.download_as_bytearray()

    prompt = caption if caption else "Analyse this image. If it contains a problem, solve it step by step. If it's a diagram, explain it."
    response_text = await ask_groq_vision(bytes(img_bytes), prompt)
    add_message(uid, "assistant", response_text)

    clean_response = response_text.replace("[GENERATE_PDF]", "").strip()
    diagram_tags = re.findall(r"\[DIAGRAM:\s*(.+?)\]", clean_response)
    display_text = re.sub(r"\[DIAGRAM:\s*.+?\]", "", clean_response).strip()

    formatted_text = escape_and_format_html(display_text)
    chunks = [formatted_text[i:i+4000] for i in range(0, len(formatted_text), 4000)]
    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)

    for tag in diagram_tags:
        result = await generate_diagram(tag)
        if result:
            img_bytes2, caption2 = result
            await update.message.reply_photo(
                photo=InputFile(io.BytesIO(img_bytes2), filename="diagram.png"),
                caption=f"📊 {escape_and_format_html(caption2)} (Nano Banana AI)",
                parse_mode=ParseMode.HTML,
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

    await update.message.reply_text(f"📝 <b>Transcribed:</b> {escape_and_format_html(transcript)}", parse_mode=ParseMode.HTML)
    await update.message.chat.send_action(ChatAction.TYPING)

    add_message(uid, "user", transcript)
    history = get_history(uid)
    response_text = await ask_groq_text(history)
    add_message(uid, "assistant", response_text)

    clean_response = response_text.replace("[GENERATE_PDF]", "").strip()
    diagram_tags = re.findall(r"\[DIAGRAM:\s*(.+?)\]", clean_response)
    display_text = re.sub(r"\[DIAGRAM:\s*.+?\]", "", clean_response).strip()

    formatted_text = escape_and_format_html(display_text)
    chunks = [formatted_text[i:i+4000] for i in range(0, len(formatted_text), 4000)]
    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)

    for tag in diagram_tags:
        result = await generate_diagram(tag)
        if result:
            img_bytes2, caption2 = result
            await update.message.reply_photo(
                photo=InputFile(io.BytesIO(img_bytes2), filename="diagram.png"),
                caption=f"📊 {escape_and_format_html(caption2)} (Nano Banana AI)",
                parse_mode=ParseMode.HTML,
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
        await update.message.chat.send_action(ChatAction.TYPING)
        doc_file = await doc.get_file()
        img_bytes = await doc_file.download_as_bytearray()
        caption = update.message.caption or "Analyse this image."
        uid = update.effective_user.id
        
        response_text = await ask_groq_vision(bytes(img_bytes), caption, mime)
        add_message(uid, "assistant", response_text)
        
        clean = response_text.replace("[GENERATE_PDF]", "").strip()
        formatted_text = escape_and_format_html(clean)
        
        for chunk in [formatted_text[i:i+4000] for i in range(0, len(formatted_text), 4000)]:
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(
            "📁 I can process image files. For PDFs or text files, "
            "please paste the content as text."
        )


async def global_error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
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
