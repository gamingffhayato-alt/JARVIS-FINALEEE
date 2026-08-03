"""
EduBot (JARVIS) - AI-Powered Education Telegram Bot
Uses Groq API (Llama 4 Scout) for AI, supports text/image/audio input,
generates PDFs with rendered math/physics and code syntax highlighting, 
generates Nano Banana diagrams.
Errors are forwarded to a separate error-reporting bot.
"""

import os
import io
import re
import sys
import json
import html
import time
import logging
import asyncio
import tempfile
import threading
import traceback
import base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from collections import OrderedDict
from functools import wraps

# Safely load local .env file if it exists (for local testing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Ignored on Railway where environment variables are injected directly

import httpx
from PIL import Image
import matplotlib

# Pygments for syntax highlighting in PDFs
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import ImageFormatter

# Use Matplotlib's built-in math renderer (requires ZERO external TeX installations)
matplotlib.use("Agg")
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# Thread-Safe Object-Oriented API
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from telegram import Update, InputFile, BotCommand
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from telegram.constants import ParseMode, ChatAction
from groq import AsyncGroq

# Import our new Nano Banana generator
from image_generator import generate_diagram

# ─────────────────────────── Configuration ───────────────────────────

BOT_TOKEN        = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ERROR_BOT_TOKEN  = os.environ.get("TELEGRAM_ERROR_BOT_TOKEN", "")
ERROR_CHAT_ID    = os.environ.get("TELEGRAM_ERROR_CHAT_ID", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")

GROQ_MODEL       = "llama-3.1-8b-instant"
WHISPER_MODEL    = "whisper-large-v3"
MAX_TOKENS       = 4096

ADMIN_ID         = "@johny8901"

# Global HTTP Client for efficient error reporting
http_client = httpx.AsyncClient(timeout=10)

# ─────────────────────────── Logging ────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EduBot")

# ─────────────────────────── Token Bucket Rate Limiter ──────────────

class TokenBucket:
    def __init__(self, max_tokens: int, refill_rate: float, interval: float):
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_rate = refill_rate
        self.interval = interval
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            if elapsed > self.interval:
                tokens_to_add = (elapsed / self.interval) * self.refill_rate
                self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
                self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

user_limiters = {}

def rate_limit(max_tokens=5, refill_rate=1, interval=10):
    def decorator(func):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            user = update.effective_user
            if not user:
                return await func(update, context, *args, **kwargs)

            # Admins bypass rate limit
            if user.username and f"@{user.username}" == ADMIN_ID:
                 return await func(update, context, *args, **kwargs)

            user_id = user.id
            if user_id not in user_limiters:
                user_limiters[user_id] = TokenBucket(max_tokens, refill_rate, interval)
            
            if not user_limiters[user_id].allow_request():
                await update.message.reply_text("⏳ You are sending messages too quickly. Please wait a moment.")
                return 
                
            return await func(update, context, *args, **kwargs)
        return wrapper
    return decorator

# ─────────────────────────── Error Reporter ─────────────────────────

async def report_error(error: Exception, context_info: str = ""):
    tb = traceback.format_exc()
    escaped_tb = html.escape(tb) # Fixes the <module> tag crashing Telegram
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    message = (
        f"🚨 <b>EduBot Error</b>\n"
        f"<b>Time:</b> {timestamp}\n"
        f"<b>Context:</b> {context_info or 'N/A'}\n"
        f"<b>Error:</b> <code>{type(error).__name__}: {html.escape(str(error))[:300]}</code>\n\n"
        f"<b>Traceback:</b>\n<pre>{escaped_tb[:2000]}</pre>"
    )
    
    try:
        response = await http_client.post(
            f"https://api.telegram.org/bot{ERROR_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": ERROR_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
        )
        # Log if Telegram rejects the request (e.g., bot blocked or chat not started)
        if response.status_code != 200:
            logger.error(f"Error Bot failed to send message! Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logger.error(f"Network failure sending error report: {e}")


def error_guard(context_label: str = ""):
    def decorator(func):
        @wraps(func)
        async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            try:
                return await func(update, ctx, *args, **kwargs)
            except Exception as exc:
                user = update.effective_user
                
                # Extract username or fallback to first_name
                if user:
                    username_str = f"@{user.username}" if user.username else f"{user.first_name} (no @username)"
                    user_info = f"{user.id} ({username_str})"
                else:
                    user_info = "?"

                info = f"{context_label} | user={user_info}"
                
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

async def send_html_chunk(message, chunk: str):
    """Attempts to send HTML text. Falls back to plain text if Telegram rejects parsing."""
    try:
        await message.reply_text(chunk, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.warning(f"HTML parsing failed, falling back to plain text: {e}")
        plain_chunk = re.sub(r'<[^>]+>', '', chunk)
        plain_chunk = html.unescape(plain_chunk)
        await message.reply_text(plain_chunk)

def sanitize_latex_for_pdf(text: str) -> str:
    text = re.sub(r"\\\\([a-zA-Z])", r"\\\1", text)
    text = text.replace(r"\$", "$")
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    text = re.sub(r"\\begin\{[a-zA-Z*]+\}", "", text)
    text = re.sub(r"\\end\{[a-zA-Z*]+\}", "", text)
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\text\{(.*?)\}", r"\\mathrm{\1}", text)
    return text

# ─────────────────────────── Groq Client ────────────────────────────

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# Use raw string r"""...""" to fix \Omega escape sequence warning
SYSTEM_PROMPT = r"""You are EduBot — an expert AI tutor.

CRITICAL MATH FORMATTING & REASONING RULES:
1. NEVER escape dollar signs. Write $5 \Omega$, NOT \$5 \Omega\$.
2. ALL equations, numbers, and variables MUST be wrapped in $...$ (inline) or $$...$$ (display math). 
3. NEVER write naked equations. Every single equation must have a delimiter.
4. DO NOT use complex LaTeX environments like \begin{vmatrix}, \begin{matrix}, \begin{array}, \begin{align}, etc.
5. For cross products and determinants, DO NOT draw a matrix. Write the algebraic expansion linearly. Example: $\vec{A} \times \vec{B} = (A_y B_z - A_z B_y)\hat{i} - ...$
6. DO NOT use \boxed{} or \text{} as they break the renderer. Use simple, basic LaTeX equations. Use \mathrm{} instead of \text{}.
7. Think step-by-step and DOUBLE-CHECK algebraic manipulations. ALWAYS convert units to standard SI (e.g., mA to A) before solving equations.
8. To generate a PDF, append: [GENERATE_PDF]
"""

# ─────────────────────────── Math & Code Renderers ──────────────────

def render_math_to_image(latex: str, dpi: int = 150, inline: bool = False) -> Optional[tuple[bytes, float, float]]:
    try:
        fig = Figure(figsize=(0.01, 0.01), facecolor="white")
        canvas = FigureCanvasAgg(fig)
        
        expr = latex.strip()
        if not expr.startswith("$"):
            expr = f"${expr}$"
            
        fontsize = 12 if inline else 16
        fig.text(0, 0, expr, fontsize=fontsize, color="black")
        
        canvas.draw()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        buf.seek(0)
        
        img_bytes = buf.read()
        with Image.open(io.BytesIO(img_bytes)) as img:
            w_px, h_px = img.size
            scale = 1.05 if inline else 1.2
            w_pt = (w_px / dpi) * 72 * scale
            h_pt = (h_px / dpi) * 72 * scale
            
        return img_bytes, w_pt, h_pt
    except Exception as e:
        logger.warning(f"Mathtext render failed for '{latex[:60]}': {e}")
        return None

def render_code_to_image(code_text: str, language: str = None) -> Optional[tuple[bytes, float, float]]:
    """Renders a code block to a syntax-highlighted image using Pygments."""
    try:
        try:
            lexer = get_lexer_by_name(language, stripall=True) if language else guess_lexer(code_text)
        except Exception:
            lexer = get_lexer_by_name("text", stripall=True)
            
        formatter = ImageFormatter(font_size=16, line_numbers=False, style="monokai")
        img_bytes = highlight(code_text, lexer, formatter)
        
        with Image.open(io.BytesIO(img_bytes)) as img:
            w_px, h_px = img.size
            # Convert pixels to points (assuming roughly 96 DPI output from Pygments)
            w_pt = w_px * 0.75
            h_pt = h_px * 0.75
        return img_bytes, w_pt, h_pt
    except Exception as e:
        logger.warning(f"Pygments render failed: {e}")
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
        
        # Extract Multi-line code blocks before splitting by newline
        code_store = []
        def extract_code(m):
            lang = m.group(1)
            code = m.group(2)
            code_store.append((lang, code))
            return f"\n__CODE_{len(code_store)-1}__\n"
            
        clean_content = re.sub(r"```(\w*)\n(.*?)```", extract_code, clean_content, flags=re.DOTALL)

        parts = extract_latex_blocks(clean_content)
        
        for i, part in enumerate(parts):
            if i % 2 == 1:
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
                lines = part.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        story.append(Spacer(1, 4))
                        continue
                    
                    # Handle Pygments Syntax Highlighted Blocks
                    code_match = re.match(r"__CODE_(\d+)__", line)
                    if code_match:
                        idx = int(code_match.group(1))
                        lang, code = code_store[idx]
                        res = render_code_to_image(code, lang)
                        if res:
                            img_b, w, h = res
                            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                            tmp.write(img_b)
                            tmp.flush()
                            temp_files.append(tmp.name)
                            tmp.close()
                            
                            # Constrain width to A4 page margins if it's too wide
                            max_w = A4[0] - (4 * cm) 
                            if w > max_w:
                                ratio = max_w / w
                                w = max_w
                                h = h * ratio

                            rl_img = RLImage(tmp.name, width=w, height=h)
                            rl_img.hAlign = "CENTER"
                            story.append(rl_img)
                            story.append(Spacer(1, 8))
                        else:
                            # Fallback if Pygments fails
                            story.append(Paragraph(f"<font name='Courier'>{html.escape(code)}</font>", style_code))
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
            f"Generated by EduBot on {datetime.now(timezone.utc).strftime('%B %d, %Y at %H:%M UTC')}",
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

class MaxSizeDict(OrderedDict):
    def __init__(self, max_size=500, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

conversation_history = MaxSizeDict(max_size=500)

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
        f"/status — check hosting environment\n"
        f"/reset — clear memory & start fresh\n"
        f"/help — contact the admin"
    )
    await send_html_chunk(update.message, msg)


@error_guard("help")
async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"🆘 Need help, have feedback, or want to report an issue?\n\n"
        f"<b>Contact the Admin:</b> {ADMIN_ID}"
    )
    await send_html_chunk(update.message, msg)


@error_guard("status_command")
@rate_limit(max_tokens=3, refill_rate=1, interval=20)
async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # Check for variables that Railway automatically injects
    is_railway = os.environ.get("RAILWAY_ENVIRONMENT") is not None
    is_cloud = os.environ.get("PORT") is not None

    if is_railway:
        location = "☁️ Railway Cloud Container"
    elif is_cloud:
        location = "☁️ Generic Cloud Hosting"
    else:
        location = "💻 Local Computer (Development Mode)"

    msg = (
        f"🤖 <b>EduBot Status</b>\n"
        f"<b>Host Environment:</b> {location}\n"
    )
    
    await send_html_chunk(update.message, msg)


@error_guard("test_crash")
async def cmd_crash(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Hidden command to test if the Error Analyser bot is working properly."""
    await update.message.reply_text("🚨 Initiating test crash! Check the Error Analyser bot.")
    1 / 0  # This purposefully raises a ZeroDivisionError to test the error reporter


@error_guard("reset")
async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    conversation_history.pop(uid, None)
    await update.message.reply_text("🔄 Memory cleared. Let's start fresh!")


@error_guard("pdf_command")
@rate_limit(max_tokens=3, refill_rate=1, interval=15)
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
    
    # Send PDF without diagrams (since conversational tags are disabled)
    pdf_bytes = build_pdf(title, last_answer, [])
    await update.message.reply_document(
        document=InputFile(io.BytesIO(pdf_bytes), filename="EduBot_Notes.pdf"),
        caption="📄 Here are your notes as a PDF!"
    )


@error_guard("diagram_command")
@rate_limit(max_tokens=2, refill_rate=1, interval=30)
async def cmd_diagram(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = " ".join(ctx.args) if ctx.args else ""
    if not query:
        await send_html_chunk(update.message, "Usage: /diagram &lt;prompt&gt;\nExample: /diagram A detailed cross-section of a plant cell")
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
        await send_html_chunk(update.message, f"❌ Failed to generate diagram for '{escape_and_format_html(query)}'.")


@error_guard("text_message")
@rate_limit(max_tokens=5, refill_rate=1, interval=10)
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

    formatted_text = escape_and_format_html(clean_response)
    chunks = [formatted_text[i:i+4000] for i in range(0, len(formatted_text), 4000)]
    for chunk in chunks:
        await send_html_chunk(update.message, chunk)

    if generate_pdf:
        await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
        title = (user_text[:60] + "...") if len(user_text) > 60 else user_text
        pdf_bytes = build_pdf(title, clean_response, [])
        await update.message.reply_document(
            document=InputFile(io.BytesIO(pdf_bytes), filename="EduBot_Notes.pdf"),
            caption="📄 Your PDF notes are ready!"
        )


@error_guard("photo_message")
@rate_limit(max_tokens=3, refill_rate=1, interval=20)
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

    formatted_text = escape_and_format_html(clean_response)
    chunks = [formatted_text[i:i+4000] for i in range(0, len(formatted_text), 4000)]
    for chunk in chunks:
        await send_html_chunk(update.message, chunk)

    if "[GENERATE_PDF]" in response_text:
        title = (caption[:60] + "...") if caption else "Image Analysis"
        pdf_bytes = build_pdf(title, clean_response, [])
        await update.message.reply_document(
            document=InputFile(io.BytesIO(pdf_bytes), filename="EduBot_Notes.pdf"),
            caption="📄 PDF notes generated!"
        )


@error_guard("audio_message")
@rate_limit(max_tokens=3, refill_rate=1, interval=20)
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

    await send_html_chunk(update.message, f"📝 <b>Transcribed:</b> {escape_and_format_html(transcript)}")
    await update.message.chat.send_action(ChatAction.TYPING)

    add_message(uid, "user", transcript)
    history = get_history(uid)
    response_text = await ask_groq_text(history)
    add_message(uid, "assistant", response_text)

    clean_response = response_text.replace("[GENERATE_PDF]", "").strip()

    formatted_text = escape_and_format_html(clean_response)
    chunks = [formatted_text[i:i+4000] for i in range(0, len(formatted_text), 4000)]
    for chunk in chunks:
        await send_html_chunk(update.message, chunk)

    if "[GENERATE_PDF]" in response_text:
        title = (transcript[:60] + "...") if len(transcript) > 60 else transcript
        pdf_bytes = build_pdf(title, clean_response, [])
        await update.message.reply_document(
            document=InputFile(io.BytesIO(pdf_bytes), filename="EduBot_Notes.pdf"),
            caption="📄 PDF notes ready!"
        )


@error_guard("document_message")
@rate_limit(max_tokens=3, refill_rate=1, interval=20)
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
            await send_html_chunk(update.message, chunk)
    else:
        await update.message.reply_text(
            "📁 I can process image files. For PDFs or text files, "
            "please paste the content as text."
        )


async def global_error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    exc = ctx.error
    logger.error(f"Unhandled error: {exc}")
    await report_error(exc, f"global_error_handler | update={type(update).__name__}")


# ─────────────────────────── Bot Initialization ─────────────────────────

async def setup_bot_commands(application: Application):
    """Sets up the Telegram bot menu commands to match the JARVIS theme."""
    commands = [
        BotCommand("start", "Wake up JARVIS ⚡️"),
        BotCommand("reset", "Clear memory & start fresh 🔄"),
        BotCommand("status", "Check host environment ☁️"),
        BotCommand("help", "Contact the Admin 🆘"),
        BotCommand("pdf", "Convert last answer to PDF 📄"),
        BotCommand("diagram", "Generate an AI diagram 📊")
    ]
    await application.bot.set_my_commands(commands)


# ─────────────────────────── Main ───────────────────────────────────

def main():
    logger.info("Starting EduBot...")
    
    # -------------------------------------------------------------------
    # CRITICAL FIX for Railway / Multithreaded Environments
    # Force the creation of a new event loop and set it for the main thread
    # -------------------------------------------------------------------
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # The post_init step ensures your custom menu applies when the bot boots up
    app = Application.builder().token(BOT_TOKEN).post_init(setup_bot_commands).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))  
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("crash", cmd_crash)) # Hidden test command
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
