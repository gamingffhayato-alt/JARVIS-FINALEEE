# 🎓 EduBot — AI Education Telegram Bot

Powered by **Groq API + Llama 4 Scout**, this bot is a full-featured AI tutor
for Telegram supporting text, images, voice notes, PDF generation, and Wikimedia diagrams.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 AI Model | Llama 4 Scout via Groq (fast & free tier available) |
| 📷 Image Input | Send photos of problems/diagrams — bot analyses them |
| 🎙️ Voice Input | Send voice notes — transcribed via Groq Whisper |
| 📄 PDF Export | `/pdf` generates a styled PDF with rendered math equations |
| 📊 Diagrams | `/diagram <topic>` fetches from Wikimedia Commons |
| ➗ Math Render | `$$...$$` blocks rendered as images inside PDFs |
| 🚨 Error Bot | All errors auto-forwarded to a separate error-reporting bot |
| 💬 Memory | Per-user conversation history (last 20 exchanges) |

---

## 🚀 Setup

### 1. Clone & install

```bash
git clone <your-repo>
cd edu_bot
pip install -r requirements.txt
```

### 2. Create your bots

1. **Main bot**: Message [@BotFather](https://t.me/BotFather) → `/newbot`
2. **Error bot**: Create another bot the same way

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your tokens
```

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Main EduBot token |
| `TELEGRAM_ERROR_BOT_TOKEN` | Error-reporting bot token |
| `TELEGRAM_ERROR_CHAT_ID` | Your Telegram chat ID (get from [@userinfobot](https://t.me/userinfobot)) |
| `GROQ_API_KEY` | From [console.groq.com](https://console.groq.com) |

### 4. Run

```bash
export $(cat .env | xargs)
python bot.py
```

Or with dotenv:

```bash
pip install python-dotenv
# Add at top of bot.py:  from dotenv import load_dotenv; load_dotenv()
python bot.py
```

---

## 🤖 Commands

| Command | Description |
|---|---|
| `/start` | Welcome message & feature list |
| `/help` | Same as start |
| `/pdf` | Convert last AI answer to PDF |
| `/diagram <topic>` | Fetch diagram from Wikimedia |
| `/clear` | Clear your conversation history |

---

## 📄 PDF Generation

- Ask any question and append **"generate a PDF"** or **"make me PDF notes"**
- The AI will respond normally AND auto-generate a PDF
- `$$...$$` math blocks are rendered as images using matplotlib
- Diagrams from Wikimedia are embedded at the end

---

## 🚨 Error Reporting

Every exception is automatically caught and forwarded to your error bot with:
- Timestamp
- User context
- Exception type and message
- Full stack traceback

---

## 🏗️ Architecture

```
User Message
    │
    ├── Text ──────────► Groq Llama 4 Scout (chat)
    ├── Photo ─────────► Groq Llama 4 Scout (vision)
    ├── Voice/Audio ───► Groq Whisper → Llama 4 Scout
    └── Document ──────► (image) → Vision / (other) → prompt
                              │
                         Response
                              │
                    ┌─────────┴──────────┐
                    │                    │
               Text Reply         [DIAGRAM: X] tag
                                         │
                                  Wikimedia API
                                         │
                                   Photo Reply
                                  
              [GENERATE_PDF] trigger
                    │
             LaTeX → matplotlib PNG
                    │
              reportlab PDF
                    │
             Document Reply
```

---

## 🔧 Customisation

- **Model**: Change `GROQ_MODEL` constant (e.g. `llama-3.3-70b-versatile`)
- **System Prompt**: Edit `SYSTEM_PROMPT` to specialise the tutor
- **PDF Style**: Colours and fonts in `build_pdf()` use ReportLab styles
- **History Length**: Adjust the `41` limit in `add_message()`

---

## 📦 Dependencies

- `python-telegram-bot` — Telegram Bot API
- `groq` — Groq SDK (Llama 4 Scout + Whisper)
- `matplotlib` — LaTeX math rendering
- `reportlab` — PDF generation
- `Pillow` — Image processing
- `httpx` / `aiohttp` — Async HTTP (Wikimedia API)
