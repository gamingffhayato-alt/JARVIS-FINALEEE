"""
image_generator.py  –  Wikipedia-Powered Scientific Diagram Generator
=======================================================================
Pipeline:
  1. Query Wikipedia REST API for the topic → fetch summary + images.
  2. Score & download the best diagram/illustration image from Wikipedia.
  3. If no suitable Wikipedia image found, fall back to Wikimedia Commons search.
  4. Fetch the article summary text for smart label generation via Groq.
  5. Groq Vision (Llama) analyses the downloaded image and returns label coordinates.
  6. Pillow composites the labels onto the image with a clean annotation style.
  7. Returns (annotated_image_bytes, caption_string) to the caller.

No Cloudflare / FLUX dependency — 100% Wikipedia + Groq + Pillow.
"""

from __future__ import annotations

import os
import re
import json
import base64
import logging
import asyncio
from io import BytesIO
from typing import Optional
from urllib.parse import quote, urljoin

import httpx
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("EduBot.ImageGen")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"
GROQ_TEXT_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

WIKIPEDIA_API   = "https://en.wikipedia.org/api/rest_v1"
WIKIPEDIA_FULL  = "https://en.wikipedia.org/w/api.php"
COMMONS_API     = "https://commons.wikimedia.org/w/api.php"

# Image extensions we accept
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp"}

# These words in filenames strongly suggest it's a diagram / illustration
DIAGRAM_KEYWORDS = {
    "diagram", "anatomy", "structure", "cross", "section", "cell",
    "organ", "system", "cycle", "process", "schematic", "model",
    "illustration", "labeled", "labelled", "chart", "map", "figure",
}

# Filenames that are almost certainly logos, icons, or useless images
SKIP_KEYWORDS = {
    "logo", "icon", "flag", "emblem", "seal", "coat", "portrait",
    "photo", "map_of", "location", "thumbnail", "stub",
}

# HTTP headers to look like a polite bot
BOT_HEADERS = {
    "User-Agent": "EduBot/2.0 (Telegram educational bot; +https://github.com/yourusername/edubot)",
    "Accept": "application/json",
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Wikipedia article + image search
# ─────────────────────────────────────────────────────────────────────────────

async def _wikipedia_search_title(client: httpx.AsyncClient, query: str) -> Optional[str]:
    """Return the best-matching Wikipedia article title for the query."""
    params = {
        "action":   "query",
        "list":     "search",
        "srsearch": query,
        "srlimit":  5,
        "format":   "json",
    }
    try:
        resp = await client.get(WIKIPEDIA_FULL, params=params, headers=BOT_HEADERS, timeout=10.0)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if results:
            return results[0]["title"]
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
    return None


async def _wikipedia_summary(client: httpx.AsyncClient, title: str) -> dict:
    """Fetch the Wikipedia REST summary for an article title."""
    encoded = quote(title.replace(" ", "_"))
    url = f"{WIKIPEDIA_API}/page/summary/{encoded}"
    try:
        resp = await client.get(url, headers=BOT_HEADERS, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Wikipedia summary fetch failed for '{title}': {e}")
        return {}


async def _wikipedia_images(client: httpx.AsyncClient, title: str) -> list[dict]:
    """
    Return a list of image metadata dicts for a Wikipedia article.
    Each dict has keys: title, url, width, height, mime.
    """
    params = {
        "action":    "query",
        "titles":    title,
        "prop":      "images",
        "imlimit":   50,
        "format":    "json",
    }
    try:
        resp = await client.get(WIKIPEDIA_FULL, params=params, headers=BOT_HEADERS, timeout=10.0)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        images = []
        for page in pages.values():
            for img in page.get("images", []):
                images.append(img["title"])  # e.g. "File:Human_heart.svg"
        return images
    except Exception as e:
        logger.warning(f"Wikipedia image list failed: {e}")
        return []


async def _imageinfo(client: httpx.AsyncClient, file_titles: list[str]) -> list[dict]:
    """Batch-fetch imageinfo (URL, size, mime) for a list of File: titles."""
    if not file_titles:
        return []
    params = {
        "action":     "query",
        "titles":     "|".join(file_titles[:50]),
        "prop":       "imageinfo",
        "iiprop":     "url|size|mime",
        "format":     "json",
    }
    try:
        resp = await client.get(WIKIPEDIA_FULL, params=params, headers=BOT_HEADERS, timeout=10.0)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        results = []
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            if info.get("url"):
                results.append({
                    "title": page.get("title", ""),
                    "url":   info["url"],
                    "width":  info.get("width",  0),
                    "height": info.get("height", 0),
                    "mime":   info.get("mime",   ""),
                })
        return results
    except Exception as e:
        logger.warning(f"imageinfo fetch failed: {e}")
        return []


def _score_image(img: dict) -> float:
    """
    Score an image dict for how likely it is to be a useful scientific diagram.
    Higher = better.
    """
    score = 0.0
    name  = img.get("title", "").lower()
    mime  = img.get("mime",  "").lower()
    w     = img.get("width",  0)
    h     = img.get("height", 0)

    # Skip SVG — rendering would require cairosvg; stick to raster
    if "svg" in mime or name.endswith(".svg"):
        return -1.0

    # Must be a raster image
    if not any(ext in mime for ext in ("jpeg", "png", "webp")):
        return -1.0

    # Hard-skip logos, flags, portraits
    if any(kw in name for kw in SKIP_KEYWORDS):
        return -1.0

    # Reward diagram keywords in filename
    for kw in DIAGRAM_KEYWORDS:
        if kw in name:
            score += 5.0

    # Reward landscape or squarish images (diagrams tend not to be portraits)
    if w > 0 and h > 0:
        aspect = w / h
        if 0.6 <= aspect <= 2.2:
            score += 3.0
        # Reward larger images (more detail)
        score += min(w * h / 1_000_000, 4.0)   # up to 4 pts for large images

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Wikimedia Commons fallback
# ─────────────────────────────────────────────────────────────────────────────

async def _commons_search(client: httpx.AsyncClient, query: str) -> list[dict]:
    """Search Wikimedia Commons for diagram images."""
    params = {
        "action":      "query",
        "generator":   "search",
        "gsrsearch":   f"filetype:bitmap {query} diagram",
        "gsrnamespace": 6,   # File namespace
        "gsrlimit":    10,
        "prop":        "imageinfo",
        "iiprop":      "url|size|mime",
        "format":      "json",
    }
    try:
        resp = await client.get(COMMONS_API, params=params, headers=BOT_HEADERS, timeout=10.0)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        results = []
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            if info.get("url"):
                results.append({
                    "title": page.get("title", ""),
                    "url":   info["url"],
                    "width":  info.get("width",  0),
                    "height": info.get("height", 0),
                    "mime":   info.get("mime",   ""),
                })
        return results
    except Exception as e:
        logger.warning(f"Commons search failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Download image bytes
# ─────────────────────────────────────────────────────────────────────────────

async def _download_image(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    """Download raw image bytes from a URL."""
    try:
        resp = await client.get(url, headers=BOT_HEADERS, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning(f"Image download failed ({url}): {e}")
        return None


def _ensure_rgb_png(image_bytes: bytes) -> bytes:
    """Convert image to RGB PNG for consistent downstream processing."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Groq: generate smart labels from Wikipedia summary
# ─────────────────────────────────────────────────────────────────────────────

async def _groq_label_coordinates(
    client: httpx.AsyncClient,
    image_bytes: bytes,
    topic: str,
    wiki_summary: str,
) -> list[dict]:
    """
    Ask Groq Vision to identify labellable parts in the image.
    Returns list of {"label": str, "x": float, "y": float}.
    """
    b64 = base64.b64encode(image_bytes).decode()
    data_uri = f"data:image/png;base64,{b64}"

    context_hint = ""
    if wiki_summary:
        # Give the model a short snippet to ground its labels in real terminology
        snippet = wiki_summary[:600].replace("\n", " ")
        context_hint = (
            f"\n\nHere is a short Wikipedia description of the topic for context:\n{snippet}"
        )

    prompt = (
        f"You are analysing a scientific diagram of: {topic}.{context_hint}\n\n"
        "Look carefully at the actual image provided. Identify 5–9 distinct, clearly visible "
        "structural parts or components. Only label things you can actually see in this specific image.\n\n"
        "Reply ONLY with a valid JSON array. No markdown, no code fences, no extra text.\n"
        "Each element must be exactly: "
        "{\"label\": \"part name\", \"x\": 0.0, \"y\": 0.0}\n"
        "x and y are decimal fractions (0.0–1.0) pointing at the CENTER of that part. "
        "x=0 is the left edge, x=1 is the right edge, y=0 is the top, y=1 is the bottom.\n"
        "Use concise scientific names (e.g. 'Mitochondria', 'Cell Wall', 'Nucleus')."
    )

    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        }],
        "temperature": 0.1,
        "max_tokens": 800,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=40.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        labels = json.loads(raw)
        # Validate structure
        valid = []
        for item in labels:
            if (
                isinstance(item, dict)
                and "label" in item
                and "x" in item and "y" in item
                and 0.0 <= float(item["x"]) <= 1.0
                and 0.0 <= float(item["y"]) <= 1.0
            ):
                valid.append({
                    "label": str(item["label"]).strip(),
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                })
        logger.info(f"Groq returned {len(valid)} valid labels.")
        return valid
    except Exception as e:
        logger.error(f"Groq label extraction failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Pillow: draw labels onto the image
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(size: int):
    """Load a TTF font if available, else fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_labels(image_bytes: bytes, labels: list[dict]) -> bytes:
    """
    Composite annotation labels onto the image.

    Style:
    - White rounded-rect pill background with a coloured border.
    - A thin line from the pill to the actual point on the diagram.
    - Alternating colours per label for easy distinction.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    W, H = img.size

    # Scale font to image size
    font_size = max(14, min(22, W // 40))
    font      = _load_font(font_size)
    padding   = int(font_size * 0.55)

    # Colour palette (border / dot colour per label)
    PALETTE = [
        "#E53935",  # red
        "#1E88E5",  # blue
        "#43A047",  # green
        "#FB8C00",  # orange
        "#8E24AA",  # purple
        "#00ACC1",  # cyan
        "#F4511E",  # deep orange
        "#3949AB",  # indigo
        "#00897B",  # teal
    ]

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for idx, item in enumerate(labels):
        label = item["label"]
        px    = int(item["x"] * W)
        py    = int(item["y"] * H)
        color = PALETTE[idx % len(PALETTE)]

        # Measure text
        bbox = draw.textbbox((0, 0), label, font=font)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]

        # Pill box dimensions
        box_w = tw + padding * 2
        box_h = th + padding * 2

        # Position the pill so it doesn't hug the target point:
        # offset to the right or left depending on which half of the image we're in
        offset_x = int(font_size * 1.5)
        offset_y = -box_h // 2

        if item["x"] > 0.6:          # right-side part → put label to the left
            lx = px - offset_x - box_w
        else:                         # left/centre part → put label to the right
            lx = px + offset_x

        ly = py + offset_y

        # Clamp pill to image bounds
        lx = max(4, min(W - box_w - 4, lx))
        ly = max(4, min(H - box_h - 4, ly))

        # Leader line: from pill centre to the actual point
        pill_cx = lx + box_w // 2
        pill_cy = ly + box_h // 2
        draw.line([(pill_cx, pill_cy), (px, py)], fill=color + "CC", width=max(1, font_size // 10))

        # Dot at the actual point
        dot_r = max(4, font_size // 4)
        draw.ellipse(
            [px - dot_r, py - dot_r, px + dot_r, py + dot_r],
            fill=color + "EE", outline="white", width=2
        )

        # Pill background (white, semi-transparent)
        draw.rounded_rectangle(
            [lx, ly, lx + box_w, ly + box_h],
            radius=box_h // 2,
            fill=(255, 255, 255, 230),
            outline=color + "FF",
            width=max(1, font_size // 12),
        )

        # Label text
        draw.text(
            (lx + padding - bbox[0], ly + padding - bbox[1]),
            label,
            font=font,
            fill=color,
        )

    # Composite and return as PNG bytes
    result = Image.alpha_composite(img, overlay).convert("RGB")
    out = BytesIO()
    result.save(out, format="PNG", optimize=True)
    return out.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def generate_diagram(topic: str) -> Optional[tuple[bytes, str]]:
    """
    Generate an annotated scientific diagram for `topic`.

    Returns:
        (png_bytes, caption_string)  on success
        None                         on failure
    """
    logger.info(f"[generate_diagram] topic='{topic}'")

    async with httpx.AsyncClient() as client:

        # ── 1. Find the best Wikipedia article title ──────────────────────
        title = await _wikipedia_search_title(client, topic)
        if not title:
            logger.warning("No Wikipedia article found — aborting.")
            return None
        logger.info(f"Wikipedia article: '{title}'")

        # ── 2. Fetch summary text (for label context) ─────────────────────
        summary_data = await _wikipedia_summary(client, title)
        wiki_summary = summary_data.get("extract", "")
        page_url     = summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")

        # ── 3a. Get image list from the article ───────────────────────────
        image_titles = await _wikipedia_images(client, title)

        best_image: Optional[dict] = None

        if image_titles:
            image_info = await _imageinfo(client, image_titles)
            scored = [(img, _score_image(img)) for img in image_info]
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored and scored[0][1] > 0:
                best_image = scored[0][0]
                logger.info(
                    f"Best Wikipedia image: {best_image['title']}  "
                    f"(score={scored[0][1]:.1f})"
                )

        # ── 3b. Commons fallback if no good Wikipedia image ───────────────
        if not best_image:
            logger.info("Falling back to Wikimedia Commons search.")
            commons_imgs = await _commons_search(client, topic)
            if commons_imgs:
                scored = [(img, _score_image(img)) for img in commons_imgs]
                scored.sort(key=lambda x: x[1], reverse=True)
                if scored and scored[0][1] > 0:
                    best_image = scored[0][0]
                    logger.info(f"Best Commons image: {best_image['title']}")

        if not best_image:
            logger.warning("No suitable image found on Wikipedia or Commons.")
            return None

        # ── 4. Download the image ─────────────────────────────────────────
        raw_bytes = await _download_image(client, best_image["url"])
        if not raw_bytes:
            return None

        # Normalise to RGB PNG
        try:
            image_bytes = _ensure_rgb_png(raw_bytes)
        except Exception as e:
            logger.error(f"Image normalisation failed: {e}")
            return None

        # ── 5. Groq: extract label coordinates from the actual image ──────
        labels = await _groq_label_coordinates(
            client, image_bytes, topic, wiki_summary
        )

        # ── 6. Draw labels onto the image ─────────────────────────────────
        if labels:
            try:
                annotated_bytes = _draw_labels(image_bytes, labels)
            except Exception as e:
                logger.error(f"Label drawing failed: {e}")
                annotated_bytes = image_bytes   # send un-annotated rather than nothing
        else:
            logger.warning("No labels returned — sending image without annotations.")
            annotated_bytes = image_bytes

        # ── 7. Build caption ──────────────────────────────────────────────
        first_sentence = wiki_summary.split(".")[0] if wiki_summary else ""
        caption_parts  = [title]
        if first_sentence:
            caption_parts.append(first_sentence + ".")
        if page_url:
            caption_parts.append(f"Source: Wikipedia — {page_url}")
        caption = "\n".join(caption_parts)

        logger.info(f"[generate_diagram] Done. Image size: {len(annotated_bytes)//1024} KB")
        return annotated_bytes, caption


# ─────────────────────────────────────────────────────────────────────────────
# Quick local test (python image_generator.py "plant cell")
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    topic = " ".join(sys.argv[1:]) or "plant cell"
    print(f"Testing with topic: '{topic}'")

    result = asyncio.run(generate_diagram(topic))
    if result:
        img_bytes, caption = result
        out_path = "test_diagram.png"
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"✅ Saved to {out_path}")
        print(f"Caption:\n{caption}")
    else:
        print("❌ Failed to generate diagram.")
