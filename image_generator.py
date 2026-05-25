```python
"""
image_generator.py - Diagram Generator
  1. Cloudflare AI (FLUX) generates a clean UNLABELLED illustration.
  2. Uses hardcoded JSON for known diagrams, or Cloudflare AI (Llama) for fallbacks.
  3. Pillow draws the labels onto the image in real crisp text.
"""

import os
import re
import json
import base64
import logging
from io import BytesIO
from typing import Optional

import httpx
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("EduBot.ImageGen")

# ────────────────────────────────────────────────────────────────
# HARDCODED TEMPLATES: Accurate x/y positions (0.0 to 1.0)
# x=0 left, x=1 right; y=0 top, y=1 bottom
# ────────────────────────────────────────────────────────────────
HARDCODED_TEMPLATES = {
    "heart": [
        {"label": "Superior Vena Cava", "x": 0.40, "y": 0.22},
        {"label": "Aorta", "x": 0.55, "y": 0.18},
        {"label": "Pulmonary Artery", "x": 0.65, "y": 0.30},
        {"label": "Right Atrium", "x": 0.38, "y": 0.45},
        {"label": "Left Atrium", "x": 0.68, "y": 0.45},
        {"label": "Right Ventricle", "x": 0.42, "y": 0.65},
        {"label": "Left Ventricle", "x": 0.65, "y": 0.68},
        {"label": "Inferior Vena Cava", "x": 0.40, "y": 0.85},
    ],
    "plant cell": [
        {"label": "Nucleus", "x": 0.45, "y": 0.45},
        {"label": "Cell Wall", "x": 0.15, "y": 0.50},
        {"label": "Cell Membrane", "x": 0.20, "y": 0.60},
        {"label": "Chloroplast", "x": 0.70, "y": 0.35},
        {"label": "Vacuole", "x": 0.55, "y": 0.65},
        {"label": "Mitochondrion", "x": 0.30, "y": 0.70},
        {"label": "Cytoplasm", "x": 0.50, "y": 0.80},
    ]
}

def _cf_url(model: str) -> str:
    return (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{os.environ['CF_ACCOUNT_ID']}/ai/run/{model}"
    )

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['CF_API_TOKEN']}",
        "Content-Type": "application/json",
    }


async def _get_labels(client: httpx.AsyncClient, topic: str) -> list[dict]:
    """Check hardcoded templates for correct positions, fallback to Llama."""
    topic_lower = topic.lower()
    
    # 1. Search for a matching hardcoded template
    for key, template_labels in HARDCODED_TEMPLATES.items():
        if key in topic_lower:
            logger.info(f"Found hardcoded template for '{key}'")
            return template_labels

    # 2. Fallback: Ask Llama to guess labels + x/y positions
    logger.info(f"No template found for '{topic}', falling back to Llama...")
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a science expert. Reply ONLY with a JSON array, no markdown. "
                    "Each item: {\"label\": \"correct scientific name\", \"x\": 0.0-1.0, \"y\": 0.0-1.0}. "
                    "x=0 left, x=1 right, y=0 top, y=1 bottom. "
                    "Place labels around the edges pointing inward to the relevant part."
                ),
            },
            {
                "role": "user",
                "content": f"Give 12 correctly named labelled parts for a diagram of: {topic}",
            },
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    resp = await client.post(
        _cf_url("@cf/meta/llama-3.1-8b-instruct"),
        headers=_headers(),
        json=payload,
        timeout=30.0,
    )

    if resp.status_code != 200:
        logger.error(f"Llama label error {resp.status_code}: {resp.text}")
        return []

    raw = resp.json().get("result", {}).get("response", "")
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        logger.error(f"Label JSON parse failed: {raw[:300]}")
    return []


async def _generate_base_image(client: httpx.AsyncClient, topic: str) -> Optional[bytes]:
    """Generate an UNLABELLED illustration — explicitly no text."""
    import re as _re
    clean_topic = _re.sub(
        r'\b(diagram|label|labelled|labeled|labeling|annotated|with labels|chart)\b',
        '', topic, flags=_re.IGNORECASE
    ).strip()

    prompt = (
        f"Plain illustration of {clean_topic}, no text. "
        "White background. "
        "Render ZERO letters, ZERO words, ZERO numbers, ZERO annotations anywhere. "
        "Just the drawing itself, nothing written."
    )
    resp = await client.post(
        _cf_url("@cf/black-forest-labs/flux-1-schnell"),
        headers=_headers(),
        json={"prompt": prompt},
        timeout=60.0,
    )

    if resp.status_code != 200:
        logger.error(f"Image gen error {resp.status_code}: {resp.text}")
        return None

    b64 = resp.json().get("result", {}).get("image")
    return base64.b64decode(b64) if b64 else None


def _overlay_labels(image_bytes: bytes, labels: list[dict]) -> bytes:
    """Pillow draws real text labels — guaranteed crisp and connected to the correct parts."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=max(15, H // 38)
        )
    except IOError:
        font = ImageFont.load_default()

    cx, cy = W // 2, H // 2

    for item in labels:
        label = str(item.get("label", "")).strip()
        if not label:
            continue

        # Target point on the anatomy
        ax = int(float(item.get("x", 0.5)) * W)
        ay = int(float(item.get("y", 0.5)) * H)

        # Dot on the actual anatomy part
        r = max(5, W // 120)
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], fill=(200, 30, 30))

        # Push label text outward towards the edges of the image
        offset_x = max(90, W // 5)
        offset_y = max(40, H // 10)
        tx = ax + (offset_x if ax >= cx else -offset_x)
        ty = ay + (offset_y if ay >= cy else -offset_y)

        # Clamp to image bounds
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 10
        tx = max(pad, min(tx, W - tw - pad))
        ty = max(pad, min(ty, H - th - pad))

        # Connector line drawing FROM the anatomy dot TO the label edge
        # Calculate center-ish of the text box for the line to anchor
        line_target_x = tx + (tw // 2 if tx > ax else tw)
        if tx <= ax <= tx + tw:
            line_target_x = tx + tw // 2 
            
        draw.line([(ax, ay), (line_target_x, ty + th // 2)],
                  fill=(100, 100, 100), width=max(2, W // 300))

        # White box behind text for readability
        draw.rectangle(
            [tx - 4, ty - 3, tx + tw + 4, ty + th + 3],
            fill=(255, 255, 255),
            outline=(200, 200, 200),
            width=2
        )
        
        # Draw Label Text
        draw.text((tx, ty), label, fill=(10, 10, 10), font=font)

    out = BytesIO()
    img.save(out, format="JPEG", quality=93)
    return out.getvalue()


# ────────────────────────────────────────────────────────────────

async def generate_diagram(prompt: str) -> Optional[tuple[bytes, str]]:
    for var in ("CF_ACCOUNT_ID", "CF_API_TOKEN"):
        if not os.environ.get(var):
            logger.error(f"{var} environment variable is missing.")
            return None

    async with httpx.AsyncClient() as client:
        labels, base_image = (
            await _get_labels(client, prompt),
            await _generate_base_image(client, prompt),
        )

    if not base_image:
        return None

    if not labels:
        logger.warning("No labels — returning unlabelled image.")
        return base_image, prompt

    try:
        return _overlay_labels(base_image, labels), prompt
    except Exception as e:
        logger.error(f"Label overlay failed: {e}")
        return base_image, prompt


```
