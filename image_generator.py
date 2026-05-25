"""
image_generator.py - Diagram Generator (Vision VLM Pipeline)
  1. Cloudflare AI (FLUX) generates a clean UNLABELLED illustration.
  2. Cloudflare AI (Vision) analyzes the GENERATED IMAGE to map x/y coordinates.
  3. Pillow draws the labels exactly where the Vision model saw them.
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

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EduBot.ImageGen")


def _cf_url(model: str) -> str:
    """Helper to construct the Cloudflare AI API URL."""
    return (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{os.environ.get('CF_ACCOUNT_ID', '')}/ai/run/{model}"
    )


def _headers() -> dict:
    """Helper to construct the Cloudflare AI API headers."""
    return {
        "Authorization": f"Bearer {os.environ.get('CF_API_TOKEN', '')}",
        "Content-Type": "application/json",
    }


async def _generate_base_image(client: httpx.AsyncClient, topic: str) -> Optional[bytes]:
    """Step 1: Generate the clean, unlabelled base image."""
    logger.info(f"Generating base image for topic: {topic}")
    
    # Strip words that trick the model into generating garbled text
    clean_topic = re.sub(
        r'\b(diagram|label|labelled|labeled|labeling|annotated|with labels|chart)\b',
        '', topic, flags=re.IGNORECASE
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


async def _get_vision_labels(client: httpx.AsyncClient, image_bytes: bytes, topic: str) -> list[dict]:
    """Step 2: Pass the generated image to a Vision model to extract dynamic coordinates."""
    logger.info("Passing image to Vision model for coordinate mapping...")
    b64_image = base64.b64encode(image_bytes).decode('utf-8')

    # You can switch this to @cf/meta/llama-3.2-11b-vision-instruct if available in your CF tier
    model = "@cf/llava-hf/llava-1.5-7b-hf" 
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Look at this image of a {topic}. Identify up to 8 distinct parts. "
                    "Reply ONLY with a JSON array, no markdown. "
                    "Each item: {\"label\": \"part name\", \"x\": 0.0-1.0, \"y\": 0.0-1.0}. "
                    "The x and y values MUST be the exact coordinates (0.0 to 1.0) pointing directly "
                    "at the center of that specific part in the image. x=0 is left, y=0 is top."
                )
            }
        ],
        "image": [b64_image],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    resp = await client.post(
        _cf_url(model),
        headers=_headers(),
        json=payload,
        timeout=45.0,
    )

    if resp.status_code != 200:
        logger.error(f"Vision label error {resp.status_code}: {resp.text}")
        return []

    raw = resp.json().get("result", {}).get("response", "")
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            logger.info(f"Successfully extracted {len(data)} labels via Vision.")
            return data
    except json.JSONDecodeError:
        logger.error(f"Vision JSON parse failed: {raw[:300]}")
        
    return []


def _overlay_labels(image_bytes: bytes, labels: list[dict]) -> bytes:
    """Step 3: Draw the labels and connector lines using the calculated coordinates."""
    logger.info("Overlaying labels onto image...")
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Load font, fallback to default if DejaVu isn't installed
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

        # Coordinate targeted by the vision model
        ax = int(float(item.get("x", 0.5)) * W)
        ay = int(float(item.get("y", 0.5)) * H)

        # Draw the anchor dot on the anatomy
        r = max(5, W // 120)
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], fill=(200, 30, 30))

        # Push label text box outward toward the edges
        offset_x = max(90, W // 5)
        offset_y = max(40, H // 10)
        tx = ax + (offset_x if ax >= cx else -offset_x)
        ty = ay + (offset_y if ay >= cy else -offset_y)

        # Ensure the text stays within the image boundaries
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 12
        tx = max(pad, min(tx, W - tw - pad))
        ty = max(pad, min(ty, H - th - pad))

        # Calculate where the connector line should attach to the text box
        line_target_x = tx + (tw // 2 if tx > ax else tw)
        if tx <= ax <= tx + tw:
            line_target_x = tx + tw // 2 
            
        # Draw connector line from the dot to the text box
        draw.line(
            [(ax, ay), (line_target_x, ty + th // 2)],
            fill=(100, 100, 100), 
            width=max(2, W // 300)
        )

        # Draw white background box for readability
        draw.rectangle(
            [tx - 4, ty - 3, tx + tw + 4, ty + th + 3],
            fill=(255, 255, 255),
            outline=(150, 150, 150),
            width=2
        )
        
        # Draw the label text
        draw.text((tx, ty), label, fill=(10, 10, 10), font=font)

    out = BytesIO()
    img.save(out, format="JPEG", quality=95)
    return out.getvalue()


# ────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────

async def generate_diagram(prompt: str) -> Optional[tuple[bytes, str]]:
    """Generates an image, maps it with a Vision model, and overlays labels."""
    for var in ("CF_ACCOUNT_ID", "CF_API_TOKEN"):
        if not os.environ.get(var):
            logger.error(f"{var} environment variable is missing. Check your setup.")
            return None

    async with httpx.AsyncClient() as client:
        # Step 1: Image MUST be generated first
        base_image = await _generate_base_image(client, prompt)
        
        if not base_image:
            return None
        
        # Step 2: Pass the generated image to the Vision model
        labels = await _get_vision_labels(client, base_image, prompt)

    if not labels:
        logger.warning("Vision model failed to extract labels. Returning unlabelled image.")
        return base_image, prompt

    # Step 3: Overlay the dynamic labels
    try:
        final_image = _overlay_labels(base_image, labels)
        logger.info("Diagram generation pipeline completed successfully.")
        return final_image, prompt
    except Exception as e:
        logger.error(f"Label overlay failed: {e}")
        return base_image, prompt
