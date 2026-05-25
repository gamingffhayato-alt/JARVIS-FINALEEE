"""
image_generator.py - Diagram Generator (FLUX + Groq Vision Pipeline)
  1. Cloudflare AI (FLUX) generates a clean UNLABELLED illustration.
  2. Groq (Llama 3.2 Vision) analyzes the GENERATED IMAGE to accurately map x/y coordinates.
  3. Pillow draws the labels exactly where Groq saw them.
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
    """Helper to construct the Cloudflare AI API URL for image generation."""
    return (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{os.environ.get('CF_ACCOUNT_ID', '')}/ai/run/{model}"
    )


async def _generate_base_image(client: httpx.AsyncClient, topic: str) -> Optional[bytes]:
    """Step 1: Generate the clean, unlabelled base image via Cloudflare FLUX."""
    logger.info(f"Generating base image for topic: {topic}")
    
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
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('CF_API_TOKEN', '')}",
        "Content-Type": "application/json",
    }
    
    resp = await client.post(
        _cf_url("@cf/black-forest-labs/flux-1-schnell"),
        headers=headers,
        json={"prompt": prompt},
        timeout=60.0,
    )

    if resp.status_code != 200:
        logger.error(f"Image gen error {resp.status_code}: {resp.text}")
        return None

    b64 = resp.json().get("result", {}).get("image")
    return base64.b64decode(b64) if b64 else None


async def _get_groq_labels(client: httpx.AsyncClient, image_bytes: bytes, topic: str) -> list[dict]:
    """Step 2: Pass the generated image to Groq Vision to extract dynamic coordinates."""
    logger.info("Passing image to Groq Vision for coordinate mapping...")
    
    # Encode image for Groq Vision payload
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_uri = f"data:image/jpeg;base64,{b64_image}"

    headers = {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "llama-3.2-90b-vision-preview", # Groq's largest vision model for best accuracy
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Look closely at this specific image of a {topic}. Identify up to 8 distinct parts. "
                            "Only label parts that are clearly visible in this specific drawing. Do not guess locations for parts that are not drawn. "
                            "Reply ONLY with a JSON array, no markdown formatting, no text outside the array. "
                            "Each item must be exactly: {\"label\": \"part name\", \"x\": 0.0-1.0, \"y\": 0.0-1.0}. "
                            "The x and y values MUST be the exact decimal coordinates (0.0 to 1.0) pointing directly "
                            "at the center of that specific part in the image. x=0 is left edge, x=1 is right edge, "
                            "y=0 is top edge, y=1 is bottom edge."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
    }

    resp = await client.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30.0,
    )

    if resp.status_code != 200:
        logger.error(f"Groq label error {resp.status_code}: {resp.text}")
        return []

    raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # Clean up any potential markdown formatting
    raw = re.sub(r"
