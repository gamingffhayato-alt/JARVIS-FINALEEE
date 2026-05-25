"""
image_generator.py - Diagram Generator using Cloudflare AI (FLUX.1-schnell)
"""

import os
import logging
import base64
from typing import Optional
import httpx

logger = logging.getLogger("EduBot.ImageGen")

async def generate_diagram(prompt: str) -> Optional[tuple[bytes, str]]:
    """
    Generate an educational diagram using FLUX.1-schnell via Cloudflare Workers AI.
    """
    cf_account_id = os.environ.get("CF_ACCOUNT_ID")
    cf_api_token = os.environ.get("CF_API_TOKEN")

    if not cf_account_id:
        logger.error("CF_ACCOUNT_ID is missing! Please add it to your environment variables.")
        return None
    if not cf_api_token:
        logger.error("CF_API_TOKEN is missing! Please add it to your environment variables.")
        return None

    API_URL = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"

    headers = {
        "Authorization": f"Bearer {cf_api_token}",
        "Content-Type": "application/json"
    }

    # Enhance the prompt to ensure a clean, academic style
    enhanced_prompt = f"A clear, educational textbook diagram, highly detailed, white background, clean labels. Subject: {prompt}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                API_URL,
                headers=headers,
                json={"prompt": enhanced_prompt}
            )

            if response.status_code == 200:
                result = response.json()
                # Cloudflare returns base64-encoded image in result.image
                image_b64 = result.get("result", {}).get("image")
                if image_b64:
                    image_bytes = base64.b64decode(image_b64)
                    return image_bytes, prompt
                else:
                    logger.error("Cloudflare AI response missing image data.")
            else:
                logger.error(f"Cloudflare AI API Error {response.status_code}: {response.text}")

    except Exception as e:
        logger.error(f"Image generation failed for '{prompt}': {e}")

    return None
