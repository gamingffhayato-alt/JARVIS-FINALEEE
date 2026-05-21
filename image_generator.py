"""
image_generator.py - Diagram Generator using Gemini Nano Banana
"""

import os
import logging
from typing import Optional
from google import genai
from google.genai import types

logger = logging.getLogger("EduBot.ImageGen")

async def generate_diagram(prompt: str) -> Optional[tuple[bytes, str]]:
    """
    Generate an educational diagram using Gemini's Nano Banana 2 model.
    """
    try:
        # Failsafe to check if you added the key to Railway
        if not os.environ.get("GEMINI_API_KEY"):
            logger.error("GEMINI_API_KEY is missing from environment variables!")
            return None

        client = genai.Client()
        
        # Nano Banana uses the generate_content endpoint with multimodal outputs, not generate_images
        response = await client.aio.models.generate_content(
            model='gemini-3.1-flash-image-preview',
            contents=f"Clear, educational textbook diagram, high quality, white background. {prompt}",
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9"
                )
            )
        )
        
        # Extract the raw image bytes from the response parts
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data, prompt
            
    except Exception as e:
        logger.error(f"Nano Banana generation failed for '{prompt}': {e}")
        
    return None
