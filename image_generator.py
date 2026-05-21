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
        # Client automatically picks up GEMINI_API_KEY from the environment
        client = genai.Client()
        
        # We use Nano Banana 2 (gemini-3.1-flash-image-preview) 
        response = await client.aio.models.generate_images(
            model='gemini-3.1-flash-image-preview',
            prompt=f"Clear, educational textbook diagram, high quality, white background. {prompt}",
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="16:9"
            )
        )
        
        if response.generated_images:
            img_bytes = response.generated_images[0].image.image_bytes
            return img_bytes, prompt
            
    except Exception as e:
        logger.error(f"Nano Banana generation failed for '{prompt}': {e}")
        
    return None
