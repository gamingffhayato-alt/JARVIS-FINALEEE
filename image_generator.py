"""
image_generator.py - Diagram Generator using Hugging Face (FLUX.1-schnell)
"""

import os
import logging
from typing import Optional
import httpx

logger = logging.getLogger("EduBot.ImageGen")

async def generate_diagram(prompt: str) -> Optional[tuple[bytes, str]]:
    """
    Generate an educational diagram using FLUX.1-schnell via Hugging Face Inference API.
    """
    # FLUX.1-schnell is exceptionally fast and great at generating text/labels
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN is missing! Please add it to your environment variables.")
        return None

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    # Enhance the prompt to ensure a clean, academic style
    enhanced_prompt = f"A clear, educational textbook diagram, highly detailed, white background, clean labels. Subject: {prompt}"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                API_URL, 
                headers=headers, 
                json={"inputs": enhanced_prompt}
            )
            
            if response.status_code == 200:
                return response.content, prompt
            else:
                logger.error(f"Hugging Face API Error {response.status_code}: {response.text}")
                
    except Exception as e:
        logger.error(f"Image generation failed for '{prompt}': {e}")
        
    return None
