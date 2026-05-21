"""
image_generator.py - Diagram Generator using Pollinations AI (Free, No API Key)
"""

import logging
import urllib.parse
from typing import Optional
import httpx

logger = logging.getLogger("EduBot.ImageGen")

async def generate_diagram(prompt: str) -> Optional[tuple[bytes, str]]:
    """
    Generate an educational diagram using Pollinations AI.
    Requires no API keys and uses the httpx library we already installed.
    """
    try:
        # We append some instructions to make it look like a textbook diagram
        enhanced_prompt = f"Clear, educational textbook diagram, highly detailed, white background. {prompt}"
        
        # URL-encode the prompt so it's safe to put in a web link
        safe_prompt = urllib.parse.quote(enhanced_prompt)
        
        # Pollinations AI generates images just by visiting a URL
        url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=1024&height=768&nologo=true&enhance=true"
        
        # FIX: Increased timeout to 60s and explicitly told httpx to follow redirects
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            
            # If the request was successful, return the image bytes
            if response.status_code == 200:
                return response.content, prompt
            else:
                logger.error(f"Image API returned status code {response.status_code}")
                
    except Exception as e:
        logger.error(f"Image generation failed for '{prompt}': {e}")
        
    return None
