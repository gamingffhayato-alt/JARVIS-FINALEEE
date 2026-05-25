"""
image_generator.py - Diagram Generator using Cloudflare Workers AI (Stable Diffusion XL)
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger("EduBot.ImageGen")

async def generate_diagram(prompt: str) -> Optional[tuple[bytes, str]]:
    """
    Generate an educational diagram using Stable Diffusion XL via Cloudflare Workers AI.
    """
    # Paste your credentials right here for testing
    account_id = "YOUR_CLOUDFLARE_ACCOUNT_ID" 
    api_token = "YOUR_CLOUDFLARE_API_TOKEN"
    
    # Using Stable Diffusion XL Base 1.0 on Cloudflare Workers AI
    model = "@cf/stabilityai/stable-diffusion-xl-base-1.0"
    api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Enhance the prompt to ensure a clean, academic style
    enhanced_prompt = f"A clear, educational textbook diagram, highly detailed, white background, clean labels. Subject: {prompt}"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                api_url, 
                headers=headers, 
                json={"prompt": enhanced_prompt} 
            )
            
            if response.status_code == 200:
                return response.content, prompt
            else:
                logger.error(f"Cloudflare API Error {response.status_code}: {response.text}")
                
    except Exception as e:
        logger.error(f"Image generation failed for '{prompt}': {e}")
        
    return None
