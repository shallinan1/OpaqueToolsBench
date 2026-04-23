from dotenv import load_dotenv; load_dotenv()
import os
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
import time
import logging
from src.generation_utils.api_request_parallel_processor import process_api_requests
logger = logging.getLogger(__name__)

default_request_url = "https://api.together.xyz/v1/chat/completions"

async def together_parallel_generate(
    requests, 
    max_requests_per_minute=2500,  # 83% of your 3000 RPM limit
    max_tokens_per_minute=10000000,
    request_url="https://api.together.xyz/v1/chat/completions"
):
    start_time = time.time()
    
    kwargs = {
        "requests": requests,
        "request_url": request_url,
        "api_key": TOGETHER_API_KEY,
        "max_requests_per_minute": max_requests_per_minute,
        "max_tokens_per_minute": max_tokens_per_minute,
        "max_attempts": 5,
        "logging_level": logging.WARNING,
        "token_encoding_name": "cl100k_base"
    }
    
    results = await process_api_requests(**kwargs)
    
    end_time = time.time()
    duration = end_time - start_time
    rps = len(requests) / duration if duration > 0 else 0
    logger.info(f"Total generation time: {duration:.2f} seconds for {len(requests)} samples ({rps:.1f} req/sec)")
    
    return results