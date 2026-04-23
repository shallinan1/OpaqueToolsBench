from dotenv import load_dotenv; load_dotenv()
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
import time
import logging
from src.generation_utils.api_request_parallel_processor import process_api_requests

logger = logging.getLogger(__name__)
import tiktoken

# Store the type of request we need to make for different models
requests_url_dict = {
    "gpt-3.5-turbo-0125": "https://api.openai.com/v1/chat/completions",
    "gpt-3.5-turbo-instruct": "https://api.openai.com/v1/completions"
}
default_request_url = requests_url_dict["gpt-3.5-turbo-0125"]

async def openai_parallel_generate(requests, 
                                   max_requests_per_minute=10000, 
                                   max_tokens_per_minute=30000000,
                                   request_url="https://api.openai.com/v1/chat/completions"):
    start_time = time.time()
    
    kwargs = {
        "requests": requests,
        "request_url": request_url,
        "api_key": OPENAI_API_KEY,
        "max_requests_per_minute": max_requests_per_minute,
        "max_tokens_per_minute": max_tokens_per_minute,
        "max_attempts": 5,
        "logging_level": logging.WARNING
    }

    try:
        encoding = tiktoken.encoding_for_model(requests[0]["model"])
        kwargs["token_encoding_name"] = encoding.name
        logger.info(f"Using token encoding: {encoding.name}")
    except KeyError:
        # Use default encoding for unknown models
        # Note: This should match the default in api_request_parallel_processor.py line 459
        kwargs["token_encoding_name"] = "o200k_base"
        logger.info(f"Model not recognized, using default token encoding: o200k_base")

    results = await process_api_requests(**kwargs)

    end_time = time.time()
    logger.info(f"Total generation time: {end_time - start_time:.2f} seconds for {len(requests)} samples")

    return results