"""
Rate limit configurations for different models.
TPM = Tokens Per Minute
RPM = Requests Per Minute
TPD = Tokens Per Day
"""

# Thresholds for recommended limits as a percentage of max
RECOMMENDED_RPM_THRESHOLD = 0.75  # 75% of max RPM
RECOMMENDED_TPM_THRESHOLD = 0.75  # 75% of max TPM

RATE_LIMITS = {
    # OpenAI Models
    "gpt-5-mini": {
        "tpm": 180_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    }, 
    "gpt-5-nano": {
        "tpm": 180_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "gpt-5": {
        "tpm": 40_000_000,
        "rpm": 15_000,
        "tpd": 15_000_000_000,
    },
    "gpt-5.1": {
        "tpm": 40_000_000,
        "rpm": 15_000,
        "tpd": 15_000_000_000,
    },
    "gpt-4.1": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 15_000_000_000,
    },
    "gpt-4o-2024-08-06": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 15_000_000_000,
    },
    "gpt-4.1-2025-04-14": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 15_000_000_000,
    },
    "gpt-4.1-mini": {
        "tpm": 150_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "gpt-4.1-nano": {
        "tpm": 150_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "o3": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 10_000_000_000,
    },
    "o4-mini": {
        "tpm": 150_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "gpt-4o": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 15_000_000_000,
    },
    "gpt-4o-mini": {
        "tpm": 150_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "gpt-4o-realtime-preview": {
        "tpm": 15_000_000,
        "rpm": 20_000,
        "tpd": None,
    },
    "gpt-4o-mini-tts": {
        "tpm": 8_000_000,
        "rpm": 10_000,
        "tpd": None,
    },
    "dall-e-3": {
        "tpm": None,
        "rpm": 10_000,
        "tpd": None,
    },
    "o1": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 10_000_000_000,
    },
    "o1-2024-12-17": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 10_000_000_000,
    },
    "o1-mini": {
        "tpm": 150_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "o1-mini-2024-09-12": {
        "tpm": 150_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "o1-preview": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 10_000_000_000,
    },
    "o1-preview-2024-09-12": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 10_000_000_000,
    },
    "o1-pro": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 5_000_000_000,
    },
    "o1-pro-2025-03-19": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 5_000_000_000,
    },
    "o3": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 10_000_000_000,
    },
    "o3-2025-04-16": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 10_000_000_000,
    },
    "o3-deep-research": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 10_000_000,
    },
    "o3-deep-research-2025-06-26": {
        "tpm": 250_000,
        "rpm": 3_000,
        "tpd": None,
    },
    "o3-mini": {
        "tpm": 150_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "o3-mini-2025-01-31": {
        "tpm": 150_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "o3-pro": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 5_000_000_000,
    },
    "o3-pro-2025-06-10": {
        "tpm": 30_000_000,
        "rpm": 10_000,
        "tpd": 5_000_000_000,
    },
    "o4-mini": {
        "tpm": 150_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "o4-mini-2025-04-16": {
        "tpm": 150_000_000,
        "rpm": 30_000,
        "tpd": 15_000_000_000,
    },
    "o4-mini-deep-research": {
        "tpm": 150_000_000,
        "rpm": 10_000,
        "tpd": 10_000_000,
    },
    "o4-mini-deep-research-2025-06-26": {
        "tpm": 250_000,
        "rpm": 3_000,
        "tpd": None,
    },
}

# o4-mini-high is o4-mini with reasoning_effort set to high:  https://www.reddit.com/r/OpenAI/comments/1kwtm56/is_o4_mini_via_the_api_the_same_as_o4_mini_high/

def get_rate_limit(model_name: str) -> int:
    """
    Get the recommended rate limit for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Recommended requests per minute
        
    Raises:
        KeyError: If model is not found in rate limits
    """
    if model_name not in RATE_LIMITS:
        raise KeyError(f"Model {model_name} not found in rate limits. Available models: {list(RATE_LIMITS.keys())}")
    
    max_rpm = RATE_LIMITS[model_name]["rpm"]
    return int(max_rpm * RECOMMENDED_RPM_THRESHOLD)

def get_token_limit(model_name: str) -> int:
    """
    Get the recommended token limit for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Recommended tokens per minute
        
    Raises:
        KeyError: If model is not found in rate limits
    """
    if model_name not in RATE_LIMITS:
        raise KeyError(f"Model {model_name} not found in rate limits. Available models: {list(RATE_LIMITS.keys())}")
    
    max_tpm = RATE_LIMITS[model_name]["tpm"]
    if max_tpm is None:
        return None
    return int(max_tpm * RECOMMENDED_TPM_THRESHOLD)

def get_model_info(model_name: str) -> dict:
    """
    Get all rate limit information for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary containing all rate limit information
        
    Raises:
        KeyError: If no matching model is found in rate limits
    """
    # Find the closest matching model name
    matching_models = [key for key in RATE_LIMITS.keys() if key in model_name]
    if not matching_models:
        raise KeyError(f"No matching model found for {model_name} in rate limits. Available models: {list(RATE_LIMITS.keys())}")
    
    # Use the longest matching model name (most specific match)
    model_key = max(matching_models, key=len)
    model_info = RATE_LIMITS[model_key].copy()
    model_info["recommended_rpm"] = get_rate_limit(model_key)
    model_info["recommended_tpm"] = get_token_limit(model_key)
    return model_info 
