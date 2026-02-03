"""
Constants for Yupp AI NextAction token management
"""

# Default fallback tokens (hardcoded)
NEXT_ACTION_TOKENS = {
    "new_conversation": "7f7de0a21bc8dc3cee8ba8b6de632ff16f769649dd",
    "existing_conversation": "7f9ec99a63cbb61f69ef18c0927689629bda07f1bf",
}

# Cache settings
TOKEN_CACHE_TTL = 3600  # 1 hour in seconds
MAX_EXTRACTION_RETRIES = 3
MIN_REQUIRED_TOKENS = 2  # Minimum tokens needed to update cache

# URLs
YUPP_BASE_URL = "https://yupp.ai"
YUPP_CHAT_URL = "https://yupp.ai/chat"

# Regex patterns for token extraction
TOKEN_PATTERNS = [
    # Standard patterns
    r'next-action["\']?\s*[:=]\s*["\']?([a-f0-9]{40,42})',
    r'"next-action"\s*:\s*"([a-f0-9]{40,42})"',
    r'"actionId"\s*:\s*"([a-f0-9]{40,42})"',
    r'nextAction["\']?\s*:\s*["\']?([a-f0-9]{40,42})',
    # Broader patterns for various formats
    r'["\']?action["\']?\s*[:=]\s*["\']?([a-f0-9]{40,42})',
    r'["\']?new_conversation["\']?\s*[:=]\s*["\']?([a-f0-9]{40,42})',
    r'["\']?existing_conversation["\']?\s*[:=]\s*["\']?([a-f0-9]{40,42})',
    r'["\']?new["\']?\s*[:=]\s*["\']?([a-f0-9]{40,42})',
    r'["\']?existing["\']?\s*[:=]\s*["\']?([a-f0-9]{40,42})',
]
