"""
Yupp AI NextAction Token Extractor
Smart extraction with multiple fallback strategies
Only attempts extraction on token failure
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .constants import (
    MAX_EXTRACTION_RETRIES,
    MIN_REQUIRED_TOKENS,
    NEXT_ACTION_TOKENS,
    TOKEN_CACHE_TTL,
    TOKEN_PATTERNS,
    YUPP_BASE_URL,
    YUPP_CHAT_URL,
)


@dataclass
class TokenCache:
    """Cache for NextAction tokens"""

    tokens: Dict[str, str] = field(default_factory=dict)
    last_updated: Optional[datetime] = None
    failed_attempts: int = 0

    def is_expired(self) -> bool:
        """Check if cache is expired"""
        if self.last_updated is None:
            return True
        return datetime.now() - self.last_updated > timedelta(seconds=TOKEN_CACHE_TTL)

    def is_valid(self) -> bool:
        """Check if cache has valid tokens"""
        return (
            not self.is_expired()
            and len(self.tokens) >= MIN_REQUIRED_TOKENS
            and all(
                k in self.tokens for k in ["new_conversation", "existing_conversation"]
            )
        )


class TokenExtractor:
    """
    Extracts NextAction tokens from Yupp AI
    Uses multiple strategies and only attempts extraction on failure
    """

    def __init__(
        self,
        jwt_token: Optional[str] = None,
        scraper: Optional["cloudscraper.CloudScraper"] = None,
    ):
        self.jwt_token = jwt_token or os.getenv("YUPP_JWT") or os.getenv("YUPP_API_KEY")
        self.scraper = scraper
        self._cache = TokenCache()
        self._extraction_in_progress = False
        self._lock = asyncio.Lock()

    async def get_token(self, token_type: str) -> str:
        """
        Get a NextAction token from cache or fallback.

        This method does NOT trigger extraction - it only returns cached
        tokens or fallbacks. Extraction is only triggered by mark_token_failed().

        Args:
            token_type: Type of token to retrieve ("new_conversation" or
                "existing_conversation")

        Returns:
            The token string from cache if valid, otherwise the fallback token
        """
        # Return cached token if valid
        if self._cache.is_valid() and token_type in self._cache.tokens:
            return self._cache.tokens[token_type]

        # Return fallback token
        return NEXT_ACTION_TOKENS.get(
            token_type, NEXT_ACTION_TOKENS["new_conversation"]
        )

    async def mark_token_failed(self, token_type: str, token_value: str) -> None:
        """
        Mark a token as failed - this triggers extraction attempt
        Only extracts if we haven't tried too many times recently
        """
        async with self._lock:
            # Check if this is actually a cached token that failed
            cached_value = self._cache.tokens.get(token_type)

            # If the failed token matches our cache, increment failures
            if cached_value == token_value:
                self._cache.failed_attempts += 1
            elif token_value in NEXT_ACTION_TOKENS.values():
                # Hardcoded token failed - definitely need to extract
                self._cache.failed_attempts += 1

            # Only attempt extraction if we haven't failed too many times
            if self._cache.failed_attempts < MAX_EXTRACTION_RETRIES:
                if not self._extraction_in_progress:
                    # Set flag immediately to prevent race conditions
                    self._extraction_in_progress = True
                    # Start extraction in background
                    asyncio.create_task(self._attempt_extraction())

    async def _attempt_extraction(self) -> bool:
        """
        Attempt to extract fresh tokens from Yupp AI
        Uses multiple strategies for robustness
        """
        async with self._lock:
            if self._extraction_in_progress:
                return False
            self._extraction_in_progress = True

        try:
            # Try multiple extraction methods
            extracted_tokens = await self._extract_from_chat_page()

            if not extracted_tokens:
                extracted_tokens = await self._extract_from_main_page()

            if not extracted_tokens:
                extracted_tokens = await self._extract_from_js_bundles()

            if extracted_tokens and len(extracted_tokens) >= MIN_REQUIRED_TOKENS:
                # Update cache with extracted tokens
                async with self._lock:
                    self._cache.tokens = {
                        "new_conversation": extracted_tokens[0],
                        "existing_conversation": extracted_tokens[1]
                        if len(extracted_tokens) > 1
                        else extracted_tokens[0],
                    }
                    self._cache.last_updated = datetime.now()
                    self._cache.failed_attempts = 0
                return True

            return False

        except Exception as e:
            print(f"[Yupp TokenExtractor] Extraction failed: {e}")
            if os.getenv("DEBUG_MODE", "").lower() == "true":
                import traceback

                traceback.print_exc()
            return False
        finally:
            async with self._lock:
                self._extraction_in_progress = False

    async def _extract_from_chat_page(self) -> List[str]:
        """Extract tokens from chat page HTML"""
        try:
            headers = self._get_headers()

            if self.scraper:
                response = self.scraper.get(YUPP_CHAT_URL, headers=headers, timeout=10)
                text = response.text
            else:
                # Try to create a scraper if not provided
                try:
                    import cloudscraper

                    scraper = cloudscraper.create_scraper(
                        browser={
                            "browser": "chrome",
                            "platform": "windows",
                            "desktop": True,
                            "mobile": False,
                        },
                        delay=10,
                    )
                    scraper.headers.update(headers)
                    if self.jwt_token:
                        scraper.cookies.set(
                            "__Secure-yupp.session-token", self.jwt_token
                        )
                    response = scraper.get(YUPP_CHAT_URL, timeout=10)
                    text = response.text
                except ImportError:
                    import aiohttp

                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            YUPP_CHAT_URL,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            text = await response.text()

            tokens = self._extract_tokens_from_html(text)
            if tokens:
                print(
                    f"[Yupp TokenExtractor] Extracted {len(tokens)} tokens "
                    f"from chat page"
                )
                return tokens

        except Exception as e:
            print(f"[Yupp TokenExtractor] Chat page extraction failed: {e}")
            if os.getenv("DEBUG_MODE", "").lower() == "true":
                import traceback

                traceback.print_exc()

        return []

    async def _extract_from_main_page(self) -> List[str]:
        """Extract tokens from main page HTML"""
        try:
            headers = self._get_headers()

            if self.scraper:
                response = self.scraper.get(YUPP_BASE_URL, headers=headers, timeout=10)
                text = response.text
            else:
                # Try to create a scraper if not provided
                try:
                    import cloudscraper

                    scraper = cloudscraper.create_scraper(
                        browser={
                            "browser": "chrome",
                            "platform": "windows",
                            "desktop": True,
                            "mobile": False,
                        },
                        delay=10,
                    )
                    scraper.headers.update(headers)
                    if self.jwt_token:
                        scraper.cookies.set(
                            "__Secure-yupp.session-token", self.jwt_token
                        )
                    response = scraper.get(YUPP_BASE_URL, timeout=10)
                    text = response.text
                except ImportError:
                    import aiohttp

                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            YUPP_BASE_URL,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            text = await response.text()

            tokens = self._extract_tokens_from_html(text)
            if tokens:
                print(
                    f"[Yupp TokenExtractor] Extracted {len(tokens)} tokens "
                    f"from main page"
                )
                return tokens

        except Exception as e:
            print(f"[Yupp TokenExtractor] Main page extraction failed: {e}")
            if os.getenv("DEBUG_MODE", "").lower() == "true":
                import traceback

                traceback.print_exc()

        return []

    async def _extract_from_js_bundles(self) -> List[str]:
        """Extract tokens from JavaScript bundles"""
        try:
            import aiohttp

            # Common Next.js bundle patterns
            bundle_patterns = [
                "/_next/static/chunks/",
                "/_next/static/app/",
            ]

            headers = self._get_headers()

            async with aiohttp.ClientSession() as session:
                # Try to fetch a page and extract script URLs
                async with session.get(
                    YUPP_BASE_URL,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    text = await response.text()

                # Extract script URLs
                script_urls = re.findall(r'src="([^"]*\.js[^"]*)"', text)

                for script_url in script_urls:
                    if any(pattern in script_url for pattern in bundle_patterns):
                        try:
                            full_url = (
                                script_url
                                if script_url.startswith("http")
                                else f"{YUPP_BASE_URL}{script_url}"
                            )
                            async with session.get(
                                full_url,
                                headers=headers,
                                timeout=aiohttp.ClientTimeout(total=5),
                            ) as js_response:
                                js_text = await js_response.text()

                            tokens = self._extract_tokens_from_html(js_text)
                            if tokens and len(tokens) >= MIN_REQUIRED_TOKENS:
                                print(
                                    f"[Yupp TokenExtractor] Extracted tokens "
                                    f"from JS bundle: {script_url}"
                                )
                                return tokens
                        except Exception:
                            continue

        except Exception as e:
            print(f"[Yupp TokenExtractor] JS bundle extraction failed: {e}")
            if os.getenv("DEBUG_MODE", "").lower() == "true":
                import traceback

                traceback.print_exc()

        return []

    def _extract_tokens_from_html(self, html: str) -> List[str]:
        """Extract tokens from HTML/JS using multiple patterns"""
        all_tokens = set()

        for pattern in TOKEN_PATTERNS:
            matches = re.findall(pattern, html, re.IGNORECASE)
            all_tokens.update(matches)

        # Filter to only 40-42 character hex strings (likely action tokens)
        filtered_tokens = [
            token
            for token in all_tokens
            if re.match(r"^[a-f0-9]{40,42}$", token.lower())
        ]

        return list(filtered_tokens)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        if self.jwt_token:
            headers["Cookie"] = f"__Secure-yupp.session-token={self.jwt_token}"

        return headers


# Global singleton instance
_token_extractor: Optional[TokenExtractor] = None


def get_token_extractor(
    jwt_token: Optional[str] = None,
    scraper: Optional["cloudscraper.CloudScraper"] = None,
) -> TokenExtractor:
    """Get or create the global token extractor instance"""
    global _token_extractor
    if _token_extractor is None:
        _token_extractor = TokenExtractor(jwt_token=jwt_token, scraper=scraper)
    return _token_extractor
