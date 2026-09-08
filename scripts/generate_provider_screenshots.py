#!/usr/bin/env python3
"""
Generate screenshots for all G4F providers via the g4f.space/screenshot endpoint.

Usage:
    python scripts/generate_provider_screenshots.py
    python scripts/generate_provider_screenshots.py --output screenshots.json
    python scripts/generate_provider_screenshots.py --query "what is python"
    python scripts/generate_provider_screenshots.py --download-dir ./screenshots

What it does:
  1. Collects all provider URLs from g4f.Provider
  2. For each provider, triggers a screenshot via:
       https://g4f.space/screenshot?url=<provider_url>
  3. For search/AI providers (Google, Bing, YouTube, Perplexity, etc.),
     also generates query-based screenshots:
       https://g4f.space/screenshot?url=https://google.com/search?q=<query>&ai-mode=true
  4. Saves all screenshot URLs to a JSON file
  5. Optionally downloads the screenshots to a local directory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ── Configuration ────────────────────────────────────────────────────────────

SCREENSHOT_BASE = "https://g4f.space/screenshot"

# Providers that support search queries — we generate extra screenshots
# with ?q=<query> appended to show search/AI response pages.
SEARCH_PROVIDERS: dict[str, dict] = {
    "GoogleSearch": {
        "url": "https://google.com",
        "search_path": "/search",
        "query_param": "q",
    },
    "GoogleAiMode": {
        "url": "https://google.com",
        "search_path": "/search",
        "query_param": "q",
        "extra_params": {"ai-mode": "true"},
    },
    "YouTube": {
        "url": "https://youtube.com",
        "search_path": "/results",
        "query_param": "search_query",
    },
    "Perplexity": {
        "url": "https://www.perplexity.ai",
        "search_path": "/search",
        "query_param": "q",
    },
    "PhindAi": {
        "url": "https://phindai.org",
        "search_path": "/search",
        "query_param": "q",
    },
    "You": {
        "url": "https://you.com",
        "search_path": "/search",
        "query_param": "q",
    },
    "BingCreateImages": {
        "url": "https://www.bing.com/images/create",
        "search_path": "/images/create",
        "query_param": "q",
    },
}

# AI chat providers where we can append ?q= to show a response
AI_CHAT_PROVIDERS: list[str] = [
    "DeepSeek",
    "ChatGpt",
    "OpenaiChat",
    "OpenaiAccount",
    "Claude",
    "Gemini",
    "Grok",
    "Copilot",
    "HuggingChat",
    "Qwen",
    "BlackboxPro",
    "Perplexity",
    "Pi",
    "MetaAI",
    "GlhfChat",
    "CablyAI",
    "FenayAI",
    "Yqcloud",
    "TeachAnything",
    "ThebApi",
    "LMArena",
    "Reka",
    "HailuoAI",
]

DEFAULT_QUERY = "what is python programming language"
MAX_WORKERS = 4
REQUEST_TIMEOUT = 120
RETRY_DELAYS = [5, 15, 30]  # seconds between retries


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_providers() -> list[tuple[str, str]]:
    """Return list of (name, url) for all providers that have a URL."""
    from g4f.Provider import ProviderLoader

    providers = []
    for name in ProviderLoader.names:
        try:
            p = ProviderLoader.from_name(name)
            url = getattr(p, "url", None)
            if url:
                providers.append((name, url))
        except Exception:
            pass
    return providers


def build_screenshot_url(target_url: str) -> str:
    """Build the g4f.space/screenshot?url=... URL."""
    return f"{SCREENSHOT_BASE}?url={urllib.parse.quote_plus(target_url)}"


def build_search_url(provider_name: str, query: str) -> str | None:
    """Build a search/AI URL for providers that support queries."""
    config = SEARCH_PROVIDERS.get(provider_name)
    if not config:
        return None

    base = config["url"]
    path = config["search_path"]
    qparam = config["query_param"]
    extra = config.get("extra_params", {})

    params = {qparam: query, **extra}
    query_string = urllib.parse.urlencode(params)
    return f"{base}{path}?{query_string}"


def trigger_screenshot(screenshot_url: str, retries: int = 3) -> dict:
    """Trigger a screenshot and return status info."""
    for attempt in range(retries):
        try:
            resp = requests.get(screenshot_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                size = len(resp.content)
                return {
                    "success": True,
                    "status": 200,
                    "content_type": content_type,
                    "size": size,
                    "attempt": attempt + 1,
                }
            # Non-200 — retry with backoff
            if attempt < retries - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                time.sleep(delay)
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                time.sleep(delay)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "attempt": attempt + 1,
            }

    return {
        "success": False,
        "error": f"Failed after {retries} attempts",
        "attempt": retries,
    }


def download_screenshot(screenshot_url: str, filepath: Path) -> bool:
    """Download a screenshot to a local file."""
    try:
        resp = requests.get(screenshot_url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200 and resp.content:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(resp.content)
            return True
    except Exception as e:
        print(f"  Download failed: {e}")
    return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate screenshots for all G4F providers"
    )
    parser.add_argument(
        "--output", "-o",
        default="generated_media/provider_screenshots.json",
        help="Output JSON file with all screenshot URLs (default: generated_media/provider_screenshots.json)",
    )
    parser.add_argument(
        "--query", "-q",
        default=DEFAULT_QUERY,
        help=f"Search query for search/AI provider screenshots (default: '{DEFAULT_QUERY}')",
    )
    parser.add_argument(
        "--download-dir", "-d",
        default=None,
        help="If set, download screenshots to this directory",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--providers-only",
        action="store_true",
        help="Only generate provider homepage screenshots (skip search/AI queries)",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Only generate search/AI response screenshots (skip provider homepages)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Generate screenshot for a single provider by name",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  G4F Provider Screenshot Generator")
    print("=" * 70)
    print(f"  Screenshot endpoint: {SCREENSHOT_BASE}")
    print(f"  Query: '{args.query}'")
    print(f"  Output: {args.output}")
    if args.download_dir:
        print(f"  Download to: {args.download_dir}")
    print(f"  Workers: {args.workers}")
    print("=" * 70)

    # Collect all screenshot tasks
    providers = get_providers()
    print(f"\nFound {len(providers)} providers with URLs\n")

    tasks: list[dict] = []  # {key, screenshot_url, target_url, provider, type}

    for name, url in providers:
        if args.provider and name != args.provider:
            continue

        # 1. Provider homepage screenshot
        if not args.search_only:
            screenshot_url = build_screenshot_url(url)
            tasks.append({
                "key": f"home:{name}",
                "screenshot_url": screenshot_url,
                "target_url": url,
                "provider": name,
                "type": "homepage",
            })

        # 2. Search/AI response screenshots
        if not args.providers_only:
            # Search providers with dedicated search paths
            search_url = build_search_url(name, args.query)
            if search_url:
                screenshot_url = build_screenshot_url(search_url)
                tasks.append({
                    "key": f"search:{name}",
                    "screenshot_url": screenshot_url,
                    "target_url": search_url,
                    "provider": name,
                    "type": "search",
                    "query": args.query,
                })

            # AI chat providers — append ?q= to the chat URL
            if name in AI_CHAT_PROVIDERS:
                chat_url = f"{url}?q={urllib.parse.quote_plus(args.query)}"
                screenshot_url = build_screenshot_url(chat_url)
                tasks.append({
                    "key": f"ai:{name}",
                    "screenshot_url": screenshot_url,
                    "target_url": chat_url,
                    "provider": name,
                    "type": "ai_response",
                    "query": args.query,
                })

    print(f"Total screenshot tasks: {len(tasks)}")
    print()

    # Process tasks
    results: dict[str, dict] = {}
    completed = 0
    failed = 0

    def process_task(task: dict) -> dict:
        task_result = {**task}
        trigger_result = trigger_screenshot(task["screenshot_url"])
        task_result.update(trigger_result)

        # Download if requested
        if args.download_dir and trigger_result.get("success"):
            safe_name = task["key"].replace(":", "_").replace("/", "_")
            filepath = Path(args.download_dir) / f"{safe_name}.webp"
            task_result["downloaded"] = download_screenshot(
                task["screenshot_url"], filepath
            )
            task_result["filepath"] = str(filepath)
        else:
            task_result["downloaded"] = False

        return task_result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(process_task, task): task for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results[task["key"]] = result

                completed += 1
                status = "✓" if result.get("success") else "✗"
                if result.get("success"):
                    pass  # count below
                else:
                    failed += 1

                size_str = ""
                if result.get("size"):
                    size_str = f" ({result['size']:,} bytes)"

                print(
                    f"  [{completed}/{len(tasks)}] {status} {task['key']}{size_str}"
                )
            except Exception as e:
                failed += 1
                print(f"  [ERROR] {task['key']}: {e}")

    # Summary
    print()
    print("=" * 70)
    print(f"  Completed: {completed - failed} succeeded, {failed} failed")
    print(f"  Total: {completed}")
    print("=" * 70)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "screenshot_base": SCREENSHOT_BASE,
        "query": args.query,
        "total_tasks": len(tasks),
        "succeeded": completed - failed,
        "failed": failed,
        "screenshots": results,
    }

    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path}")

    # Print example URLs
    print("\nExample screenshot URLs:")
    for key, result in list(results.items())[:5]:
        print(f"  {key}: {result['screenshot_url']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
