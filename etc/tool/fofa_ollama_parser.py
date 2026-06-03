#!/usr/bin/env python3
"""
FOFA Ollama Parser
==================
Discovers open Ollama instances via FOFA search engine.

Supports three data sources:
  1. Web scraping  (--token, default) — no API credits needed
  2. Internal API JSON (--from-json FILE) — paste output from browser curl
  3. Stdin pipe     (--from-json -)    — cat results.json | python ... --from-json -

Usage:
    # Web scraping (needs fofa_token cookie from browser)
    python etc/tool/fofa_ollama_parser.py --token "eyJhbG..." --pages 10

    # Internal API JSON (copy curl from browser DevTools, save response)
    python etc/tool/fofa_ollama_parser.py --from-json results.json

    # Pipe from curl
    curl '...' | python etc/tool/fofa_ollama_parser.py --from-json -

    # Update OllamaSwarm.py directly
    python etc/tool/fofa_ollama_parser.py --token "..." --update

Get your token:
    1. Log in to fofa.info in your browser
    2. DevTools (F12) → Application → Cookies → fofa.info
    3. Copy the value of 'fofa_token' (starts with eyJ...)

Environment variable: FOFA_TOKEN
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FOFA_RESULT_URL = "https://fofa.info/result"
DEFAULT_QUERY = 'port="11434" && body="Ollama"'
DEFAULT_PAGES = 10
DEFAULT_TIMEOUT = 5
DEFAULT_WORKERS = 30
OLLAMA_SWARM_PATH = (
    Path(__file__).resolve().parents[2] / "g4f" / "Provider" / "OllamaSwarm.py"
)
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Data source 1: Web Scraping with HTML model extraction
# ---------------------------------------------------------------------------
def create_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    s.cookies.set("fofa_token", token, domain="fofa.info")
    s.cookies.set("theme", "dark", domain="fofa.info")
    return s


def _extract_models_from_html(html: str, host: str) -> list[str]:
    """Extract model names from FOFA HTML detail columns for a specific host.

    FOFA embeds structured info in hsxa-database-table elements.
    Model names appear in body/banner text snippets like "model_name:tag".
    """
    models: list[str] = []
    # Look for struct_info / body content near this host
    # Typical pattern: Ollama model tags like "llama3:latest", "qwen2:7b"
    # These appear in <td> cells or inline text blocks
    model_pattern = re.compile(
        r'\b([a-zA-Z][a-zA-Z0-9._-]{1,40}(?::[a-zA-Z0-9._-]{1,30}))\b'
    )

    # Search in area around the host IP within the HTML
    ip = host.split(":")[0]
    # Find all sections mentioning this IP and extract model-like strings
    for block in re.findall(
        rf'{re.escape(ip)}.*?(?:</tr>|</div>)', html, re.DOTALL | re.IGNORECASE
    ):
        for m in model_pattern.findall(block):
            # Filter: must have a colon (model:tag format) and not be a URL/protocol
            if ":" in m and not m.startswith(("http:", "https:", "ftp:")):
                name = m.strip()
                if name not in models:
                    models.append(name)
    return models


def scrape_page(session: requests.Session, query: str, page: int) -> list[dict]:
    """Scrape one FOFA result page, return list of {host, models} dicts."""
    qbase64 = base64.b64encode(query.encode()).decode()
    # FOFA allows up to 50 results per page even on free accounts.
    # This is crucial because free accounts can't access page 2.
    params = {"qbase64": qbase64, "page": page, "page_size": 50}
    url = f"{FOFA_RESULT_URL}?{urlencode(params)}"

    for attempt in range(3):
        resp = session.get(url, timeout=30)
        if resp.status_code == 429:
            import time
            print(f" [429 Rate Limit] Retrying in 5s...", end="", flush=True)
            time.sleep(5)
            continue
        resp.raise_for_status()
        break

    html = resp.text
    if '800004' in html or '权限不足' in html:
        raise Exception("Free tier limit reached! [800004] 权限不足 (FOFA limits pagination for free accounts)")
        
    results: list[dict] = []
    seen: set[str] = set()

    # Extract host:port from clipboard buttons and links
    hosts: list[str] = []
    for match in re.findall(r'data-clipboard-text="([^"]+)"', html):
        host = match.strip()
        if host.startswith("http"):
            host = host.split("//", 1)[-1].rstrip("/")
        if re.match(r"[\w.\-]+:\d+", host) and host not in seen:
            seen.add(host)
            hosts.append(host)

    # Fallback: link hrefs
    if not hosts:
        for match in re.findall(r'href="https?://([\d.]+:\d+)"', html):
            if match not in seen:
                seen.add(match)
                hosts.append(match)

    # Try to extract model info from HTML for each host
    for host in hosts:
        models = _extract_models_from_html(html, host)
        results.append({"host": host, "models": models})

    return results


import random

def build_query(base: str, country: str = None, model: str = None) -> str:
    query = base
    if country:
        query += f' && country="{country}"'
    if model:
        query += f' && "{model}"'
    return query

def scrape_all(session: requests.Session, base_query: str, pages: int, is_deep_scan: bool = False, specific_country: str = None, specific_model: str = None) -> list[dict]:
    """Scrape multiple pages, optionally using dork slicing for deep scans."""
    all_results: list[dict] = []
    seen: set[str] = set()

    # Dork Slicing configuration
    countries = ["US", "CN", "DE", "KR", "SG", "JP", "GB", "FR", "TW", "RU"] if is_deep_scan and not specific_country else [specific_country]
    models = ["llama3", "deepseek", "qwen", "mistral", "gemma", "phi3"] if is_deep_scan and not specific_model else [specific_model]
    
    # If not deep scan and no specific filters, just use a single empty iteration
    if not is_deep_scan and not specific_country and not specific_model:
        countries = [None]
        models = [None]

    queries_to_run = []
    for c in countries:
        for m in models:
            queries_to_run.append(build_query(base_query, c, m))

    print(f"\n[*] Generated {len(queries_to_run)} queries to bypass limits.")

    for q_idx, query in enumerate(queries_to_run, 1):
        if len(queries_to_run) > 1:
            print(f"\n[*] Executing Query {q_idx}/{len(queries_to_run)}: {query}")
            
        for page in range(1, pages + 1):
            try:
                print(f"[*] Page {page}/{pages} ...", end="", flush=True)
                results = scrape_page(session, query, page)
                
                # Count how many are actually new
                new_results = []
                for r in results:
                    if r["host"] not in seen:
                        seen.add(r["host"])
                        new_results.append(r)
                        
                print(f" {len(results)} found, {len(new_results)} new")
                
                if not results:
                    # No results on this page at all
                    break
                    
                all_results.extend(new_results)
                
                # If FOFA starts repeating the same results (common on free tier pagination limit)
                if len(new_results) == 0 and len(results) > 0:
                    print("[-] FOFA is repeating results. Free tier limit reached for this query. Moving to next.")
                    break

                if page < pages:
                    time.sleep(random.uniform(1.0, 2.5))
            except Exception as exc:
                print(f" ERROR: {exc}")
                break
                
        if q_idx < len(queries_to_run):
            # Sleep between different queries to avoid FOFA rate limiting
            sleep_time = random.uniform(2.0, 5.0)
            print(f"[*] Sleeping for {sleep_time:.1f}s before next query...")
            time.sleep(sleep_time)

    print(f"\n[+] Total unique endpoints across all queries: {len(all_results)}")
    return all_results


# ---------------------------------------------------------------------------
# Data source 2: Internal API JSON (--from-json)
# ---------------------------------------------------------------------------
def parse_internal_json(data: dict) -> list[dict]:
    """Parse JSON from FOFA internal API (/v1/search).

    Expected format:
        { "results": [ { "ip": "...", "port": 11434,
                          "struct_info": { "banner": "... model info ..." },
                          ... }, ... ] }

    Also supports the simpler format:
        { "results": [ ["ip:port", ...], ... ] }
    """
    results: list[dict] = []
    seen: set[str] = set()
    raw_results = data.get("results", [])

    for item in raw_results:
        host = None
        models: list[str] = []

        if isinstance(item, dict):
            # Structured result from /v1/search
            ip = item.get("ip", "")
            port = item.get("port", 11434)
            host = f"{ip}:{port}" if ip else None

            # Extract models from struct_info / body / banner
            struct = item.get("struct_info", {})
            if isinstance(struct, dict):
                banner = struct.get("banner", "") or ""
                body = struct.get("body", "") or ""
                text = f"{banner} {body}"
            else:
                text = str(struct)

            # Also check top-level body/banner
            text += " " + (item.get("body", "") or "")
            text += " " + (item.get("banner", "") or "")

            # Extract model names (format: name:tag)
            for m in re.findall(
                r'\b([a-zA-Z][a-zA-Z0-9._-]{1,40}:[a-zA-Z0-9._-]{1,30})\b', text
            ):
                if not m.startswith(("http:", "https:", "ftp:")):
                    if m not in models:
                        models.append(m)

            # If struct_info has explicit models list
            if isinstance(struct, dict) and "models" in struct:
                for m in struct["models"]:
                    name = m.get("name", "") if isinstance(m, dict) else str(m)
                    if name and name not in models:
                        models.append(name)

        elif isinstance(item, list) and len(item) >= 1:
            # Array format: ["ip:port", ...]
            host_str = str(item[0])
            if host_str.startswith("http"):
                host_str = host_str.split("//", 1)[-1].rstrip("/")
            if re.match(r"[\w.\-]+:\d+", host_str):
                host = host_str

        if host and host not in seen:
            seen.add(host)
            results.append({"host": host, "models": models})

    print(f"[+] Parsed {len(results)} endpoints from JSON")
    return results

def parse_har_file(har_path: str) -> list[dict]:
    """Parse FOFA HAR file to extract IPs and ports."""
    print(f"[*] Parsing HAR file: {har_path}")
    try:
        with open(har_path, 'r', encoding='utf-8') as f:
            har_data = json.load(f)
    except Exception as e:
        print(f"[!] Error reading HAR file: {e}", file=sys.stderr)
        return []

    ip_port_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}:[0-9]{1,5}\b')
    all_matches = set()

    for entry in har_data.get('log', {}).get('entries', []):
        response = entry.get('response', {})
        content = response.get('content', {})
        text = content.get('text', '')
        if text:
            matches = ip_port_pattern.findall(text)
            all_matches.update(matches)

    results = []
    for host in all_matches:
        results.append({"host": host, "models": []})
        
    print(f"[+] Found {len(results)} endpoints in HAR.")
    return results

def parse_txt_file(txt_path: str) -> list[dict]:
    """Parse any text file to extract IPs and ports using regex."""
    print(f"[*] Parsing raw text file: {txt_path}")
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"[!] Error reading text file: {e}", file=sys.stderr)
        return []

    ip_port_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}:[0-9]{1,5}\b')
    matches = ip_port_pattern.findall(content)
    unique_matches = set(matches)

    results = []
    for host in unique_matches:
        results.append({"host": host, "models": []})
        
    print(f"[+] Found {len(results)} endpoints in text file.")
    return results

# ---------------------------------------------------------------------------
# Validation via /api/tags
# ---------------------------------------------------------------------------
def _check_one(host: str, timeout: int) -> tuple[str, bool, list[str], int]:
    try:
        start_t = time.time()
        resp = requests.get(f"http://{host}/api/tags", timeout=timeout)
        resp.raise_for_status()
        elapsed = int((time.time() - start_t) * 1000)
        models = [m.get("name", "?") for m in resp.json().get("models", [])]
        return host, True, models, elapsed
    except Exception:
        return host, False, [], 99999


def validate(
    endpoints: list[dict], timeout: int = DEFAULT_TIMEOUT, workers: int = DEFAULT_WORKERS
) -> list[dict]:
    """Validate concurrently. Updates model lists with live data."""
    hosts = [ep["host"] for ep in endpoints]
    print(f"\n[*] Validating {len(hosts)} endpoints (timeout={timeout}s, workers={workers}) ...")
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_check_one, h, timeout): h for h in hosts}
        for i, fut in enumerate(as_completed(futures), 1):
            host, ok, models, speed = fut.result()
            sym = "✓" if ok else "✗"
            
            speed_str = ""
            if ok:
                if speed < 500:
                    speed_str = f"[Fast {speed}ms]"
                elif speed < 1500:
                    speed_str = f"[Med {speed}ms]"
                else:
                    speed_str = f"[Slow {speed}ms]"
            
            info = ", ".join(models[:4]) if models else ("unreachable" if not ok else "no models")
            if len(models) > 4:
                info += f" +{len(models) - 4}"
            print(f"  [{i}/{len(hosts)}] {sym} {host:<22} {speed_str:<14} {info}")
            if ok:
                results.append({"host": host, "models": models, "speed": speed})

    print(f"\n[+] Alive: {len(results)} / {len(hosts)}")
    
    # Sort by speed
    results.sort(key=lambda x: x.get("speed", 99999))
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_table(endpoints: list[dict], validated: bool = True) -> None:
    """Print formatted table of results."""
    label = "alive" if validated else "found"
    print(f"\n{'=' * 70}")
    print(f"  OLLAMA SERVERS ({len(endpoints)} {label})")
    print(f"  [*] Created specifically for g4f OllamaSwarm Provider")
    print("=" * 70)
    print(f"  {'ENDPOINT':<26} {'SPEED':<12} {'MODELS'}")
    print(f"  {'─' * 26} {'─' * 12} {'─' * 30}")
    
    model_counts: dict[str, int] = {}
    
    for ep in endpoints:
        models_list = ep.get("models", [])
        for m in models_list:
            model_counts[m] = model_counts.get(m, 0) + 1
            
        models = ", ".join(models_list[:3])
        if len(models_list) > 3:
            models += f" +{len(models_list) - 3}"
        speed = ep.get('speed', 0)
        speed_fmt = f"{speed}ms" if speed else "—"
        print(f"  http://{ep['host']:<19} {speed_fmt:<12} {models or '—'}")
    print("=" * 70)
    
    if model_counts:
        print(f"\n  [+] MODEL SUMMARY (Top 15):")
        sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (m, count) in enumerate(sorted_models[:15]):
            print(f"      {i+1}. {m:<30} : {count} servers")
        print()


def output_json(endpoints: list[dict]) -> None:
    """Print JSON output."""
    print(json.dumps({
        "count": len(endpoints),
        "servers": [
            {"url": f"http://{ep['host']}", "models": ep.get("models", [])}
            for ep in endpoints
        ],
    }, indent=2))


# ---------------------------------------------------------------------------
# Update OllamaSwarm cache
# ---------------------------------------------------------------------------
def update_swarm(hosts: list[str], path: Path = OLLAMA_SWARM_PATH) -> None:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from g4f.Provider.OllamaSwarm import _get_cached_servers, _save_servers_to_cache
    
    cached = _get_cached_servers()
    new_count = 0
    for h in hosts:
        url = f"http://{h}"
        if url not in cached:
            cached.append(url)
            new_count += 1
            
    if new_count > 0:
        _save_servers_to_cache(cached)
        print(f"\n[+] Added {new_count} new servers to cache (Total: {len(cached)})")
    else:
        print("\n[+] All alive servers are already in the cache.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="FOFA Ollama endpoint discovery tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data sources:
  --token TOKEN       Web scrape fofa.info (default, no API credits)
  --from-json FILE    Parse saved internal API JSON (use - for stdin)

Examples:
  %(prog)s --token "eyJhbG..." --pages 5 --save
  %(prog)s --from-json results.json --filter-model qwen
  curl '...' | %(prog)s --from-json - --min-models 2
""",
    )
    # Data source
    src = p.add_argument_group("data source")
    src.add_argument("--token", default=os.environ.get("FOFA_TOKEN"),
                     help="fofa_token cookie value (or FOFA_TOKEN env var)")
    src.add_argument("--from-json", metavar="FILE",
                     help="Read FOFA internal API JSON (file path or - for stdin)")
    src.add_argument("--from-har", metavar="FILE",
                     help="Parse endpoints directly from a saved FOFA .har file")
    src.add_argument("--from-txt", metavar="FILE",
                     help="Parse endpoints directly from any raw text file using regex")

    # Scraping options
    scrape = p.add_argument_group("scraping options")
    scrape.add_argument("--query", default=DEFAULT_QUERY, help="Base FOFA query")
    scrape.add_argument("--pages", type=int, default=DEFAULT_PAGES,
                        help=f"Pages to scrape per query, ~50 results each (default: {DEFAULT_PAGES})")
    scrape.add_argument("--deep-scan", action="store_true",
                        help="Dork Slicing: automatically run multiple queries by country/model to bypass free tier limits")
    scrape.add_argument("--country", default=None,
                        help="Filter by country code (e.g., US, CN, DE)")
    scrape.add_argument("--model", default=None,
                        help="Append a specific model to the query (e.g., deepseek)")

    # Validation & filtering
    filt = p.add_argument_group("validation & filtering")
    filt.add_argument("--no-validate", action="store_true",
                      help="Skip /api/tags validation (use HTML models only)")
    filt.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    filt.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    filt.add_argument("--filter-model", default=None,
                      help="Only keep endpoints with this model (substring match)")
    filt.add_argument("--min-models", type=int, default=0,
                      help="Only keep endpoints with at least N models")

    # Output
    out = p.add_argument_group("output")
    out.add_argument("--json", action="store_true", dest="output_json",
                     help="Output as JSON")
    out.add_argument("--save", action="store_true",
                     help="Save newly found servers to the provider cache (ollama_servers.json)")

    args = p.parse_args()

    # ------------------------------------------------------------------
    # 1. Acquire endpoints
    # ------------------------------------------------------------------
    endpoints: list[dict] = []

    if args.from_json:
        # Source: internal API JSON
        if args.from_json == "-":
            raw = sys.stdin.read()
        else:
            raw = Path(args.from_json).read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"[!] Invalid JSON: {exc}", file=sys.stderr)
            sys.exit(1)
        endpoints = parse_internal_json(data)
    elif args.from_har:
        # Source: HAR file
        endpoints = parse_har_file(args.from_har)
    elif args.from_txt:
        # Source: Raw Text File
        endpoints = parse_txt_file(args.from_txt)
    elif args.token:
        # Source: web scraping
        session = create_session(args.token)
        endpoints = scrape_all(
            session, 
            base_query=args.query, 
            pages=args.pages, 
            is_deep_scan=args.deep_scan,
            specific_country=args.country,
            specific_model=args.model
        )
    else:
        p.error(
            "Data source required.\n\n"
            "  --token TOKEN       Web scrape fofa.info\n"
            "  --from-json FILE    Parse saved API JSON\n\n"
            "Get token: fofa.info → DevTools → Cookies → fofa_token"
        )

    if not endpoints:
        print("[!] No results.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Validate (optional)
    # ------------------------------------------------------------------
    has_html_models = any(ep.get("models") for ep in endpoints)

    if not args.no_validate:
        endpoints = validate(endpoints, args.timeout, args.workers)
    elif has_html_models:
        print(f"[*] Skipping validation. Using {sum(1 for e in endpoints if e.get('models'))} "
              f"endpoints with HTML-extracted models.")

    if not endpoints:
        print("[!] No endpoints remaining after validation.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Filter
    # ------------------------------------------------------------------
    if args.filter_model:
        pat = args.filter_model.lower()
        before = len(endpoints)
        endpoints = [
            ep for ep in endpoints
            if any(pat in m.lower() for m in ep.get("models", []))
        ]
        print(f"[*] Filter model='{args.filter_model}': {len(endpoints)}/{before}")

    if args.min_models > 0:
        before = len(endpoints)
        endpoints = [
            ep for ep in endpoints
            if len(ep.get("models", [])) >= args.min_models
        ]
        print(f"[*] Filter min-models≥{args.min_models}: {len(endpoints)}/{before}")

    if not endpoints:
        print("[!] No endpoints remaining after filtering.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Output
    # ------------------------------------------------------------------
    if args.output_json:
        output_json(endpoints)
    else:
        print_table(endpoints, validated=not args.no_validate)

    # ------------------------------------------------------------------
    # 5. Update OllamaSwarm cache
    # ------------------------------------------------------------------
    if args.save:
        final_hosts = [ep["host"] for ep in endpoints]
        update_swarm(final_hosts)


if __name__ == "__main__":
    main()
