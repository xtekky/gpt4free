"""Live homepage checks: python -m etc.testing.test_provider_urls [APIRoute ...].

No API keys are needed. Run explicitly, separately from the offline unit suite.
A failed check can indicate bot protection or a local network problem, so it is
not sufficient evidence on its own to remove a provider.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import unittest
from urllib.parse import urlsplit

import requests

from g4f.Provider import ProviderLoader


def check_homepage(target):
    name, url = target
    try:
        parsed = urlsplit(url or "")
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return name, False, f"missing or invalid homepage: {url!r}"
        with requests.get(url, timeout=(5, 15), allow_redirects=True, stream=True) as response:
            detail = f"{url} -> HTTP {response.status_code} ({response.url})"
            return name, 200 <= response.status_code < 300, detail
    except Exception as error:
        return name, False, f"{url or name}: {type(error).__name__}: {error}"


class TestExtraProviderURLs(unittest.TestCase):
    # OpenaiTemplate is a configurable base class, not a service with a homepage.
    provider_names = [name for name in ProviderLoader.extra if name != "OpenaiTemplate"]

    def test_homepages_are_reachable(self):
        # Load providers serially: concurrent lazy imports can deadlock when
        # provider modules import each other. Only HTTP requests run in parallel.
        targets = []
        for name in self.provider_names:
            with self.subTest(provider=name):
                provider = ProviderLoader.from_name(name)
                targets.append((name, provider.url))
        with ThreadPoolExecutor(max_workers=6) as executor:
            for name, reachable, detail in executor.map(check_homepage, targets):
                with self.subTest(provider=name):
                    print(f"{name}: {detail}", flush=True)
                    self.assertTrue(reachable, detail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("providers", nargs="*", help="extra provider names; default: all services")
    args = parser.parse_args()
    unknown = set(args.providers) - set(TestExtraProviderURLs.provider_names)
    if unknown:
        parser.error(f"unknown service names: {', '.join(sorted(unknown))}")
    if args.providers:
        TestExtraProviderURLs.provider_names = args.providers
    unittest.main(argv=[parser.prog], verbosity=2)
