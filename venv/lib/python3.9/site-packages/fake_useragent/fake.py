import random
from threading import Lock

from fake_useragent import settings
from fake_useragent.errors import FakeUserAgentError
from fake_useragent.log import logger
from fake_useragent.utils import load, load_cached, str_types, update


class FakeUserAgent:
    def __init__(
        self,
        use_external_data=False,
        cache_path=settings.DB,
        fallback=None,
        browsers=["chrome", "edge", "internet explorer", "firefox", "safari", "opera"],
        verify_ssl=True,
        safe_attrs=tuple(),
    ):
        assert isinstance(
            use_external_data, bool
        ), "use_external_data must be True or False"

        self.use_external_data = use_external_data

        assert isinstance(cache_path, str_types), "cache_path must be string or unicode"

        self.cache_path = cache_path

        if fallback is not None:
            assert isinstance(fallback, str_types), "fallback must be string or unicode"

        self.fallback = fallback

        assert isinstance(browsers, (list, str)), "browsers must be list or string"

        self.browsers = browsers

        assert isinstance(verify_ssl, bool), "verify_ssl must be True or False"

        self.verify_ssl = verify_ssl

        assert isinstance(
            safe_attrs, (list, set, tuple)
        ), "safe_attrs must be list\\tuple\\set of strings or unicode"

        if safe_attrs:
            str_types_safe_attrs = [isinstance(attr, str_types) for attr in safe_attrs]

            assert all(
                str_types_safe_attrs
            ), "safe_attrs must be list\\tuple\\set of strings or unicode"

        self.safe_attrs = set(safe_attrs)

        # initial empty data
        self.data_browsers = {}

        self.load()

    def load(self):
        try:
            with self.load.lock:
                if self.use_external_data:
                    # Use external resource to retrieve browser data
                    self.data_browsers = load_cached(
                        self.cache_path,
                        self.browsers,
                        verify_ssl=self.verify_ssl,
                    )
                else:
                    # By default we will try to load our local file
                    self.data_browsers = load(
                        self.browsers,
                        verify_ssl=self.verify_ssl,
                    )
        except FakeUserAgentError:
            if self.fallback is None:
                raise
            else:
                logger.warning(
                    "Error occurred during fetching data, "
                    "but was suppressed with fallback.",
                )

    load.lock = Lock()

    def update(self, use_external_data=None):
        with self.update.lock:
            if use_external_data is not None:
                assert isinstance(
                    use_external_data, bool
                ), "use_external_data must be True or False"

                self.use_external_data = use_external_data

            # Update tmp cache file from external data source
            if self.use_external_data:
                update(
                    self.cache_path,
                    self.browsers,
                    verify_ssl=self.verify_ssl,
                )

            self.load()

    update.lock = Lock()

    def __getitem__(self, attr):
        return self.__getattr__(attr)

    def __getattr__(self, attr):
        if attr in self.safe_attrs:
            return super(UserAgent, self).__getattr__(attr)

        try:
            for value, replacement in settings.REPLACEMENTS.items():
                attr = attr.replace(value, replacement)

            attr = attr.lower()

            if attr == "random":
                # Pick a random browser from the browsers argument list
                browser_name = random.choice(self.browsers)
            else:
                browser_name = settings.SHORTCUTS.get(attr, attr)

            # Pick a random user-agent string for a specific browser
            return random.choice(self.data_browsers[browser_name])
        except (KeyError, IndexError):
            if self.fallback is None:
                raise FakeUserAgentError(
                    f"Error occurred during getting browser: {attr}"
                )  # noqa
            else:
                logger.warning(
                    f"Error occurred during getting browser: {attr}, "
                    "but was suppressed with fallback.",
                )

                return self.fallback

    @property
    def chrome(self):
        return self.__getattr__("chrome")

    @property
    def googlechrome(self):
        return self.chrome

    @property
    def edge(self):
        return self.__getattr__("edge")

    @property
    def ie(self):
        return self.__getattr__("ie")

    @property
    def internetexplorer(self):
        return self.ie

    @property
    def msie(self):
        return self.ie

    @property
    def firefox(self):
        return self.__getattr__("firefox")

    @property
    def ff(self):
        return self.firefox

    @property
    def safari(self):
        return self.__getattr__("safari")

    @property
    def opera(self):
        return self.__getattr__("opera")

    @property
    def random(self):
        return self.__getattr__("random")


# common alias
UserAgent = FakeUserAgent
