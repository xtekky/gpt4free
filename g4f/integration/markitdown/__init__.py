import re
import sys
import io
from typing import List, Union, BinaryIO, Optional, Any
from markitdown import MarkItDown as BaseMarkItDown
from markitdown._stream_info import StreamInfo
from markitdown._base_converter import DocumentConverterResult

from markitdown._exceptions import (
    FileConversionException,
    UnsupportedFormatException,
    FailedConversionAttempt,
)

from ._audio_converter import AudioConverter
from ._image_converter import ImageConverter
from ._youtube_converter import YouTubeConverter

GITHUB_SKIP_LINES = [
    "[Skip to content](#start-of-content)",
    "## Navigation Menu",
    "Appearance settings",
    "* Platform",
    "+ AI CODE CREATION",
    "- [GitHub CopilotWrite better code with AI](https://github.com/features/copilot)",
    "- [GitHub Copilot appDirect agents from issue to merge](https://github.com/features/ai/github-app)",
    "- [MCP RegistryIntegrate external tools](https://github.com/mcp)",
    "+ DEVELOPER WORKFLOWS",
    "- [ActionsAutomate any workflow](https://github.com/features/actions)",
    "- [CodespacesInstant dev environments](https://github.com/features/codespaces)",
    "- [IssuesPlan and track work](https://github.com/features/issues)",
    "- [Code ReviewManage code changes](https://github.com/features/code-review)",
    "- [Code QualityEnforce quality at merge](https://github.com/features/code-quality)",
    "+ APPLICATION SECURITY",
    "- [GitHub Advanced SecurityFind and fix vulnerabilities](https://github.com/security/advanced-security)",
    "- [Code securitySecure your code as you build](https://github.com/security/advanced-security/code-security)",
    "- [Secret protectionStop leaks before they start](https://github.com/security/advanced-security/secret-protection)",
    "+ EXPLORE",
    "- [Why GitHub](https://github.com/why-github)",
    "- [Documentation](https://docs.github.com)",
    "- [Blog](https://github.blog)",
    "- [Changelog](https://github.blog/changelog)",
    "- [Marketplace](https://github.com/marketplace)",
    "[View all features](https://github.com/features)",
    "* Solutions",
    "+ BY COMPANY SIZE",
    "- [Enterprises](https://github.com/enterprise)",
    "- [Small and medium teams](https://github.com/team)",
    "- [Startups](https://github.com/enterprise/startups)",
    "- [Nonprofits](https://github.com/solutions/industry/nonprofits)",
    "+ BY USE CASE",
    "- [App Modernization](https://github.com/solutions/use-case/app-modernization)",
    "- [DevSecOps](https://github.com/solutions/use-case/devsecops)",
    "- [DevOps](https://github.com/solutions/use-case/devops)",
    "- [CI/CD](https://github.com/solutions/use-case/ci-cd)",
    "- [View all use cases](https://github.com/solutions/use-case)",
    "+ BY INDUSTRY",
    "- [Healthcare](https://github.com/solutions/industry/healthcare)",
    "- [Financial services](https://github.com/solutions/industry/financial-services)",
    "- [Manufacturing](https://github.com/solutions/industry/manufacturing)",
    "- [Government](https://github.com/solutions/industry/government)",
    "- [View all industries](https://github.com/solutions/industry)",
    "[View all solutions](https://github.com/solutions)",
    "* Resources",
    "+ EXPLORE BY TOPIC",
    "- [AI](https://github.com/resources/articles?topic=ai)",
    "- [Software Development](https://github.com/resources/articles?topic=software-development)",
    "- [DevOps](https://github.com/resources/articles?topic=devops)",
    "- [Security](https://github.com/resources/articles?topic=security)",
    "- [View all topics](https://github.com/resources/articles)",
    "+ EXPLORE BY TYPE",
    "- [Customer stories](https://github.com/customer-stories)",
    "- [Events & webinars](https://github.com/resources/events)",
    "- [Ebooks & reports](https://github.com/resources/whitepapers)",
    "- [Business insights](https://github.com/solutions/executive-insights)",
    "- [GitHub Skills](https://skills.github.com)",
    "+ SUPPORT & SERVICES",
    "- [Documentation](https://docs.github.com)",
    "- [Customer support](https://support.github.com)",
    "- [Community forum](https://github.com/orgs/community/discussions)",
    "- [Trust center](https://github.com/trust-center)",
    "- [Partners](https://github.com/partners)",
    "[View all resources](https://github.com/resources)",
    "* Open Source",
    "+ COMMUNITY",
    "- [GitHub SponsorsFund open source developers](https://github.com/open-source/sponsors)",
    "+ PROGRAMS",
    "- [Security Lab](https://securitylab.github.com)",
    "- [Maintainer Community](https://maintainers.github.com)",
    "- [GitHub Stars](https://stars.github.com)",
    "- [Archive Program](https://archiveprogram.github.com)",
    "+ REPOSITORIES",
    "- [Topics](https://github.com/topics)",
    "- [Trending](https://github.com/trending)",
    "- [Collections](https://github.com/collections)",
    "* Enterprise",
    "+ ENTERPRISE SOLUTIONS",
    "- [Enterprise platformAI-powered developer platform](https://github.com/enterprise)",
    "+ AVAILABLE ADD-ONS",
    "- [GitHub Advanced SecurityEnterprise-grade security features](https://github.com/security/advanced-security)",
    "- [Copilot for BusinessEnterprise-grade AI features](https://github.com/features/copilot/copilot-business)",
    "- [Premium SupportEnterprise-grade 24/7 support](https://github.com/enterprise/premium-support)",
    "* [Pricing](https://github.com/pricing)",
    "Search`/`",
    "Appearance settings",
    "You signed in with another tab or window. Reload to refresh your session.",
    "You signed out in another tab or window. Reload to refresh your session.",
    "You switched accounts on another tab or window. Reload to refresh your session.",
    "Dismiss alert",
    "{{ message }}",
    "Public",
    "* ### Uh oh!"
    "There was an error while loading. Please reload this page.",
    "* [Notifications]",
    "Choose two branches to see what’s changed or to start a new pull request.",
    "If you need to, you can also  compare across forks",
    "or",
    "[learn more about diff comparisons](https://docs.github.com/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-comparing-branches-in-pull-requests#three-dot-and-two-dot-git-diff-comparisons).",
    "# Open a pull request",
    "Create a new pull request by comparing changes across two branches. If you need to, you can also  compare across forks",
    "[Learn more about diff comparisons here](https://docs.github.com/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-comparing-branches-in-pull-requests#three-dot-and-two-dot-git-diff-comparisons).",
    "Failed to load repositories. Confirm that selected base ref is valid, then try again.",
    "Loading",
    "### Uh oh!",
    "Could not load branches",
    "Could not load tags",
    "Nothing to show",
    "Loading",
    "[{{ refName }}",
    "...",
    "Failed to load repositories. Confirm that selected head ref is valid, then try again.",
    "### Uh oh!",
    "There was an error while loading. Please reload this page.",
    "### This comparison is taking too long to generate.",
    "Unfortunately it looks like we can’t render this comparison for you right now. It might be too big, or there might be something weird with your repository.",
    "You can try running this command locally to see the comparison on your machine:",
    "There was an error while loading. Please reload this page.",
    "### Footer navigation",
    "* [Terms](https://docs.github.com/site-policy/github-terms/github-terms-of-service)",
    "* [Privacy](https://docs.github.com/site-policy/privacy-policies/github-privacy-statement)",
    "* [Security](https://github.com/security)",
    "* [Status](https://www.githubstatus.com/)",
    "* [Community](https://github.community/)",
    "* [Docs](https://docs.github.com/)",
    "* [Contact](https://support.github.com?tags=dotcom-footer)",
    "* Manage cookies",
    "* Do not share my personal information",
    "You can’t perform that action at this time.",
    "* ### Uh oh!",
    "/",
    ".",
    "Retry",
    "* [FeedbackPreview](https://gh.io/issues-sidebar-feedback)",
    "* Collapse sidebar",
    "Search Issues",
    "Search"
]

GITHUB_STARTS_WITH = [
    "[Sign in]",
    "[Sign up]",
    "default](",
    "* [Notifications]",
    "* [Views]",
    "* [Projects]",
    "* [Milestones]",
    "* [Labels]",
    "* [Insights]",
    "[Report repository]"
]

class MarkItDown(BaseMarkItDown):
    """(In preview) An extremely simple text-based document reader, suitable for LLM use.
    This reader will convert common file-types or webpages to Markdown."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.register_converter(AudioConverter())
        self.register_converter(ImageConverter())
        self.register_converter(YouTubeConverter())

    def _convert(
        self, *, file_stream: BinaryIO, stream_info_guesses: List[StreamInfo], **kwargs
    ) -> DocumentConverterResult:
        res: Union[None, DocumentConverterResult] = None

        # Keep track of which converters throw exceptions
        failed_attempts: List[FailedConversionAttempt] = []

        # Create a copy of the page_converters list, sorted by priority.
        # We do this with each call to _convert because the priority of converters may change between calls.
        # The sort is guaranteed to be stable, so converters with the same priority will remain in the same order.
        sorted_registrations = sorted(self._converters, key=lambda x: x.priority)

        # Remember the initial stream position so that we can return to it
        cur_pos = file_stream.tell()

        for stream_info in stream_info_guesses + [StreamInfo()]:
            for converter_registration in sorted_registrations:
                converter = converter_registration.converter
                # Sanity check -- make sure the cur_pos is still the same
                assert (
                    cur_pos == file_stream.tell()
                ), f"File stream position should NOT change between guess iterations"

                _kwargs = {k: v for k, v in kwargs.items()}

                # Copy any additional global options
                if "llm_client" not in _kwargs and self._llm_client is not None:
                    _kwargs["llm_client"] = self._llm_client

                if "llm_model" not in _kwargs and self._llm_model is not None:
                    _kwargs["llm_model"] = self._llm_model

                if "style_map" not in _kwargs and self._style_map is not None:
                    _kwargs["style_map"] = self._style_map

                if "exiftool_path" not in _kwargs and self._exiftool_path is not None:
                    _kwargs["exiftool_path"] = self._exiftool_path

                # Add the list of converters for nested processing
                _kwargs["_parent_converters"] = self._converters

                # Add legaxy kwargs
                if stream_info is not None:
                    if stream_info.extension is not None:
                        _kwargs["file_extension"] = stream_info.extension

                    if stream_info.url is not None:
                        _kwargs["url"] = stream_info.url

                # Check if the converter will accept the file, and if so, try to convert it
                _accepts = False
                try:
                    _accepts = converter.accepts(file_stream, stream_info, **_kwargs)
                except NotImplementedError:
                    pass

                # accept() should not have changed the file stream position
                assert (
                    cur_pos == file_stream.tell()
                ), f"{type(converter).__name__}.accept() should NOT change the file_stream position"

                # Attempt the conversion
                if _accepts:
                    try:
                        res = converter.convert(file_stream, stream_info, **_kwargs)
                    except Exception:
                        failed_attempts.append(
                            FailedConversionAttempt(
                                converter=converter, exc_info=sys.exc_info()
                            )
                        )
                    finally:
                        file_stream.seek(cur_pos)

                if res is not None:
                    if isinstance(res.text_content, str):
                        # Normalize the content
                        res.text_content = "\n".join(
                            [
                                line.rstrip()
                                for line in re.split(r"\r?\n", res.text_content)
                            ]
                        )
                        res.text_content = re.sub(r"\n{3,}", "\n\n", res.text_content)
                    return res

        # If we got this far without success, report any exceptions
        if len(failed_attempts) > 0:
            raise FileConversionException(attempts=failed_attempts)

        # Nothing can handle it!
        raise UnsupportedFormatException(
            f"Could not convert stream to Markdown. No converter attempted a conversion, suggesting that the filetype is simply not supported."
        )

    def convert_stream(
        self,
        stream: BinaryIO,
        *,
        stream_info: Optional[StreamInfo] = None,
        file_extension: Optional[str] = None,  # Deprecated -- use stream_info
        url: Optional[str] = None,  # Deprecated -- use stream_info
        **kwargs: Any,
    ) -> DocumentConverterResult:
        guesses: List[StreamInfo] = []

        # Do we have anything on which to base a guess?
        base_guess = None
        if stream_info is not None or file_extension is not None or url is not None:
            # Start with a non-Null base guess
            if stream_info is None:
                base_guess = StreamInfo()
            else:
                base_guess = stream_info

            if file_extension is not None:
                # Deprecated -- use stream_info
                assert base_guess is not None  # for mypy
                base_guess = base_guess.copy_and_update(extension=file_extension)

            if url is not None:
                # Deprecated -- use stream_info
                assert base_guess is not None  # for mypy
                base_guess = base_guess.copy_and_update(url=url)

        # Check if we have a seekable stream. If not, load the entire stream into memory.
        if not hasattr(stream, "seekable") or not stream.seekable():
            buffer = io.BytesIO()
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                buffer.write(chunk)
            buffer.seek(0)
            stream = buffer

        # Add guesses based on stream content
        guesses = self._get_stream_info_guesses(
            file_stream=stream, base_guess=base_guess or StreamInfo()
        )
        return self._convert(file_stream=stream, stream_info_guesses=guesses, **kwargs)

    @staticmethod
    def _convert_github_url_to_raw(url: str) -> str:
        """Convert a github.com URL to a raw.githubusercontent.com content URL.

        Handles the following patterns:
        - https://github.com/{owner}/{repo}/blob/{ref}/{path}
            -> https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}
        - https://github.com/{owner}/{repo}/raw/{ref}/{path}
            -> https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}
        - https://gist.github.com/{user}/{gist_id}
            -> https://gist.githubusercontent.com/{user}/{gist_id}/raw
        - URLs already pointing to raw.githubusercontent.com or
          gist.githubusercontent.com are returned unchanged.

        Tree URLs (directories) and repository root URLs cannot be converted
        to a single raw file and are returned unchanged so the caller can
        decide how to handle them.
        """
        if url is None:
            raise ValueError("url must not be None")

        # Already raw -- nothing to do
        if url.startswith(
            (
                "https://raw.githubusercontent.com/",
                "https://gist.githubusercontent.com/",
            )
        ):
            return url

        # Gist URLs
        m = re.match(
            r"^https?://gist\.github\.com/([^/]+)/([0-9a-fA-F]+)(?:/.*)?$",
            url,
        )
        if m:
            user, gist_id = m.group(1), m.group(2)
            return f"https://gist.githubusercontent.com/{user}/{gist_id}/raw"

        # github.com/{owner}/{repo}/blob/{ref}/{path}
        m = re.match(
            r"^https?://github\.com/([^/]+)/([^/]+)/(?:blob|raw)/([^/]+)/(.+?)(?:[?#].*)?$",
            url,
        )
        if m:
            owner, repo, ref, path = (m.group(1), m.group(2), m.group(3), m.group(4))
            # Strip a trailing slash if any
            path = path.rstrip("/")
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"

        # Tree (directory) URLs and repo roots: cannot map to a single raw file
        return url

    def convert_url(
        self,
        url: str,
        *,
        stream_info: Optional[StreamInfo] = None,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        if url is None or not isinstance(url, str) or url.strip() == "":
            raise ValueError("url must be a non-empty string")
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        if url.startswith("https://github.com/"):
            # Special case for GitHub URLs -- convert to raw content URL
            url = self._convert_github_url_to_raw(url)
        result = super().convert_url(url, stream_info=stream_info, **kwargs)
        if url.startswith("https://github.com/"):
            result.text_content = re.sub(r"Additional navigation options[\s\S]+---", "", result.text_content)
            result.text_content = re.sub(r"\* \[Security and quality[\s\S]+\* Collapse sidebar", "", result.text_content)
            lines = []
            empty = 0
            for l in [l for l in result.text_content.splitlines() if l.strip() not in GITHUB_SKIP_LINES and not l.strip().startswith(tuple(GITHUB_STARTS_WITH))]:
                striped = l.strip()
                if striped == "":
                    empty += 1
                else:
                    empty = 0
                if empty > 2:
                    continue
                else:
                    lines.append(l)
            result.text_content = "\n".join(lines).strip()
        return result
