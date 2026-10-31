#!/usr/bin/env python3
"""Publish gpt4free Markdown documentation to the g4f.dev static site.

The public documentation is served from the ``g4f.dev`` checkout.  This module
copies the Markdown sources from ``gpt4free/docs`` into ``g4f.dev/docs`` and
renders the matching ``.html`` files with the template that already lives in
the website checkout.  It only uses the Python standard library so it can run
inside CI and in minimal sandbox images.

Typical use::

    bin/publish-docs --check
    bin/publish-docs
    bin/publish-docs --clean

``README.md`` is intentionally not copied by default because the website has
its own documentation hub.  Pass ``--include-readme`` to publish it as well.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "docs"
DEFAULT_TARGET = REPO_ROOT.parent / "g4f.dev" / "docs"
MANIFEST_NAME = ".g4f-docs-manifest.json"
README_NAME = "README.md"
MANIFEST_SCHEMA = 1


class DocsPublishError(Exception):
    """Raised when the documentation source or target is not usable."""


@dataclass
class PublishResult:
    """Summary of a publish or check operation."""

    copied: List[str] = field(default_factory=list)
    rendered: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    stale: List[str] = field(default_factory=list)
    manifest: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.copied or self.rendered or self.removed)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)")
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_HR_RE = re.compile(r"^ {0,3}([-*_])(?:\s*\1){2,}\s*$")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_COMMENT_START_RE = re.compile(r"^\s*<!--")
_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[\"']([^\"']*)[\"'])?\)"
)
_LINK_RE = re.compile(
    r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+[\"']([^\"']*)[\"'])?\)"
)
_AUTOLINK_RE = re.compile(r"<((?:https?://|mailto:)[^>\s]+)>")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC_RE = re.compile(
    r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)|"
    r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)"
)


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _rewrite_url(url: str) -> str:
    """Point Markdown links at their generated HTML counterparts."""

    if url.startswith(("http://", "https://", "mailto:", "tel:", "data:", "#", "/")):
        return url
    anchor = ""
    if "#" in url:
        url, anchor = url.split("#", 1)
        anchor = "#" + anchor
    if url in ("README.md", "./README.md", "../README.md"):
        return "index.html" + anchor
    if url.endswith(".md"):
        return url[:-3] + ".html" + anchor
    return url + anchor


def _render_inline(text: str) -> str:
    """Render inline Markdown without external dependencies."""

    if not text:
        return ""

    placeholders: List[str] = []

    def stash(value: str) -> str:
        token = "\x00{}\x00".format(len(placeholders))
        placeholders.append(value)
        return token

    text = _CODE_SPAN_RE.sub(
        lambda match: stash("<code>{}</code>".format(html.escape(match.group(1)))),
        text,
    )
    text = html.escape(text, quote=False)

    def image(match: "re.Match[str]") -> str:
        alt = match.group(1)
        src = _rewrite_url(match.group(2))
        title = match.group(3)
        title_attr = ' title="{}"'.format(html.escape(title, quote=True)) if title else ""
        return stash(
            '<img src="{}" alt="{}"{} />'.format(
                html.escape(src, quote=True), html.escape(alt, quote=True), title_attr
            )
        )

    def link(match: "re.Match[str]") -> str:
        label = match.group(1)
        href = _rewrite_url(match.group(2))
        title = match.group(3)
        title_attr = ' title="{}"'.format(html.escape(title, quote=True)) if title else ""
        return stash(
            '<a href="{}"{}>{}</a>'.format(
                html.escape(href, quote=True), title_attr, label
            )
        )

    def autolink(match: "re.Match[str]") -> str:
        url = match.group(1)
        href = _rewrite_url(url)
        return stash(
            '<a href="{}">{}</a>'.format(
                html.escape(href, quote=True), html.escape(url)
            )
        )

    text = _IMAGE_RE.sub(image, text)
    text = _LINK_RE.sub(link, text)
    text = _AUTOLINK_RE.sub(autolink, text)
    text = _STRIKE_RE.sub(r"<del>\1</del>", text)
    text = _BOLD_RE.sub(
        lambda match: "<strong>{}</strong>".format(match.group(1) or match.group(2)),
        text,
    )
    text = _ITALIC_RE.sub(
        lambda match: "<em>{}</em>".format(match.group(1) or match.group(2)),
        text,
    )

    for index, value in enumerate(placeholders):
        text = text.replace("\x00{}\x00".format(index), value)
    return text


def _is_block_start(lines: Sequence[str], index: int) -> bool:
    line = lines[index]
    if not line.strip():
        return True
    if _FENCE_RE.match(line) or _HEADING_RE.match(line) or _HR_RE.match(line):
        return True
    if _LIST_RE.match(line) or line.lstrip().startswith(">"):
        return True
    if _COMMENT_START_RE.match(line):
        return True
    if "|" in line and index + 1 < len(lines):
        if _TABLE_SEPARATOR_RE.match(lines[index + 1]):
            return True
    return False


def _split_table_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells = re.split(r"(?<!\\)\|", line)
    return [cell.strip().replace(r"\|", "|") for cell in cells]


def _table_alignments(separator: str) -> List[str]:
    return [
        "center"
        if cell.startswith(":") and cell.endswith(":")
        else "right"
        if cell.endswith(":")
        else "left"
        if cell.startswith(":")
        else ""
        for cell in _split_table_row(separator)
    ]


def _render_table(lines: Sequence[str], index: int) -> Tuple[str, int]:
    header = _split_table_row(lines[index])
    alignments = _table_alignments(lines[index + 1])
    index += 2
    rows: List[List[str]] = []
    while index < len(lines):
        line = lines[index]
        if not line.strip() or "|" not in line:
            break
        rows.append(_split_table_row(line))
        index += 1

    def cell(tag: str, value: str, column: int) -> str:
        align = alignments[column] if column < len(alignments) else ""
        style = ' style="text-align: {}"'.format(align) if align else ""
        return "<{tag}{style}>{value}</{tag}>".format(
            tag=tag, style=style, value=_render_inline(value)
        )

    out = ["<table>", "<thead>", "<tr>"]
    out.extend(cell("th", value, column) for column, value in enumerate(header))
    out.extend(["</tr>", "</thead>"])
    if rows:
        out.append("<tbody>")
        for row in rows:
            out.append("<tr>")
            out.extend(cell("td", value, column) for column, value in enumerate(row))
            out.append("</tr>")
        out.append("</tbody>")
    out.append("</table>")
    return "\n".join(out), index


def _render_list(lines: Sequence[str], index: int) -> Tuple[str, int]:
    first = _LIST_RE.match(lines[index])
    if first is None:
        return "", index
    base_indent = len(first.group(1))
    ordered = first.group(2)[0].isdigit()
    items: List[str] = []

    while index < len(lines):
        match = _LIST_RE.match(lines[index])
        if match is None:
            break
        indent = len(match.group(1))
        marker = match.group(2)
        if indent != base_indent or marker[0].isdigit() != ordered:
            break

        content_indent = indent + len(marker) + 1
        item_lines = [match.group(3)]
        index += 1

        while index < len(lines):
            line = lines[index]
            if not line.strip():
                lookahead = index + 1
                while lookahead < len(lines) and not lines[lookahead].strip():
                    lookahead += 1
                if lookahead >= len(lines):
                    index = lookahead
                    break
                next_line = lines[lookahead]
                next_match = _LIST_RE.match(next_line)
                next_indent = _line_indent(next_line)
                if next_match is not None and next_indent == base_indent:
                    index = lookahead
                    break
                if next_indent >= content_indent:
                    item_lines.append("")
                    index += 1
                    continue
                index = lookahead
                break

            next_match = _LIST_RE.match(line)
            next_indent = _line_indent(line)
            if next_match is not None and next_indent <= base_indent:
                break
            if next_indent >= content_indent:
                item_lines.append(line[content_indent:])
                index += 1
            elif next_indent > base_indent:
                item_lines.append(line[next_indent:])
                index += 1
            elif not _is_block_start(lines, index):
                item_lines.append(line)
                index += 1
            else:
                break

        items.append(_render_blocks(item_lines).strip())

    tag = "ol" if ordered else "ul"
    body = "".join("<li>{}</li>".format(item) for item in items)
    return "<{0}>{1}</{0}>".format(tag, body), index


def _render_blocks(lines: Sequence[str]) -> str:
    out: List[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        if _COMMENT_START_RE.match(line):
            while index < len(lines) and "-->" not in lines[index]:
                index += 1
            if index < len(lines):
                index += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence is not None:
            fence_text = fence.group(1)
            fence_char = fence_text[0]
            fence_length = len(fence_text)
            language = fence.group(2)
            index += 1
            code_lines: List[str] = []
            closing = re.compile(
                r"^ {0,3}" + re.escape(fence_char) + r"{%d,}\s*$" % fence_length
            )
            while index < len(lines):
                if closing.match(lines[index]):
                    index += 1
                    break
                code_lines.append(lines[index])
                index += 1
            class_attr = (
                ' class="language-{}"'.format(html.escape(language))
                if language
                else ""
            )
            out.append(
                "<pre><code{}>{}</code></pre>".format(
                    class_attr, html.escape("\n".join(code_lines))
                )
            )
            continue

        heading = _HEADING_RE.match(line)
        if heading is not None:
            level = len(heading.group(1))
            out.append(
                "<h{0}>{1}</h{0}>".format(level, _render_inline(heading.group(2)))
            )
            index += 1
            continue

        if _HR_RE.match(line):
            out.append("<hr>")
            index += 1
            continue

        if (
            "|" in line
            and index + 1 < len(lines)
            and _TABLE_SEPARATOR_RE.match(lines[index + 1])
        ):
            table, index = _render_table(lines, index)
            out.append(table)
            continue

        if line.lstrip().startswith(">"):
            quoted: List[str] = []
            while index < len(lines) and lines[index].lstrip().startswith(">"):
                quoted.append(re.sub(r"^\s*>\s?", "", lines[index]))
                index += 1
            out.append("<blockquote>\n{}\n</blockquote>".format(_render_blocks(quoted)))
            continue

        if _LIST_RE.match(line):
            rendered_list, index = _render_list(lines, index)
            out.append(rendered_list)
            continue

        if line.lstrip().startswith("<"):
            raw: List[str] = []
            while index < len(lines) and lines[index].strip():
                raw.append(lines[index])
                index += 1
            out.append("\n".join(raw))
            continue

        paragraph: List[str] = []
        while index < len(lines) and not _is_block_start(lines, index):
            paragraph.append(lines[index])
            index += 1
        if not paragraph:
            paragraph = [line]
            index += 1
        rendered: List[str] = []
        for position, paragraph_line in enumerate(paragraph):
            hard_break = paragraph_line.endswith("  ")
            rendered.append(_render_inline(paragraph_line.rstrip()))
            if position < len(paragraph) - 1:
                rendered.append("<br>" if hard_break else "\n")
        out.append("<p>{}</p>".format("".join(rendered)))

    return "\n".join(out)


def markdown_to_html(text: str) -> str:
    """Convert the subset of GitHub Markdown used by the docs to HTML."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _render_blocks(text.split("\n"))


def extract_title(text: str) -> str:
    """Return the first heading, or a neutral fallback."""

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match is None:
            continue
        title = re.sub(r"[`*_~]", "", match.group(2)).strip()
        if title:
            return title
    return "GPT4Free Documentation"


def render_document(
    markdown: str, template: Optional[str], filename: str
) -> str:
    """Render one Markdown document into a complete HTML page."""

    title = extract_title(markdown)
    article = markdown_to_html(markdown)
    if template:
        return (
            template.replace("{{ title }}", html.escape(title))
            .replace("{{ article }}", article)
            .replace("{{ filename }}", html.escape(filename, quote=True))
        )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8" />'
        "<title>{}</title></head><body>"
        '<main class="markdown-body">\n{}\n</main>'
        "</body></html>\n"
    ).format(html.escape(title), article)


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def _html_relative_path(relative: Path) -> str:
    if relative.as_posix() == README_NAME:
        return "index.html"
    return relative.with_suffix(".html").as_posix()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_manifest(target: Path) -> Dict[str, dict]:
    path = target / MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(_read(path))
    except (OSError, ValueError):
        return {}
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        return {}
    return {
        str(name): value
        for name, value in files.items()
        if isinstance(value, dict)
    }


def _select_documents(
    source: Path, include_readme: bool
) -> List[Tuple[str, Path]]:
    if not source.is_dir():
        raise DocsPublishError("documentation source does not exist: {}".format(source))
    documents: List[Tuple[str, Path]] = []
    for path in sorted(source.rglob("*.md")):
        relative = path.relative_to(source)
        if relative.as_posix() == README_NAME and not include_readme:
            continue
        documents.append((relative.as_posix(), path))
    return documents


def publish_docs(
    source: Path = DEFAULT_SOURCE,
    target: Path = DEFAULT_TARGET,
    *,
    template_path: Optional[Path] = None,
    include_readme: bool = False,
    clean: bool = False,
    check: bool = False,
    dry_run: bool = False,
    render_html: bool = True,
    create_target: bool = False,
) -> PublishResult:
    """Copy docs into *target* and render HTML pages.

    ``check`` never writes to disk and reports differences in ``stale``.
    """

    source = Path(source)
    target = Path(target)

    if not target.is_dir():
        if not create_target or check or dry_run:
            raise DocsPublishError(
                "documentation target does not exist: {}. "
                "Clone the g4f.dev checkout or pass --create-target.".format(target)
            )
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)

    documents = _select_documents(source, include_readme)
    template = None
    if render_html:
        resolved_template = template_path or (target / "template.html")
        if resolved_template.is_file():
            template = _read(resolved_template)
        elif template_path is not None:
            raise DocsPublishError(
                "documentation template does not exist: {}".format(resolved_template)
            )

    result = PublishResult(manifest=str(target / MANIFEST_NAME))
    old_manifest = _load_manifest(target)
    desired_names = {name for name, _ in documents}
    new_manifest: Dict[str, dict] = {}

    for name, source_path in documents:
        markdown = _read(source_path)
        markdown_target = target / name
        html_name = _html_relative_path(Path(name)) if render_html else ""
        html_target = target / html_name if html_name else None
        entry = {
            "markdown_sha256": _sha256(markdown),
            "html": html_name,
        }

        markdown_current = markdown_target.is_file() and _read(markdown_target) == markdown
        if markdown_current:
            result.unchanged.append(name)
        elif check:
            result.stale.append(name)
        elif not dry_run:
            _write(markdown_target, markdown)
            result.copied.append(name)
        else:
            result.copied.append(name)

        if html_target is not None:
            rendered = render_document(markdown, template, html_name)
            entry["html_sha256"] = _sha256(rendered)
            html_current = html_target.is_file() and _read(html_target) == rendered
            if html_current:
                result.unchanged.append(html_name)
            elif check:
                result.stale.append(html_name)
            elif not dry_run:
                _write(html_target, rendered)
                result.rendered.append(html_name)
            else:
                result.rendered.append(html_name)

        new_manifest[name] = entry

    removed_names: List[str] = []
    if clean:
        for name, entry in old_manifest.items():
            if name in desired_names:
                continue
            removed_names.append(name)
            removed_html = str(entry.get("html") or "")
            for relative in [name, removed_html]:
                if not relative:
                    continue
                old_path = target / relative
                if not old_path.is_file():
                    continue
                if check:
                    result.stale.append(relative)
                elif not dry_run:
                    old_path.unlink()
                    result.removed.append(relative)
                else:
                    result.removed.append(relative)
    else:
        for name, entry in old_manifest.items():
            if name not in new_manifest:
                new_manifest[name] = entry

    manifest_payload = {
        "schema": MANIFEST_SCHEMA,
        "source": str(source.resolve()),
        "files": new_manifest,
    }
    manifest_text = json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    if not check and not dry_run:
        manifest_target = target / MANIFEST_NAME
        previous = _read(manifest_target) if manifest_target.is_file() else ""
        if previous != manifest_text:
            _write(manifest_target, manifest_text)

    if check:
        result.stale.sort()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy gpt4free Markdown docs into the g4f.dev checkout and "
            "generate the matching HTML pages."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Markdown source directory (default: %(default)s)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="g4f.dev docs directory (default: %(default)s)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="HTML template (default: <target>/template.html, or built in)",
    )
    parser.add_argument(
        "--include-readme",
        action="store_true",
        help="Also publish README.md as README.md/index.html",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove files that were published earlier but no longer exist",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 when the published docs are stale",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Only copy Markdown and skip HTML rendering",
    )
    parser.add_argument(
        "--create-target",
        action="store_true",
        help="Create the target directory when it does not exist",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors and the final status line",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = publish_docs(
            args.source,
            args.target,
            template_path=args.template,
            include_readme=args.include_readme,
            clean=args.clean,
            check=args.check,
            dry_run=args.dry_run,
            render_html=not args.no_html,
            create_target=args.create_target,
        )
    except DocsPublishError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    if args.check:
        if result.stale:
            print(
                "documentation is out of date: {}".format(", ".join(result.stale)),
                file=sys.stderr,
            )
            return 1
        print("documentation is up to date")
        return 0

    action = "Would update" if args.dry_run else "Published"
    if not args.quiet:
        for name in result.copied:
            print("copied    {}".format(name))
        for name in result.rendered:
            print("rendered  {}".format(name))
        for name in result.removed:
            print("removed   {}".format(name))
    print(
        "{} docs: {} copied, {} rendered, {} removed, {} unchanged".format(
            action,
            len(result.copied),
            len(result.rendered),
            len(result.removed),
            len(result.unchanged),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
