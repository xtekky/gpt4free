from __future__ import annotations

import re
import os
import json
from pathlib import Path
from typing import Iterator, Optional, AsyncIterator
from aiohttp import ClientSession, ClientError, ClientResponse, ClientTimeout
import urllib.parse
import time
import zipfile
import asyncio
import hashlib
import base64
import tempfile
import shutil

try:
    import PyPDF2
    from PyPDF2.errors import PdfReadError
    has_pypdf2 = True
except ImportError:
    has_pypdf2 = False
try:
    import pdfplumber
    has_pdfplumber = True
except ImportError:
    has_pdfplumber = False
try:
    from pdfminer.high_level import extract_text
    has_pdfminer = True
except ImportError:
    has_pdfminer = False
try:
    from docx import Document
    has_docx = True
except ImportError:
    has_docx = False
try:
    import docx2txt
    has_docx2txt = True
except ImportError:
    has_docx2txt = False
try:
    from odf.opendocument import load
    from odf.text import P
    has_odfpy = True
except ImportError:
    has_odfpy = False
try:
    import ebooklib
    from ebooklib import epub
    has_ebooklib = True
except ImportError:
    has_ebooklib = False
try:
    import pandas as pd
    has_openpyxl = True
except ImportError:
    has_openpyxl = False
try:
    import spacy
    has_spacy = True
except:
    has_spacy = False
try:
    from bs4 import BeautifulSoup
    has_beautifulsoup4 = True
except ImportError:
    has_beautifulsoup4 = False
try:
    from markitdown import MarkItDown
    has_markitdown = True
except ImportError:
    has_markitdown = False

from .web_search import scrape_text
from ..files import secure_filename, get_bucket_dir
from ..image import is_allowed_extension
from ..requests.aiohttp import get_connector
from ..providers.asyncio import to_sync_generator
from ..errors import MissingRequirementsError
from .. import debug

PLAIN_FILE_EXTENSIONS = ["txt", "xml", "json", "js", "har", "sh", "py", "php", "css", "yaml", "sql", "log", "csv", "twig", "md", "arc"]
PLAIN_CACHE = "plain.cache"
DOWNLOADS_FILE = "downloads.json"
FILE_LIST = "files.txt"

def supports_filename(filename: str):
    if filename.endswith(".pdf"):
        if has_pypdf2:
            return True
        elif has_pdfplumber:
            return True
        elif has_pdfminer:
            return True
        raise MissingRequirementsError(f'Install "pypdf2" requirements | pip install -U g4f[files]')
    elif filename.endswith(".docx"):
        if has_docx:
            return True
        elif has_docx2txt:
            return True
        raise MissingRequirementsError(f'Install "docx" requirements | pip install -U g4f[files]')
    elif has_odfpy and filename.endswith(".odt"):
        return True
    elif has_ebooklib and filename.endswith(".epub"):
        return True
    elif has_openpyxl and filename.endswith(".xlsx"):
        return True
    elif filename.endswith(".html"):
        if not has_beautifulsoup4:
            raise MissingRequirementsError(f'Install "beautifulsoup4" requirements | pip install -U g4f[files]')
        return True
    elif filename.endswith(".zip"):
        return True
    elif filename.endswith("package-lock.json") and filename != FILE_LIST:
        return False
    else:
        extension = os.path.splitext(filename)[1][1:]
        if extension in PLAIN_FILE_EXTENSIONS:
            return True
    return False

def spacy_refine_chunks(source_iterator):
    if not has_spacy:
        raise MissingRequirementsError(f'Install "spacy" requirements | pip install -U g4f[files]')

    nlp = spacy.load("en_core_web_sm")
    for page in source_iterator:
        doc = nlp(page)
        #for chunk in doc.noun_chunks:
        #    yield " ".join([token.lemma_ for token in chunk if not token.is_stop])
        # for token in doc:
        #     if not token.is_space:
        #         yield token.lemma_.lower()
        #         yield " "
        sentences = list(doc.sents)
        summary = sorted(sentences, key=lambda x: len(x.text), reverse=True)[:2]
        for sent in summary:
            yield sent.text

def get_filenames(bucket_dir: Path):
    files = bucket_dir / FILE_LIST
    if files.exists():
        with files.open('r') as f:
            return [filename.strip() for filename in f.readlines()]
    return []

def stream_read_files(bucket_dir: Path, filenames: list[str], delete_files: bool = False) -> Iterator[str]:
    for filename in filenames:
        if filename.startswith(DOWNLOADS_FILE):
            continue
        file_path: Path = bucket_dir / filename
        if not file_path.exists() or file_path.lstat().st_size <= 0:
            continue
        extension = os.path.splitext(filename)[1][1:]
        if filename.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(bucket_dir)
                try:
                    yield from stream_read_files(bucket_dir, [f for f in zip_ref.namelist() if supports_filename(f)], delete_files)
                except zipfile.BadZipFile:
                    pass
                finally:
                    if delete_files:
                        for unlink in zip_ref.namelist()[::-1]:
                            filepath = os.path.join(bucket_dir, unlink)
                            if os.path.exists(filepath):
                                if os.path.isdir(filepath):
                                    os.rmdir(filepath)
                                else:
                                    os.unlink(filepath)
            continue
        yield f"<!-- File: {filename} -->\n"
        if has_pypdf2 and filename.endswith(".pdf"):
            try:
                reader = PyPDF2.PdfReader(file_path)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    yield page.extract_text()
            except PdfReadError:
                continue
        if has_pdfplumber and filename.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    yield page.extract_text()
        if has_pdfminer and filename.endswith(".pdf"):
            yield extract_text(file_path)
        elif has_docx and filename.endswith(".docx"):
            doc = Document(file_path)
            for para in doc.paragraphs:
                yield para.text
        elif has_docx2txt and filename.endswith(".docx"):
            yield docx2txt.process(file_path)
        elif has_odfpy and filename.endswith(".odt"):
            textdoc = load(file_path)
            allparas = textdoc.getElementsByType(P)
            for p in allparas:
                yield p.firstChild.data if p.firstChild else ""
        elif has_ebooklib and filename.endswith(".epub"):
            book = epub.read_epub(file_path)
            for doc_item in book.get_items():
                if doc_item.get_type() == ebooklib.ITEM_DOCUMENT:
                    yield doc_item.get_content().decode(errors='ignore')
        elif has_openpyxl and filename.endswith(".xlsx"):
            df = pd.read_excel(file_path)
            for row in df.itertuples(index=False):
                yield " ".join(str(cell) for cell in row)
        elif has_beautifulsoup4 and filename.endswith(".html"):
            yield from scrape_text(file_path.read_text(errors="ignore"))
        elif extension in PLAIN_FILE_EXTENSIONS:
            yield file_path.read_text(errors="ignore").strip()
        yield f"\n<-- End -->\n\n"

def cache_stream(stream: Iterator[str], bucket_dir: Path) -> Iterator[str]:
    cache_file = bucket_dir / PLAIN_CACHE
    tmp_file = bucket_dir / f"{PLAIN_CACHE}.{time.time()}.tmp"
    if cache_file.exists():
        for chunk in read_path_chunked(cache_file):
            yield chunk
        return
    with open(tmp_file, "wb") as f:
        for chunk in stream:
            f.write(chunk.encode(errors="replace"))
            yield chunk
    tmp_file.rename(cache_file)

def is_complete(data: str):
    return data.endswith("\n```\n\n") and data.count("```") % 2 == 0

def read_path_chunked(path: Path):
    with path.open("r", encoding='utf-8') as f:
        current_chunk_size = 0
        buffer = ""
        for line in f:
            current_chunk_size += len(line.encode('utf-8'))
            buffer += line
            if current_chunk_size >= 4096:
                if is_complete(buffer) or current_chunk_size >= 8192:
                    yield buffer
                    buffer = ""
                    current_chunk_size = 0
        if current_chunk_size > 0:
            yield buffer

def read_bucket(bucket_dir: Path):
    bucket_dir = Path(bucket_dir)
    cache_file = bucket_dir / PLAIN_CACHE
    spacy_file = bucket_dir / f"spacy_0001.cache"
    if not spacy_file.is_file() and cache_file.is_file():
        yield cache_file.read_text(errors="replace")
    for idx in range(1, 1000):
        spacy_file = bucket_dir / f"spacy_{idx:04d}.cache"
        plain_file = bucket_dir / f"plain_{idx:04d}.cache"
        if spacy_file.is_file():
            yield spacy_file.read_text(errors="replace")
        elif plain_file.is_file():
            yield plain_file.read_text(errors="replace")
        else:
            break

def stream_read_parts_and_refine(bucket_dir: Path, delete_files: bool = False) -> Iterator[str]:
    cache_file = bucket_dir / PLAIN_CACHE
    space_file = Path(bucket_dir) / f"spacy_0001.cache"
    part_one = bucket_dir / f"plain_0001.cache"
    if not space_file.exists() and not part_one.exists() and cache_file.exists():
        split_file_by_size_and_newline(cache_file, bucket_dir)
    for idx in range(1, 1000):
        part = bucket_dir / f"plain_{idx:04d}.cache"
        tmp_file = Path(bucket_dir) / f"spacy_{idx:04d}.{time.time()}.tmp"
        cache_file = Path(bucket_dir) / f"spacy_{idx:04d}.cache"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                yield f.read(errors="replace")
            continue
        if not part.exists():
            break
        with tmp_file.open("w") as f:
            for chunk in spacy_refine_chunks(read_path_chunked(part)):
                f.write(chunk)
                yield chunk
        tmp_file.rename(cache_file)
        if delete_files:
            part.unlink()

def split_file_by_size_and_newline(input_filename, output_dir, chunk_size_bytes=1024*1024): # 1MB
    """Splits a file into chunks of approximately chunk_size_bytes, splitting only at newline characters.

    Args:
        input_filename: Path to the input file.
        output_prefix: Prefix for the output files (e.g., 'output_part_').
        chunk_size_bytes: Desired size of each chunk in bytes.
    """
    split_filename = os.path.splitext(os.path.basename(input_filename))
    output_prefix = os.path.join(output_dir, split_filename[0] + "_")

    with open(input_filename, 'r', encoding='utf-8') as infile:
        chunk_num = 1
        current_chunk = ""
        current_chunk_size = 0

        for line in infile:
            current_chunk += line
            current_chunk_size += len(line.encode('utf-8'))

            if current_chunk_size >= chunk_size_bytes:
                if is_complete(current_chunk) or current_chunk_size >= chunk_size_bytes * 2:
                    output_filename = f"{output_prefix}{chunk_num:04d}{split_filename[1]}"
                    with open(output_filename, 'w', encoding='utf-8') as outfile:
                        outfile.write(current_chunk)
                    current_chunk = ""
                    current_chunk_size = 0
                    chunk_num += 1

        # Write the last chunk
        if current_chunk:
            output_filename = f"{output_prefix}{chunk_num:04d}{split_filename[1]}"
            with open(output_filename, 'w', encoding='utf-8') as outfile:
                outfile.write(current_chunk)

def get_filename_from_url(url: str, extension: str = ".md") -> str:
    parsed_url = urllib.parse.urlparse(url)
    sha256_hash = hashlib.sha256(url.encode()).digest()
    base32_encoded = base64.b32encode(sha256_hash).decode()
    url_hash = base32_encoded[:24].lower()
    return f"{parsed_url.netloc}+{parsed_url.path[1:].replace('/', '_')}+{url_hash}{extension}"

async def get_filename(response: ClientResponse) -> str:
    """
    Attempts to extract a filename from an aiohttp response. Prioritizes Content-Disposition, then URL.

    Args:
        response: The aiohttp ClientResponse object.

    Returns:
        The filename as a string, or None if it cannot be determined.
    """

    content_disposition = response.headers.get('Content-Disposition')
    if content_disposition:
        try:
            filename = content_disposition.split('filename=')[1].strip('"')
            if filename:
                return secure_filename(filename)
        except IndexError:
            pass

    content_type = response.headers.get('Content-Type')
    url = str(response.url)
    if content_type and url:
        extension = await get_file_extension(response)
        if extension:
            return get_filename_from_url(url, extension)

    return None

async def get_file_extension(response: ClientResponse):
    """
    Attempts to determine the file extension from an aiohttp response.  Improved to handle more types.

    Args:
        response: The aiohttp ClientResponse object.

    Returns:
        The file extension (e.g., ".html", ".json", ".pdf", ".zip", ".md", ".txt") as a string,
        or None if it cannot be determined.
    """

    content_type = response.headers.get('Content-Type')
    if content_type:
        if "html" in content_type.lower():
            return ".html"
        elif "json" in content_type.lower():
            return ".json"
        elif "pdf" in content_type.lower():
            return ".pdf"
        elif "zip" in content_type.lower():
            return ".zip"
        elif "text/plain" in content_type.lower():
            return ".txt"
        elif "markdown" in content_type.lower():
            return ".md"

    url = str(response.url)
    if url:
        return Path(url).suffix.lower()

    return None

def read_links(html: str, base: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    for selector in [
            "main",
            ".main-content-wrapper",
            ".main-content",
            ".emt-container-inner",
            ".content-wrapper",
            "#content",
            "#mainContent",
        ]:
        select = soup.select_one(selector)
        if select:
            soup = select
            break
    urls = []
    for link in soup.select("a"):
        if "rel" not in link.attrs or "nofollow" not in link.attrs["rel"]:
            url = link.attrs.get("href")
            if url and url.startswith("https://") or url.startswith("/"):
                urls.append(url.split("#")[0])
    return set([urllib.parse.urljoin(base, link) for link in urls])

async def download_urls(
    bucket_dir: Path,
    urls: list[str],
    max_depth: int = 0,
    loading_urls: set[str] = set(),
    lock: asyncio.Lock = None,
    delay: int = 3,
    new_urls: list[str] = list(),
    group_size: int = 5,
    timeout: int = 10,
    proxy: Optional[str] = None
) -> AsyncIterator[str]:
    if lock is None:
        lock = asyncio.Lock()
    md = MarkItDown()
    async with ClientSession(
        connector=get_connector(proxy=proxy),
        timeout=ClientTimeout(timeout)
    ) as session:
        async def download_url(url: str, max_depth: int) -> str:
            text_content = None
            if has_markitdown:
                try:
                    text_content = md.convert(url).text_content
                    if text_content:
                        filename = get_filename_from_url(url)
                        target = bucket_dir / filename
                        text_content = f"{text_content.strip()}\n\nSource: {url}\n"
                        target.write_text(text_content, errors="replace")
                        return filename
                except Exception as e:
                    debug.log(f"Failed to convert URL to text: {type(e).__name__}: {e}")
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    filename = await get_filename(response)
                    if not filename:
                        debug.log(f"Failed to get filename for {url}")
                        return None
                    if not is_allowed_extension(filename) and not supports_filename(filename) or filename == DOWNLOADS_FILE:
                        return None
                    if filename.endswith(".html") and max_depth > 0:
                        add_urls = read_links(await response.text(), str(response.url))
                        if add_urls:
                            async with lock:
                                add_urls = [add_url for add_url in add_urls if add_url not in loading_urls]
                                [loading_urls.add(add_url) for add_url in add_urls]
                                [new_urls.append(add_url) for add_url in add_urls if add_url not in new_urls]
                    if is_allowed_extension(filename):
                        target = bucket_dir / "media" / filename
                        target.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        target = bucket_dir / filename
                    with target.open("wb") as f:
                        async for chunk in response.content.iter_any():
                            if filename.endswith(".html") and b'<link rel="canonical"' not in chunk:
                                f.write(chunk.replace(b'</head>', f'<link rel="canonical" href="{response.url}">\n</head>'.encode()))
                            else:
                                f.write(chunk)
                    return filename
            except (ClientError, asyncio.TimeoutError) as e:
                debug.log(f"Download failed: {e.__class__.__name__}: {e}")
            return None
        for filename in await asyncio.gather(*[download_url(url, max_depth) for url in urls]):
            if filename:
                yield filename
            else:
                await asyncio.sleep(delay)
        while new_urls:
            next_urls = list()
            for i in range(0, len(new_urls), group_size):
                chunked_urls = new_urls[i:i + group_size]
                async for filename in download_urls(bucket_dir, chunked_urls, max_depth - 1, loading_urls, lock, delay + 1, next_urls):
                    yield filename
                await asyncio.sleep(delay)
            new_urls = next_urls

def get_downloads_urls(bucket_dir: Path, delete_files: bool = False) -> Iterator[str]:
    download_file = bucket_dir / DOWNLOADS_FILE
    if download_file.exists():
        with download_file.open('r') as f:
            data = json.load(f) 
        if delete_files:
            download_file.unlink()
        if isinstance(data, list):
            for item in data:
                if "url" in item:
                    yield {"urls": [item.pop("url")], **item}
                elif "urls" in item:
                    yield item

def read_and_download_urls(bucket_dir: Path, delete_files: bool = False, event_stream: bool = False) -> Iterator[str]:
    urls = get_downloads_urls(bucket_dir, delete_files)
    if urls:
        count = 0
        with open(os.path.join(bucket_dir, FILE_LIST), 'a') as f:
            for url in urls:
                for filename in to_sync_generator(download_urls(bucket_dir, **url)):
                    f.write(f"{filename}\n")
                    if event_stream:
                        count += 1
                        yield f'data: {json.dumps({"action": "download", "count": count})}\n\n'

async def async_read_and_download_urls(bucket_dir: Path, delete_files: bool = False, event_stream: bool = False) -> AsyncIterator[str]:
    urls = get_downloads_urls(bucket_dir, delete_files)
    if urls:
        count = 0
        with open(os.path.join(bucket_dir, FILE_LIST), 'a') as f:
            for url in urls:
                async for filename in download_urls(bucket_dir, **url):
                    f.write(f"{filename}\n")
                    if event_stream:
                        count += 1
                        yield f'data: {json.dumps({"action": "download", "count": count})}\n\n'

def stream_chunks(bucket_dir: Path, delete_files: bool = False, refine_chunks_with_spacy: bool = False, event_stream: bool = False) -> Iterator[str]:
    size = 0
    if refine_chunks_with_spacy:
        for chunk in stream_read_parts_and_refine(bucket_dir, delete_files):
            if event_stream:
                size += len(chunk.encode())
                yield f'data: {json.dumps({"action": "refine", "size": size})}\n\n'
            else:
                yield chunk
    else:
        streaming = stream_read_files(bucket_dir, get_filenames(bucket_dir), delete_files)
        streaming = cache_stream(streaming, bucket_dir)
        for chunk in streaming:
            if event_stream:
                size += len(chunk.encode())
                yield f'data: {json.dumps({"action": "load", "size": size})}\n\n'
            else:
                yield chunk
        files_txt = os.path.join(bucket_dir, FILE_LIST)
        if os.path.exists(files_txt):
            for filename in get_filenames(bucket_dir):
                if is_allowed_extension(filename):
                    yield f'data: {json.dumps({"action": "media", "filename": filename})}\n\n'
                if delete_files and os.path.exists(os.path.join(bucket_dir, filename)):
                    os.remove(os.path.join(bucket_dir, filename))
            os.remove(files_txt)
            if event_stream:
                yield f'data: {json.dumps({"action": "delete_files"})}\n\n'
    if event_stream:
        yield f'data: {json.dumps({"action": "done", "size": size})}\n\n'

def get_streaming(bucket_dir: str, delete_files = False, refine_chunks_with_spacy = False, event_stream: bool = False) -> Iterator[str]:
    bucket_dir = Path(bucket_dir)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield from read_and_download_urls(bucket_dir, delete_files, event_stream)
        yield from stream_chunks(bucket_dir, delete_files, refine_chunks_with_spacy, event_stream)
    except Exception as e:
        if event_stream:
            yield f'data: {json.dumps({"error": {"message": str(e)}})}\n\n'
        raise e

async def get_async_streaming(bucket_dir: str, delete_files = False, refine_chunks_with_spacy = False, event_stream: bool = False) -> Iterator[str]:
    bucket_dir = Path(bucket_dir)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    try:
        async for chunk in async_read_and_download_urls(bucket_dir, delete_files, event_stream):
            yield chunk
        for chunk in stream_chunks(bucket_dir, delete_files, refine_chunks_with_spacy, event_stream):
            yield chunk
    except Exception as e:
        if event_stream:
            yield f'data: {json.dumps({"error": {"message": str(e)}})}\n\n'
        raise e

def get_tempfile(file, suffix: str = None):
    copyfile = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    shutil.copyfileobj(file, copyfile)
    copyfile.close()
    file.close()
    return copyfile.name