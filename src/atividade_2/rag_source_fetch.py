"""Fetch and normalize source URL text for RAG vector enrichment."""

from __future__ import annotations

import re
import http.client
import urllib.error
import urllib.request
from html import unescape
from io import BytesIO
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qsl, unquote, urljoin, urlparse, urlsplit


@dataclass(frozen=True)
class SourceUrlContent:
    url: str
    content: str
    content_type: str | None
    title: str | None = None


@dataclass(frozen=True)
class SourceUrlFailure:
    url: str
    reason: str


@dataclass(frozen=True)
class SourceUrlFetchReport:
    successes: list[SourceUrlContent]
    failures: list[SourceUrlFailure]


def fetch_source_url_contents(
    urls: list[str],
    *,
    timeout_seconds: int = 20,
    max_bytes: int = 5_000_000,
) -> SourceUrlFetchReport:
    """Fetch text from curated source URLs, returning explicit failures."""
    successes: list[SourceUrlContent] = []
    failures: list[SourceUrlFailure] = []
    for url in split_source_urls(urls):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            failures.append(SourceUrlFailure(url=url, reason="URL deve usar http ou https."))
            continue
        try:
            successes.append(_fetch_one(url=url, timeout_seconds=timeout_seconds, max_bytes=max_bytes))
        except SourceFetchError as error:
            failures.append(SourceUrlFailure(url=url, reason=str(error)))
    return SourceUrlFetchReport(successes=successes, failures=failures)


def split_source_urls(values: list[str]) -> list[str]:
    """Extract unique http(s) URLs from curation source fields."""
    seen: set[str] = set()
    urls: list[str] = []
    for value in values:
        for match in re.finditer(r"https?://[^\s,;]+", value.strip()):
            url = match.group(0).rstrip(").]")
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


class SourceFetchError(RuntimeError):
    """Raised for one source URL fetch failure."""


def _fetch_one(*, url: str, timeout_seconds: int, max_bytes: int) -> SourceUrlContent:
    raw, content_type = _fetch_raw(url=url, timeout_seconds=timeout_seconds, max_bytes=max_bytes)
    return _content_from_raw(url=url, raw=raw, content_type=content_type, timeout_seconds=timeout_seconds, max_bytes=max_bytes)


def _fetch_raw(*, url: str, timeout_seconds: int, max_bytes: int) -> tuple[bytes, str | None]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "atividade-2-rag-source-fetch/0.1"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise SourceFetchError(f"HTTP {status}.")
            content_type = response.headers.get("content-type")
            raw = response.read(max_bytes + 1)
    except urllib.error.HTTPError as error:
        raise SourceFetchError(f"HTTP {error.code}.") from error
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", error)
        raise SourceFetchError(f"Falha de rede: {reason}.") from error
    except http.client.InvalidURL as error:
        raise SourceFetchError(f"URL invalida: {error}.") from error
    except TimeoutError as error:
        raise SourceFetchError("Tempo limite ao consultar a URL.") from error

    if len(raw) > max_bytes:
        raise SourceFetchError(f"Conteudo excede o limite de {max_bytes} bytes.")
    return raw, content_type


def _content_from_raw(
    *,
    url: str,
    raw: bytes,
    content_type: str | None,
    timeout_seconds: int,
    max_bytes: int,
) -> SourceUrlContent:
    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_type not in {"", "text/html", "application/xhtml+xml", "text/plain", "application/pdf"}:
        raise SourceFetchError(f"Tipo de conteudo nao suportado: {normalized_type or 'desconhecido'}.")

    if normalized_type == "application/pdf":
        content = _normalize_whitespace(_extract_pdf_text(raw))
        title = None
    elif normalized_type == "text/plain":
        text = _decode_response(raw, content_type=content_type)
        content = _normalize_whitespace(text)
        title = None
    else:
        text = _decode_response(raw, content_type=content_type)
        parser = _ReadableHtmlParser()
        parser.feed(text)
        content = _normalize_whitespace(" ".join(parser.parts))
        title = _normalize_whitespace(parser.title or "") or None
        if _looks_like_pdf_viewer_text(content):
            pdf_url = _extract_pdf_url_from_html(text, base_url=url)
            if pdf_url is None:
                raise SourceFetchError("Conteudo de visualizador PDF detectado sem URL direta confiavel do documento.")
            pdf_raw, pdf_content_type = _fetch_raw(url=pdf_url, timeout_seconds=timeout_seconds, max_bytes=max_bytes)
            pdf_content_type = _normalized_resolved_pdf_content_type(pdf_url, pdf_content_type)
            pdf_content = _normalize_whitespace(_extract_pdf_text(pdf_raw))
            if not pdf_content:
                raise SourceFetchError("Conteudo textual vazio.")
            return SourceUrlContent(url=url, content=pdf_content, content_type=pdf_content_type, title=title)
        if _looks_like_spaced_markup(content):
            raise SourceFetchError("Conteudo HTML aparenta markup bruto ou decodificacao incorreta.")
    if not content:
        raise SourceFetchError("Conteudo textual vazio.")
    return SourceUrlContent(url=url, content=content, content_type=content_type, title=title)


def _decode_response(raw: bytes, *, content_type: str | None) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    charset_match = re.search(r"charset=([^;\s]+)", content_type or "", flags=re.IGNORECASE)
    encodings = [charset_match.group(1)] if charset_match else []
    encodings.extend(["utf-8", "iso-8859-1"])
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise SourceFetchError("Leitor de PDF indisponivel: instale a dependencia pypdf.") from error

    try:
        reader = PdfReader(BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise SourceFetchError(f"Falha ao extrair texto PDF: {error}.") from error


def _looks_like_pdf_viewer_text(content: str) -> bool:
    normalized = content.lower()
    signatures = [
        "thumbnails document outline attachments",
        "presentation mode open print download",
        "enter the password to open this pdf file",
        "pdf version: page count:",
    ]
    return sum(1 for signature in signatures if signature in normalized) >= 2


def _extract_pdf_url_from_html(text: str, *, base_url: str) -> str | None:
    decoded = unescape(text)
    candidates: list[str] = []
    candidates.extend(
        match.group(1)
        for match in re.finditer(
            r"""(?:href|src|data-url|data-file|file)\s*=\s*["']([^"']+?\.pdf(?:\?[^"']*)?)["']""",
            decoded,
            flags=re.IGNORECASE,
        )
    )
    candidates.extend(
        match.group(0)
        for match in re.finditer(r"""https?://[^\s"'<>]+?\.pdf(?:\?[^\s"'<>]*)?""", decoded, flags=re.IGNORECASE)
    )
    for _key, value in parse_qsl(urlsplit(base_url).query, keep_blank_values=False):
        if ".pdf" in value.lower():
            candidates.append(value)
    for match in re.finditer(r"""[?&]file=([^"'&<>]+?\.pdf(?:[^"'&<>]*)?)""", decoded, flags=re.IGNORECASE):
        candidates.append(unquote(match.group(1)))

    for candidate in candidates:
        pdf_url = urljoin(base_url, candidate.strip())
        if urlsplit(pdf_url).scheme in {"http", "https"} and ".pdf" in urlsplit(pdf_url).path.lower():
            return pdf_url
    return None


def _normalized_resolved_pdf_content_type(url: str, content_type: str | None) -> str | None:
    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_type in {"", "application/octet-stream"} and urlsplit(url).path.lower().endswith(".pdf"):
        return "application/pdf"
    if normalized_type != "application/pdf":
        raise SourceFetchError(f"URL de visualizador apontou para conteudo nao PDF: {normalized_type or 'desconhecido'}.")
    return content_type


def _looks_like_spaced_markup(content: str) -> bool:
    normalized = content.lower()
    if "ÿþ" in normalized:
        return True
    spaced_tags = ("< h t m l", "< / p >", "c l a s s = \" m s o n o r m a l", "t e x t - a l i g n")
    return sum(1 for signature in spaced_tags if signature in normalized) >= 2


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\x00", " ")).strip()


class _ReadableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title: str = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text or self._skip_depth:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        else:
            self.parts.append(text)
