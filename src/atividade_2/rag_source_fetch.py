"""Fetch and normalize source URL text for RAG vector enrichment."""

from __future__ import annotations

import re
import http.client
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse


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

    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_type not in {"", "text/html", "application/xhtml+xml", "text/plain"}:
        raise SourceFetchError(f"Tipo de conteudo nao suportado: {normalized_type or 'desconhecido'}.")

    text = _decode_response(raw, content_type=content_type)
    if normalized_type == "text/plain":
        content = _normalize_whitespace(text)
        title = None
    else:
        parser = _ReadableHtmlParser()
        parser.feed(text)
        content = _normalize_whitespace(" ".join(parser.parts))
        title = _normalize_whitespace(parser.title or "") or None
    if not content:
        raise SourceFetchError("Conteudo textual vazio.")
    return SourceUrlContent(url=url, content=content, content_type=content_type, title=title)


def _decode_response(raw: bytes, *, content_type: str | None) -> str:
    charset_match = re.search(r"charset=([^;\s]+)", content_type or "", flags=re.IGNORECASE)
    encodings = [charset_match.group(1)] if charset_match else []
    encodings.extend(["utf-8", "iso-8859-1"])
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


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
