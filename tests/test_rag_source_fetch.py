"""Tests for source URL fetch normalization."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import urllib.request

from atividade_2.rag_source_fetch import fetch_source_url_contents, split_source_urls


class FakeHttpResponse:
    def __init__(self, *, body: bytes, content_type: str, status: int = 200) -> None:
        self._body = BytesIO(body)
        self.headers = {"content-type": content_type}
        self.status = status

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


def test_split_source_urls_extracts_multiple_urls_from_one_field() -> None:
    urls = split_source_urls(
        [
            "https://www.planalto.gov.br/ccivil_03/leis/l7210.htm, "
            "http://www.planalto.gov.br/ccivil_03/decreto-lei/del2848compilado.htm",
            "https://www.planalto.gov.br/ccivil_03/leis/l7210.htm",
        ]
    )

    assert urls == [
        "https://www.planalto.gov.br/ccivil_03/leis/l7210.htm",
        "http://www.planalto.gov.br/ccivil_03/decreto-lei/del2848compilado.htm",
    ]


def test_fetch_source_url_contents_extracts_readable_html(monkeypatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        assert request.full_url == "https://fonte.example/lei"
        assert timeout == 20
        return FakeHttpResponse(
            body=b"""
            <html>
              <head><title>Lei teste</title><style>.x{}</style></head>
              <body><script>ignore()</script><h1>Lei 14.133</h1><p>Texto da fonte.</p></body>
            </html>
            """,
            content_type="text/html; charset=utf-8",
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    report = fetch_source_url_contents(["https://fonte.example/lei"])

    assert not report.failures
    assert report.successes[0].title == "Lei teste"
    assert "Lei 14.133 Texto da fonte." in report.successes[0].content
    assert "ignore" not in report.successes[0].content


def test_fetch_source_url_contents_reports_unsupported_content_type(monkeypatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        return FakeHttpResponse(body=b"%PDF", content_type="application/pdf")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    report = fetch_source_url_contents(["https://fonte.example/lei.pdf"])

    assert not report.successes
    assert report.failures[0].url == "https://fonte.example/lei.pdf"
    assert "Tipo de conteudo nao suportado" in report.failures[0].reason
