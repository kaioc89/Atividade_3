"""Tests for source URL fetch normalization."""

from __future__ import annotations

import sys
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


def test_fetch_source_url_contents_removes_nul_characters(monkeypatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        return FakeHttpResponse(
            body=b"<html><body><p>Texto\x00 da fonte.</p></body></html>",
            content_type="text/html; charset=utf-8",
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    report = fetch_source_url_contents(["https://fonte.example/lei"])

    assert not report.failures
    assert report.successes[0].content == "Texto da fonte."


def test_fetch_source_url_contents_decodes_utf16_html_with_bom(monkeypatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        html = "<html><body><p>Lei Maria da Penha.</p></body></html>"
        return FakeHttpResponse(
            body=html.encode("utf-16"),
            content_type="text/html",
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    report = fetch_source_url_contents(["https://fonte.example/lei"])

    assert not report.failures
    assert report.successes[0].content == "Lei Maria da Penha."


def test_fetch_source_url_contents_tolerates_truncated_utf16_html(monkeypatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        html = "<html><body><p>Lei Maria da Penha.</p></body></html>"
        return FakeHttpResponse(
            body=html.encode("utf-16") + b" ",
            content_type="text/html",
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    report = fetch_source_url_contents(["https://fonte.example/lei"])

    assert not report.failures
    assert report.successes[0].content.startswith("Lei Maria da Penha.")


def test_fetch_source_url_contents_extracts_pdf_text(monkeypatch) -> None:
    class FakePdfPage:
        def __init__(self, text: str | None) -> None:
            self._text = text

        def extract_text(self) -> str | None:
            return self._text

    class FakePdfReader:
        def __init__(self, stream: BytesIO) -> None:
            assert stream.read() == b"%PDF"
            self.pages = [FakePdfPage("Codigo de Etica"), FakePdfPage("Disciplina da OAB")]

    class FakePypdfModule:
        PdfReader = FakePdfReader

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        return FakeHttpResponse(body=b"%PDF", content_type="application/pdf")

    monkeypatch.setitem(sys.modules, "pypdf", FakePypdfModule())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    report = fetch_source_url_contents(["https://fonte.example/lei.pdf"])

    assert not report.failures
    assert report.successes[0].content == "Codigo de Etica Disciplina da OAB"


def test_fetch_source_url_contents_rejects_pdf_viewer_without_direct_pdf(monkeypatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        return FakeHttpResponse(
            body=b"""
            <html><body>
              Thumbnails Document Outline Attachments
              Presentation Mode Open Print Download
              Enter the password to open this PDF file
            </body></html>
            """,
            content_type="text/html; charset=utf-8",
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    report = fetch_source_url_contents(["https://www.oab.org.br/visualizador/19/codigo-de-etica-e-disciplina"])

    assert not report.successes
    assert "visualizador PDF detectado" in report.failures[0].reason


def test_fetch_source_url_contents_resolves_pdf_viewer_with_direct_pdf(monkeypatch) -> None:
    class FakePdfPage:
        def extract_text(self) -> str:
            return "Codigo de Etica recuperado do PDF"

    class FakePdfReader:
        def __init__(self, stream: BytesIO) -> None:
            assert stream.read() == b"%PDF"
            self.pages = [FakePdfPage()]

    class FakePypdfModule:
        PdfReader = FakePdfReader

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        if request.full_url == "https://www.oab.org.br/visualizador/19/codigo-de-etica-e-disciplina":
            return FakeHttpResponse(
                body=b"""
                <html><body>
                  Thumbnails Document Outline Attachments
                  Presentation Mode Open Print Download
                  Enter the password to open this PDF file
                  <iframe src="/arquivos/codigo-de-etica.pdf"></iframe>
                </body></html>
                """,
                content_type="text/html; charset=utf-8",
            )
        assert request.full_url == "https://www.oab.org.br/arquivos/codigo-de-etica.pdf"
        return FakeHttpResponse(body=b"%PDF", content_type="application/pdf")

    monkeypatch.setitem(sys.modules, "pypdf", FakePypdfModule())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    report = fetch_source_url_contents(["https://www.oab.org.br/visualizador/19/codigo-de-etica-e-disciplina"])

    assert not report.failures
    assert report.successes[0].url == "https://www.oab.org.br/visualizador/19/codigo-de-etica-e-disciplina"
    assert report.successes[0].content_type == "application/pdf"
    assert report.successes[0].content == "Codigo de Etica recuperado do PDF"


def test_fetch_source_url_contents_reports_unsupported_content_type(monkeypatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        return FakeHttpResponse(body=b"ZIP", content_type="application/zip")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    report = fetch_source_url_contents(["https://fonte.example/arquivo.zip"])

    assert not report.successes
    assert report.failures[0].url == "https://fonte.example/arquivo.zip"
    assert "Tipo de conteudo nao suportado" in report.failures[0].reason
