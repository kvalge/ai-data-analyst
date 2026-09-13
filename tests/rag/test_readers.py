# test_readers.py

"""Tests for context-file readers. No dataset rows; text only."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.readers import ContextReadError, read_context_file
from src.validation.uploads import FileValidationError

_MAX = 50 * 1024 * 1024


def _pdf_with_text(text: str) -> bytes:
    """Minimal one-page PDF with extractable ASCII text. Test helper only."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 24 Tf 50 700 Td ({escaped}) Tj ET".encode("ascii")

    def obj(number: int, body: bytes) -> bytes:
        return f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    pieces = [
        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        obj(2, b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>"),
        obj(
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        ),
        obj(4, b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"),
        obj(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    header = b"%PDF-1.4\n"
    offsets: list[int] = []
    pos = len(header)
    body = b""
    for piece in pieces:
        offsets.append(pos)
        body += piece
        pos += len(piece)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n".encode()
    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(len(header) + len(body)).encode()
        + b"\n%%EOF\n"
    )
    return header + body + xref + trailer


def test_reads_glossary_markdown_fixture(sample_glossary_md: Path):
    """The committed glossary fixture loads as text, not as a table."""
    result = read_context_file(sample_glossary_md, max_bytes=_MAX)
    assert result["source_file"] == "sample_glossary.md"
    assert "Revenue is net of returns." in result["text"]
    assert "date,region,revenue" not in result["text"]


def test_reads_plain_text(tmp_path: Path):
    """A .txt context file returns its UTF-8 text."""
    path = tmp_path / "notes.txt"
    path.write_text("Region is the sales territory.\n", encoding="utf-8")
    result = read_context_file(path, max_bytes=_MAX)
    assert result["source_file"] == "notes.txt"
    assert result["text"] == "Region is the sales territory.\n"


def test_extracts_text_from_uploaded_pdf(tmp_path: Path):
    """An already-written PDF yields extractable text, not the raw bytes."""
    path = tmp_path / "glossary.pdf"
    path.write_bytes(_pdf_with_text("Revenue is net of returns."))
    result = read_context_file(path, max_bytes=_MAX)
    assert result["source_file"] == "glossary.pdf"
    assert "Revenue is net of returns." in result["text"]
    assert not result["text"].startswith("%PDF")


def test_pdf_without_text_raises(tmp_path: Path):
    """A PDF with no extractable text fails closed."""
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    with pytest.raises(ContextReadError, match="no extractable text|Could not read"):
        read_context_file(path, max_bytes=_MAX)


def test_invalid_utf8_plain_text_raises(tmp_path: Path):
    """A .txt with invalid UTF-8 maps to ContextReadError."""
    path = tmp_path / "notes.txt"
    path.write_bytes(b"Region is the sales territory.\n\xff")
    with pytest.raises(ContextReadError, match="Could not read"):
        read_context_file(path, max_bytes=_MAX)


def test_rejects_csv_as_context(tmp_path: Path, sample_sales_csv: Path):
    """A data file is not read as domain context."""
    path = tmp_path / "sales.csv"
    path.write_bytes(sample_sales_csv.read_bytes())
    with pytest.raises(FileValidationError, match="suffix"):
        read_context_file(path, max_bytes=_MAX)
