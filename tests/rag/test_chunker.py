# test_chunker.py

"""Tests for context-document chunking. No dataset rows; text only."""

from __future__ import annotations

from pathlib import Path

from src.rag.chunker import chunk_context
from src.rag.readers import read_context_file

_MAX = 50 * 1024 * 1024


def test_chunks_glossary_fixture(sample_glossary_md: Path):
    """The glossary fixture is one heading section with both definitions."""
    chunks = chunk_context(read_context_file(sample_glossary_md, max_bytes=_MAX))
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["source_file"] == "sample_glossary.md"
    assert chunk["chunk_id"] == "sample_glossary.md:0"
    assert chunk["text"].startswith("# Sales glossary")
    assert "Revenue is net of returns." in chunk["text"]
    assert "Region is the sales territory." in chunk["text"]
    assert "date,region,revenue" not in chunk["text"]


def test_splits_on_headings():
    """Each ATX heading starts a new chunk."""
    chunks = chunk_context(
        {
            "source_file": "glossary.md",
            "text": (
                "# Revenue\n\nRevenue is net of returns.\n\n"
                "# Region\n\nRegion is the sales territory.\n"
            ),
        }
    )
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "glossary.md:0",
        "glossary.md:1",
    ]
    assert chunks[0]["source_file"] == "glossary.md"
    assert chunks[0]["text"].startswith("# Revenue")
    assert "Revenue is net of returns." in chunks[0]["text"]
    assert "Region is the sales territory." not in chunks[0]["text"]
    assert chunks[1]["text"].startswith("# Region")
    assert "Region is the sales territory." in chunks[1]["text"]


def test_splits_on_blank_line_paragraphs():
    """Plain text with no headings splits on blank lines."""
    chunks = chunk_context(
        {
            "source_file": "notes.txt",
            "text": "Revenue is net of returns.\n\nRegion is the sales territory.\n",
        }
    )
    assert [chunk["text"] for chunk in chunks] == [
        "Revenue is net of returns.",
        "Region is the sales territory.",
    ]
    assert chunks[0]["chunk_id"] == "notes.txt:0"
    assert chunks[1]["chunk_id"] == "notes.txt:1"


def test_drops_html_comment_only_paragraphs():
    """A file-banner comment is not a retrieval chunk."""
    chunks = chunk_context(
        {
            "source_file": "notes.md",
            "text": "<!-- notes.md -->\n\nRevenue is net of returns.\n",
        }
    )
    assert [chunk["text"] for chunk in chunks] == ["Revenue is net of returns."]
    assert chunks[0]["chunk_id"] == "notes.md:0"


def test_does_not_split_on_hash_inside_fenced_code():
    """A # comment inside ``` is not an ATX heading."""
    chunks = chunk_context(
        {
            "source_file": "notes.md",
            "text": (
                "# Revenue\n\n"
                "Revenue is net of returns.\n\n"
                "```python\n"
                "# revenue = price * qty\n"
                "```\n\n"
                "# Region\n\n"
                "Region is the sales territory.\n"
            ),
        }
    )
    assert [chunk["text"].split("\n", 1)[0] for chunk in chunks] == [
        "# Revenue",
        "# Revenue",
        "# Region",
    ]
    assert any("# revenue = price * qty" in chunk["text"] for chunk in chunks)


def test_keeps_section_heading_on_each_paragraph():
    """A heading is repeated on every paragraph in that section."""
    chunks = chunk_context(
        {
            "source_file": "glossary.md",
            "text": (
                "# Sales glossary\n\n"
                "Revenue is net of returns.\n\n"
                "Region is the sales territory.\n"
            ),
        }
    )
    assert [chunk["text"] for chunk in chunks] == [
        "# Sales glossary\n\nRevenue is net of returns.",
        "# Sales glossary\n\nRegion is the sales territory.",
    ]
