# chunker.py

"""Split context-document text into heading and paragraph chunks."""

from __future__ import annotations

import logging
import re

_LOG = logging.getLogger(__name__)

_HEADING = re.compile(r"^#{1,6}[ \t]+\S")
_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})")
_BLANK_SPLIT = re.compile(r"\n[ \t]*\n+")
_HTML_COMMENT = re.compile(r"\A<!--(?:(?!-->).)*-->\Z", re.DOTALL)


def chunk_context(document: dict[str, str]) -> list[dict[str, str]]:
    """Split `{source_file, text}` into `{source_file, chunk_id, text}` chunks.

    ATX headings (`#`–`######`) outside fenced code blocks start a
    section. Blank lines split paragraphs. Each paragraph keeps its
    section heading. Does not read files or log document text.
    """
    source_file = document["source_file"]
    chunks: list[dict[str, str]] = []
    for section in _sections(document["text"]):
        for paragraph in _paragraphs(section):
            if _HTML_COMMENT.match(paragraph):
                continue
            chunks.append(
                {
                    "source_file": source_file,
                    "chunk_id": f"{source_file}:{len(chunks)}",
                    "text": paragraph,
                }
            )
    _LOG.info("chunk context name=%s chunks=%s", source_file, len(chunks))
    return chunks


def _sections(text: str) -> list[str]:
    """Split on ATX headings outside ``` / ~~~ fences. Keep a preamble."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    starts = _heading_starts(normalized)
    if not starts:
        return [normalized] if normalized.strip() else []
    sections: list[str] = []
    first = starts[0]
    if first > 0:
        preamble = normalized[:first]
        if preamble.strip():
            sections.append(preamble)
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(normalized)
        sections.append(normalized[start:end])
    return sections


def _heading_starts(text: str) -> list[int]:
    """Offsets of ATX heading lines that are not inside a fenced code block."""
    starts: list[int] = []
    fence_char: str | None = None
    fence_len = 0
    offset = 0
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if fence_char is not None:
            if _is_fence_close(line, fence_char, fence_len):
                fence_char = None
                fence_len = 0
        else:
            opened = _fence_open(line)
            if opened is not None:
                fence_char, fence_len = opened
            elif _HEADING.match(line):
                starts.append(offset)
        offset += len(line)
        if index < len(lines) - 1:
            offset += 1
    return starts


def _fence_open(line: str) -> tuple[str, int] | None:
    """Return `(marker, length)` when `line` opens a ``` or ~~~ fence."""
    match = _FENCE.match(line)
    if match is None:
        return None
    marker = match.group(2)
    rest = line[match.end() :]
    if marker[0] == "`" and "`" in rest:
        return None
    return marker[0], len(marker)


def _is_fence_close(line: str, char: str, min_len: int) -> bool:
    """True when `line` closes a fence of `char` with at least `min_len` marks."""
    match = _FENCE.match(line)
    if match is None:
        return False
    marker = match.group(2)
    if marker[0] != char or len(marker) < min_len:
        return False
    return not line[match.end() :].strip()


def _paragraphs(section: str) -> list[str]:
    """Split a section on blank lines. Keep the heading on each paragraph."""
    stripped = section.strip()
    if not stripped:
        return []
    lines = stripped.split("\n")
    heading = ""
    body = stripped
    if _HEADING.match(lines[0]):
        heading = lines[0]
        body = "\n".join(lines[1:]).strip()
        if not body:
            return [heading]
    parts = [part.strip() for part in _BLANK_SPLIT.split(body) if part.strip()]
    if not parts:
        return [heading] if heading else []
    if heading:
        return [f"{heading}\n\n{part}" for part in parts]
    return parts
