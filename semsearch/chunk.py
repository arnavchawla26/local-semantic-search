"""Split document text into overlapping chunks for indexing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    doc_path: str
    chunk_index: int
    text: str


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into character-based chunks with overlap.

    Character-based (not token-based) to avoid a tokenizer dependency; word
    boundaries are respected reasonably well by splitting on whitespace runs.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(current):
                overlap_len += len(w) + 1
                overlap_words.insert(0, w)
                if overlap_len >= overlap:
                    break
            current = overlap_words
            current_len = sum(len(w) + 1 for w in current)

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_chunks(doc_path: str, text: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    return [
        Chunk(doc_path=doc_path, chunk_index=i, text=t)
        for i, t in enumerate(chunk_text(text, chunk_size, overlap))
    ]
