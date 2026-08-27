"""Local semantic index: TF-IDF + truncated SVD (LSA), no external API calls.

This is a genuinely local "semantic" search: TF-IDF captures term
importance, and truncated SVD projects that into a lower-dimensional
"latent semantic" space so that documents sharing related vocabulary (not
just exact keyword overlap) end up with similar vectors. It's not a
transformer embedding model, but it requires no API key, no GPU, and no
network access, and it demonstrably groups semantically related text —
which is what "local-first" means here.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunk import Chunk, build_chunks
from .extract import iter_documents

INDEX_VERSION = 1


@dataclass
class SearchResult:
    doc_path: str
    chunk_index: int
    score: float
    snippet: str


class SemanticIndex:
    def __init__(self, n_components: int = 128, chunk_size: int = 800, overlap: int = 150):
        self.n_components = n_components
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.doc_vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def build(self, root: Path) -> int:
        """Index every supported document under `root`. Returns chunk count."""
        self.chunks = []
        for doc_path, text in iter_documents(root):
            rel = str(doc_path.relative_to(root)) if doc_path.is_relative_to(root) else str(doc_path)
            self.chunks.extend(build_chunks(rel, text, self.chunk_size, self.overlap))

        if not self.chunks:
            raise ValueError(f"No supported documents (.txt/.md/.pdf) with extractable text found under {root}")

        texts = [c.text for c in self.chunks]
        n_components = min(self.n_components, max(2, len(texts) - 1))

        self.vectorizer = TfidfVectorizer(
            max_df=0.95, min_df=1, stop_words="english", ngram_range=(1, 2)
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)

        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.vectorizer is None or self.svd is None or self.doc_vectors is None:
            raise RuntimeError("Index not built or loaded yet")

        query_tfidf = self.vectorizer.transform([query])
        query_vec = self.svd.transform(query_tfidf)

        sims = cosine_similarity(query_vec, self.doc_vectors)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for i in top_indices:
            chunk = self.chunks[i]
            snippet = chunk.text[:280] + ("..." if len(chunk.text) > 280 else "")
            results.append(
                SearchResult(
                    doc_path=chunk.doc_path,
                    chunk_index=chunk.chunk_index,
                    score=float(sims[i]),
                    snippet=snippet,
                )
            )
        return results

    def save(self, path: Path) -> None:
        payload = {
            "version": INDEX_VERSION,
            "n_components": self.n_components,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "vectorizer": self.vectorizer,
            "svd": self.svd,
            "doc_vectors": self.doc_vectors,
            "chunks": self.chunks,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "SemanticIndex":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("version") != INDEX_VERSION:
            raise ValueError(
                f"Index file version mismatch (got {payload.get('version')}, expected {INDEX_VERSION}) — rebuild it"
            )
        idx = cls(
            n_components=payload["n_components"],
            chunk_size=payload["chunk_size"],
            overlap=payload["overlap"],
        )
        idx.vectorizer = payload["vectorizer"]
        idx.svd = payload["svd"]
        idx.doc_vectors = payload["doc_vectors"]
        idx.chunks = payload["chunks"]
        return idx
