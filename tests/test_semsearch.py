import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from semsearch.chunk import chunk_text, build_chunks
from semsearch.index import SemanticIndex


def test_chunk_text_respects_size_roughly():
    text = " ".join(["word"] * 500)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks[:-1]:
        assert len(c) >= 50


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_build_chunks_indexes_correctly():
    chunks = build_chunks("doc.md", "a " * 300, chunk_size=100, overlap=10)
    assert all(c.doc_path == "doc.md" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_index_build_and_search_finds_semantically_related_doc(tmp_path):
    (tmp_path / "ml.md").write_text(
        "Backpropagation computes gradients for training neural networks "
        "using the chain rule across layers, then gradient descent updates weights.",
        encoding="utf-8",
    )
    (tmp_path / "baking.md").write_text(
        "Autolyse rests flour and water before adding yeast, improving "
        "gluten development and dough extensibility for bread baking.",
        encoding="utf-8",
    )

    idx = SemanticIndex(n_components=2, chunk_size=1000)
    n_chunks = idx.build(tmp_path)
    assert n_chunks == 2

    results = idx.search("how does training a neural network work with gradients", top_k=2)
    assert len(results) == 2
    assert results[0].doc_path == "ml.md"


def test_index_raises_on_empty_directory(tmp_path):
    idx = SemanticIndex()
    with pytest.raises(ValueError):
        idx.build(tmp_path)


def test_index_save_and_load_roundtrip(tmp_path):
    (tmp_path / "a.md").write_text("apples and oranges are fruit", encoding="utf-8")
    (tmp_path / "b.md").write_text("cars and trucks are vehicles", encoding="utf-8")

    idx = SemanticIndex(n_components=2, chunk_size=1000)
    idx.build(tmp_path)

    index_path = tmp_path / "index.pkl"
    idx.save(index_path)

    loaded = SemanticIndex.load(index_path)
    results = loaded.search("fruit", top_k=1)
    assert results[0].doc_path == "a.md"
