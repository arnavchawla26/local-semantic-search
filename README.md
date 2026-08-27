# Local Semantic Search

Semantic search over your own notes and PDFs that runs entirely on your machine — no API key, no cloud embedding service, no network call at query time.

Point it at a folder of `.txt` / `.md` / `.pdf` files, build an index, then search it with natural-language queries. Results are ranked by meaning, not just exact keyword overlap — a query like "how do I train a model with gradients" correctly surfaces a note about backpropagation even though it doesn't share most of those exact words.

## How it works

- **Extraction** (`semsearch/extract.py`) pulls plain text out of `.txt`, `.md`, and `.pdf` files.
- **Chunking** (`semsearch/chunk.py`) splits long documents into overlapping ~800-character chunks so search results point at a specific passage, not just "somewhere in this 50-page PDF."
- **Indexing** (`semsearch/index.py`) fits a TF-IDF vectorizer over all chunks, then a truncated SVD (latent semantic analysis) to project them into a smaller "latent semantic" space, so chunks about the same topic land close together even when they use different words. This is classic, well-understood NLP — not a transformer embedding model — chosen specifically because it needs no GPU, no API key, and no internet access, while still doing real semantic matching (see the test that verifies a gradient-descent query ranks a backpropagation note above a bread-baking note).
- **Search** projects your query into the same space and ranks chunks by cosine similarity.

## Getting Started

```sh
git clone https://github.com/arnavchawla26/local-semantic-search.git
cd local-semantic-search
pip install -r requirements.txt

# Build an index over a folder of notes/PDFs
python cli.py index ./sample_docs --out .semsearch_index.pkl

# Search it
python cli.py search .semsearch_index.pkl "how do I train a model with gradients"
```

Try it against the bundled `sample_docs/` (three short notes on neural networks, databases, and bread baking) to see the ranking work before pointing it at your own files.

## CLI Reference

```
python cli.py index <directory> [--out PATH] [--components N] [--chunk-size N] [--overlap N]
python cli.py search <index-file> "<query>" [--top N]
```

## Running the tests

```sh
python -m pytest tests/ -v
```

6 tests cover chunking edge cases (empty input, size bounds), correct chunk indexing, index save/load round-tripping, the empty-directory error path, and — the important one — that a gradient/training query actually ranks the neural-networks note above an unrelated bread-baking note.

## Limitations / Roadmap

- TF-IDF+SVD works well for topically distinct documents but won't capture deep paraphrase/synonym relationships the way a transformer embedding model would; swapping in a local sentence-embedding model (e.g. via `sentence-transformers`, still no API key needed) is a natural upgrade if you want stronger semantic matching and don't mind a heavier dependency.
- No incremental indexing yet — re-running `index` rebuilds from scratch.
- Chunking is character-based, not sentence/paragraph-aware.
