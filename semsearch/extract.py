"""Text extraction for supported file types (.txt, .md, .pdf)."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text(path: Path) -> str:
    """Extract plain text from a supported file. Returns '' on failure."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            return _extract_pdf_text(path)
    except Exception:
        return ""
    return ""


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)


def iter_documents(root: Path):
    """Yield (path, text) for every supported file under root."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = extract_text(path)
            if text.strip():
                yield path, text
