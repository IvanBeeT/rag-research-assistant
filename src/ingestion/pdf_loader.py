"""Extract text and metadata from research paper PDFs.

pdfplumber handles multi-column layouts and tables better than pypdf,
which matters for papers formatted in IEEE/ACM two-column style.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field

import pdfplumber


@dataclass
class Document:
    """A single loaded document with its text and metadata."""
    text: str
    metadata: dict = field(default_factory=dict)


def load_pdf(path: Path) -> Document:
    """Load a PDF and return its full text with metadata.

    Strips hyphenated line-breaks that appear when pdfplumber extracts
    columnar text, and collapses excessive whitespace.
    """
    path = Path(path)
    pages_text = []

    with pdfplumber.open(path) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                pages_text.append(text)

    raw = "\n".join(pages_text)
    cleaned = _clean_text(raw)

    metadata = {
        "source": path.name,
        "title": _infer_title(path.name),
        "num_pages": num_pages,
        "file_path": str(path.resolve()),
    }

    return Document(text=cleaned, metadata=metadata)


def load_papers_dir(papers_dir: Path) -> list[Document]:
    """Load all PDFs in a directory. Skips unreadable files with a warning."""
    papers_dir = Path(papers_dir)
    docs = []
    pdf_files = sorted(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {papers_dir}")
        return docs

    for pdf_path in pdf_files:
        try:
            doc = load_pdf(pdf_path)
            docs.append(doc)
            print(f"  Loaded: {pdf_path.name} ({doc.metadata['num_pages']} pages)")
        except Exception as e:
            print(f"  Warning: could not load {pdf_path.name}: {e}")

    return docs


def _clean_text(text: str) -> str:
    """Remove PDF extraction artifacts while preserving structure."""
    # Rejoin words broken across lines with a hyphen
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse runs of whitespace (but preserve paragraph breaks)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _infer_title(filename: str) -> str:
    """Convert a filename like 'attention_is_all_you_need.pdf' to a readable title."""
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").title()
