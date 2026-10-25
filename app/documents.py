"""Read source documents and turn them into labeled text for retrieval."""
import csv
from pathlib import Path
from pypdf import PdfReader
from .config import DATA_DIR, UPLOAD_DIR

def load_documents() -> list[dict]:
    """Load calendar rows and supported files from the knowledge folders."""
    documents = []
    calendar = DATA_DIR / "academic_calendar.csv"
    if calendar.exists():
        with calendar.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                text = " | ".join(f"{key}: {value}" for key, value in row.items() if value)
                if text:
                    documents.append({"text": text, "source": "Academic calendar", "page": None})

    # Include both built-in course notes and documents added by the user.
    for folder in (DATA_DIR, UPLOAD_DIR):
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if (folder == DATA_DIR and path.name == "academic_calendar.csv") or not path.is_file():
                continue
            try:
                if path.suffix.lower() in {".md", ".txt"}:
                    raw = path.read_text(encoding="utf-8")
                    pages = [(None, raw)]
                elif path.suffix.lower() == ".csv":
                    # A calendar CSV is converted row-by-row so each event is easy to retrieve.
                    with path.open(encoding="utf-8-sig", newline="") as handle:
                        rows = csv.DictReader(handle)
                        pages = [(None, "\n".join(" | ".join(f"{key}: {value}" for key, value in row.items() if value) for row in rows))]
                elif path.suffix.lower() == ".pdf":
                    pages = [(index + 1, page.extract_text() or "") for index, page in enumerate(PdfReader(path).pages)]
                else:
                    continue
                for page_num, text in pages:
                    for section in _split_into_chunks(text):
                        documents.append({"text": section, "source": path.name, "page": page_num})
            except Exception as error:
                # One malformed file should not prevent the rest of the app from starting.
                print(f"Could not read {path.name}: {error}")
    return documents

def _split_into_chunks(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    """Split long documents into overlapping passages without losing context."""
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks, start = [], 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        if end < len(clean):
            boundary = clean.rfind(". ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunks.append(clean[start:end].strip())
        if end == len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks
