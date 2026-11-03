"""Read source documents and turn them into labeled text for retrieval."""
import csv
import re
from datetime import date, datetime
from pathlib import Path
from pypdf import PdfReader
from .config import DATA_DIR

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
MONTH_PATTERN = "|".join([*MONTHS.split("|"), *(name[:3] for name in MONTHS.split("|"))])
MONTH_NAMES = {name.casefold(): number for number, name in enumerate(MONTHS.split("|"), start=1)}
MONTH_NAMES.update({name[:3].casefold(): number for number, name in enumerate(MONTHS.split("|"), start=1)})
DATE_ROW = re.compile(
    rf"^\s*(?P<month>{MONTHS})\s+(?P<day>\d{{1,2}})"
    rf"(?:\s*[-–—]\s*(?P<end_day>\d{{1,2}}))?,?\s*(?P<year>\d{{4}})?\s*$",
    re.IGNORECASE,
)
# Academic calendars often use numeric dates (for example, 30.10.2026) or put
# the event name and date range together on one line. Recognize both forms.
NUMERIC_DATE = re.compile(r"(?<!\d)(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-](?P<year>20\d{2})(?!\d)")
ISO_DATE = re.compile(r"(?<!\d)(?P<year>20\d{2})[./-](?P<month>\d{1,2})[./-](?P<day>\d{1,2})(?!\d)")
MONTH_DATE = re.compile(
    rf"(?<![A-Za-z])(?:(?P<month_first>{MONTH_PATTERN})(?:\s+(?P<day_first>\d{{1,2}}))|"
    rf"(?P<day_second>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month_second>{MONTH_PATTERN}))"
    rf"(?:,?\s*(?P<year>20\d{{2}}))?",
    re.IGNORECASE,
)
MONTH_DAY_RANGE_END = re.compile(r"\s*[-–—]\s*(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(?P<year>20\d{2}))?", re.IGNORECASE)

def load_documents(user_upload_dir: Path | None = None) -> list[dict]:
    """Load shared course notes plus files from only the current browser user's folder."""
    documents = []
    calendar = DATA_DIR / "academic_calendar.csv"
    if calendar.exists():
        with calendar.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                text = " | ".join(f"{key}: {value}" for key, value in row.items() if value)
                if text:
                    documents.append({"text": text, "source": "Academic calendar", "page": None})

    # The course syllabus is shared; uploaded files are scoped to one user folder.
    folders = [DATA_DIR]
    if user_upload_dir is not None:
        folders.append(user_upload_dir)
    for folder in folders:
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
                    pages = list(enumerate(_read_pdf_pages(path), start=1))
                else:
                    continue
                for page_num, text in pages:
                    for section in _split_into_chunks(text):
                        documents.append({"text": section, "source": path.name, "page": page_num})
            except Exception as error:
                # One malformed file should not prevent the rest of the app from starting.
                print(f"Could not read {path.name}: {error}")
    return documents

def extract_calendar_events(user_upload_dir: Path | None = None) -> list[dict]:
    """Extract dated event rows for the dashboard from this user's uploaded files."""
    events = []
    if not user_upload_dir or not user_upload_dir.exists():
        return events

    for path in user_upload_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        date_text = next((value for key, value in row.items() if key and key.lower() in {"date", "start_date", "start date"} and value), "")
                        event_name = next((value for key, value in row.items() if key and key.lower() in {"event", "academic event", "title", "name"} and value), "Academic event")
                        details = next((value for key, value in row.items() if key and key.lower() in {"details", "description"} and value), "")
                        parsed = _parse_date_range(date_text)
                        if parsed:
                            start, end = parsed
                            events.append(_event_record(start, end, event_name, details, path.name))
            elif path.suffix.lower() == ".pdf":
                for page_number, text in enumerate(_read_pdf_pages(path), start=1):
                    events.extend(_events_from_lines(text.splitlines(), path.name, page_number))
            elif path.suffix.lower() in {".md", ".txt"}:
                events.extend(_events_from_lines(path.read_text(encoding="utf-8").splitlines(), path.name, None))
        except Exception as error:
            # Keep the chat usable even when a calendar file cannot be parsed for the preview.
            print(f"Could not extract calendar events from {path.name}: {error}")

    # De-duplicate repeated mentions (for example, a date in both a summary and a table).
    unique = {}
    for event in events:
        key = (event["start_date"], event["end_date"], event["title"].casefold())
        unique[key] = event
    return sorted(unique.values(), key=lambda event: (event["start_date"], event["title"]))

def _parse_date_range(value: str) -> tuple[date, date] | None:
    """Parse common calendar date formats, including ranges and numeric dates."""
    text = " ".join((value or "").split())
    match = DATE_ROW.fullmatch(text)
    if match and match.group("year"):
        try:
            month = datetime.strptime(match.group("month")[:3].title(), "%b").month
            start = date(int(match.group("year")), month, int(match.group("day")))
            end = date(int(match.group("year")), month, int(match.group("end_day"))) if match.group("end_day") else start
            return start, end
        except ValueError:
            return None
    # Also accept a pair such as 30.10.2026 - 03.11.2026. Day-first numeric
    # dates are common in institutional calendars outside the United States.
    date_matches = _date_matches(text)
    if date_matches:
        return date_matches[0][0], date_matches[-1][0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            parsed = datetime.strptime(text, fmt).date()
            return parsed, parsed
        except ValueError:
            continue
    return None


def _read_pdf_pages(path: Path) -> list[str]:
    """Read PDF pages in both text layouts, then OCR pages that still lack events."""
    pages = []
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages):
        candidates = []
        try:
            # Layout mode often keeps table rows together; plain mode can work
            # better when the PDF stores event names and dates in separate columns.
            candidates.append(page.extract_text(extraction_mode="layout") or "")
        except (TypeError, ValueError):
            pass
        try:
            candidates.append(page.extract_text() or "")
        except Exception as error:
            print(f"PDF text extraction failed for {path.name}, page {page_number + 1}: {error}")
        text = max(
            candidates,
            key=lambda candidate: (
                len(_events_from_lines(candidate.splitlines(), path.name, page_number + 1)),
                len(re.sub(r"\W", "", candidate)),
            ),
            default="",
        )
        # A PDF page can contain a text header but an image-only event table.
        # OCR it when extraction found no events on a calendar-like page.
        has_calendar_clues = bool(re.search(
            r"calendar|semester|academic year|term|examination|exam|deadline|classes|holiday|recess",
            text, re.IGNORECASE,
        ))
        has_events = bool(_events_from_lines(text.splitlines(), path.name, page_number + 1))
        sparse_text = len(re.sub(r"\W", "", text)) < 40
        if sparse_text or (has_calendar_clues and not has_events):
            ocr_text = _ocr_pdf_page(path, page_number)
            if ocr_text and len(_events_from_lines(ocr_text.splitlines(), path.name, page_number + 1)) > len(_events_from_lines(text.splitlines(), path.name, page_number + 1)):
                text = ocr_text
        pages.append(text)
    return pages


def _ocr_pdf_page(path: Path, page_number: int) -> str:
    """OCR one scanned page when Tesseract is installed (it is included in Docker)."""
    document = None
    page = None
    try:
        import pypdfium2 as pdfium
        import pytesseract

        document = pdfium.PdfDocument(str(path))
        page = document[page_number]
        image = page.render(scale=2).to_pil()
        return pytesseract.image_to_string(image, lang="eng", config="--psm 6", timeout=15)
    except Exception as error:
        # Keep text-based PDFs usable even if local OCR is not installed or a page fails.
        print(f"OCR unavailable for {path.name}, page {page_number + 1}: {error}")
        return ""
    finally:
        if page is not None:
            page.close()
        if document is not None:
            document.close()

def _events_from_lines(lines: list[str], source: str, page: int | None) -> list[dict]:
    """Extract dated events whether dates and event names share a line or not."""
    found = []
    clean_lines = [line.strip() for line in lines]
    index = 0
    term = "Academic calendar"
    while index < len(clean_lines):
        line = clean_lines[index]
        upper = line.upper()
        if "AUTUMN SEMESTER" in upper:
            term = "Autumn semester"
        elif "SPRING SEMESTER" in upper:
            term = "Spring semester"
        elif "SUMMER SESSION" in upper:
            term = "Summer session"

        date_text = line
        match = DATE_ROW.fullmatch(date_text)
        if match and not match.group("year") and index + 1 < len(clean_lines) and re.fullmatch(r"20\d{2}", clean_lines[index + 1]):
            date_text = f"{date_text} {clean_lines[index + 1]}"
            index += 1
        parsed = _parse_date_range(date_text)
        if parsed:
            start, end = parsed
            # Keep any event label that appears beside the date. If the PDF
            # places dates in their own table cell/line, use the next line.
            title = _remove_date_text(date_text).strip(" \t:|,;–—-.")
            title = re.sub(
                r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
                " ", title, flags=re.IGNORECASE,
            )
            title = re.sub(r"\s+", " ", title).strip(" \t:|,;–—-.")
            title_index = None
            if not title:
                title_index = index + 1
                while title_index < len(clean_lines) and not clean_lines[title_index]:
                    title_index += 1
                title = clean_lines[title_index] if title_index < len(clean_lines) else "Academic event"
            # Join short lowercase continuation lines such as "Course withdrawal / deadline".
            if title_index is not None and title_index + 1 < len(clean_lines) and clean_lines[title_index + 1] and clean_lines[title_index + 1][0].islower() and len(clean_lines[title_index + 1]) < 28:
                title += " " + clean_lines[title_index + 1]
            if title.upper() not in {"DATE", "ACADEMIC EVENT", "DETAILS"} and not title.upper().startswith("NORTHBRIDGE UNIVERSITY"):
                found.append(_event_record(start, end, title, "", source, page, term))
        index += 1
    return found


def _date_matches(text: str) -> list[tuple[date, re.Match]]:
    """Find date tokens in a line and return them in document order."""
    matches = []
    # In ranges such as "October 30 – November 3, 2026", the year is often
    # written only once at the end. Apply that year to the other date token.
    year_positions = [(int(match.group()), match.start()) for match in re.finditer(r"20\d{2}", text)]
    for pattern, day_first in ((NUMERIC_DATE, True), (ISO_DATE, False), (MONTH_DATE, None)):
        for match in pattern.finditer(text):
            try:
                if day_first is not None:
                    day = int(match.group("day"))
                    month = int(match.group("month"))
                    year = int(match.group("year"))
                else:
                    month_name = (match.group("month_first") or match.group("month_second")).casefold()
                    month = MONTH_NAMES[month_name]
                    day = int(match.group("day_first") or match.group("day_second"))
                    year_text = match.group("year")
                    if year_text:
                        year = int(year_text)
                    elif year_positions:
                        year = min(year_positions, key=lambda item: abs(item[1] - match.start()))[0]
                    else:
                        continue
                matches.append((date(year, month, day), match))
                # Same-month ranges are often shortened to "November 05-06, 2026".
                if day_first is None:
                    range_end = MONTH_DAY_RANGE_END.match(text, match.end())
                    if range_end:
                        end_year = int(range_end.group("year") or year)
                        matches.append((date(end_year, month, int(range_end.group("day"))), range_end))
            except (ValueError, KeyError):
                continue
    # Different patterns can recognize the same token; de-duplicate by span.
    unique = {match.span(): (parsed, match) for parsed, match in matches}
    return sorted(unique.values(), key=lambda item: item[1].start())


def _remove_date_text(text: str) -> str:
    """Remove recognized dates and range separators, leaving an event label."""
    spans = [match.span() for _, match in _date_matches(text)]
    for start, end in reversed(spans):
        text = text[:start] + " " + text[end:]
    text = re.sub(r"\b(?:to|through|until|till)\b|\s+[-–—]\s+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text)

def _event_record(start: date, end: date, title: str, details: str, source: str, page: int | None = None, term: str = "") -> dict:
    """Keep the API shape consistent for PDF and CSV calendar rows."""
    return {
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "title": title.strip(), "details": details.strip(),
        "source": source, "page": page, "term": term,
    }

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
