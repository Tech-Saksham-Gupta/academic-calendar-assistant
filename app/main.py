"""FastAPI entry point with a private document space for each browser visitor."""
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
import json
import re
import secrets
import string
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import STATIC_DIR, USER_DATA_DIR, DATA_DIR, MAX_UPLOAD_MB, GEMINI_API_KEY
from .rag import KnowledgeBase
from .gemini import answer_question, stream_answer
from .documents import extract_calendar_events

# Shared syllabus is indexed once. Each visitor with an upload gets a separate index.
shared_knowledge = KnowledgeBase(index_file=DATA_DIR / "shared_index.json")
private_knowledge: dict[str, KnowledgeBase] = {}
index_error = ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the shared course notes once when the server starts."""
    global index_error
    if not shared_knowledge.items or shared_knowledge.sources_changed():
        try:
            shared_knowledge.rebuild()
            index_error = ""
        except Exception as error:
            index_error = str(error)
            print(f"Initial indexing skipped: {error}")
    yield

app = FastAPI(title="Campus Compass", description="Private, RAG academic calendar assistant", version="1.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class ChatTurn(BaseModel):
    """One recent user or assistant message, used to keep follow-up questions coherent."""
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1200)

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1200)
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)

def get_session_id(request: Request, response: Response) -> str:
    """Create an unguessable browser session cookie when a visitor first arrives."""
    token = request.cookies.get("campus_session", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{40,64}", token):
        token = secrets.token_urlsafe(32)
        # Render terminates HTTPS at its proxy and forwards this header to the app.
        is_https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(
            "campus_session", token, max_age=60 * 60 * 24 * 180,
            httponly=True, secure=is_https, samesite="lax", path="/",
        )
    return token

def user_directories(session_id: str) -> tuple[Path, Path]:
    """Map a private cookie to safe per-user storage paths without exposing it."""
    folder_name = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    user_root = USER_DATA_DIR / folder_name
    return user_root, user_root / "uploads"

def knowledge_for(session_id: str) -> KnowledgeBase:
    """Use shared syllabus alone, or combine it with this visitor's own uploads."""
    user_root, upload_dir = user_directories(session_id)
    user_files = [path for path in upload_dir.iterdir() if path.is_file()] if upload_dir.exists() else []
    if not user_files:
        return shared_knowledge

    cache_key = str(user_root)
    if cache_key not in private_knowledge:
        private_knowledge[cache_key] = KnowledgeBase(
            user_upload_dir=upload_dir,
            index_file=user_root / "index.json",
        )
    kb = private_knowledge[cache_key]
    if not kb.items or kb.sources_changed():
        kb.rebuild()
    return kb

def is_study_plan_question(question: str) -> bool:
    """Include the user's calendar schedule for planning and what-if questions."""
    text = question.casefold()
    planning_phrases = (
        "study plan", "study schedule", "plan my study", "plan my preparation",
        "make a plan", "create a plan", "prepare for", "preparation plan",
        "what if", "what-if", "padhai ka plan", "study karne ka plan",
    )
    return any(phrase in text for phrase in planning_phrases)

def study_plan_has_exam_details(question: str, history: list[dict] | None = None) -> bool:
    """Require a usable subject and date before retrieving citations for a plan."""
    text = question.casefold()
    month = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    has_date = bool(re.search(
        rf"\b(?:{month})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+20\d{{2}})?\b"
        rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month})\b"
        rf"|\b20\d{{2}}-\d{{2}}-\d{{2}}\b",
        text,
    ))
    subject_match = re.search(
        r"\b(?:my|a|an)\s+([a-z][a-z0-9&+ /-]{0,45}?)\s+(?:exam|test)\b",
        text,
    )
    subject = subject_match.group(1).strip() if subject_match else ""
    # Short course names are often sent as a separate reply, for example "DSA".
    acronyms = set(re.findall(r"\b[A-Z][A-Z0-9]{1,5}\b", question))
    month_abbreviations = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "SEPT", "OCT", "NOV", "DEC"}
    has_subject = bool(
        (subject and subject not in {"final", "upcoming", "semester", "the"})
        or (acronyms - month_abbreviations)
    )

    # If the assistant previously asked for a subject, treat the student's next
    # descriptive reply as that subject (for example, "Data Structures").
    if not has_subject and history:
        ignored_words = {
            "i", "a", "an", "the", "my", "me", "is", "am", "are", "have", "for", "on",
            "and", "or", "please", "subject", "exam", "test", "date", "hours", "hour", "hrs",
            "daily", "study", "time", "minutes", "mins", "min", "per", "day",
            "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
            "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
            "oct", "october", "nov", "november", "dec", "december",
        }
        for index, turn in enumerate(history):
            if turn["role"] != "assistant" or "subject" not in turn["content"].casefold():
                continue
            next_answer = next((item["content"] for item in history[index + 1:] if item["role"] == "user"), "")
            # Remove dates and time amounts; any remaining meaningful words likely
            # name the subject requested by the assistant.
            without_date = re.sub(
                rf"\b(?:{month})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+20\d{{2}})?\b"
                rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month})\b"
                rf"|\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|h|minutes?|mins?)\b",
                " ", next_answer.casefold(),
            )
            remaining = [word for word in re.findall(r"[a-z]+", without_date) if word not in ignored_words]
            if remaining:
                has_subject = True
                break
    return has_date and has_subject

def answer_context(question: str, session_id: str, history: list[dict] | None = None) -> tuple[list[dict], list[dict] | None]:
    """Retrieve evidence, avoiding irrelevant citations for an incomplete study-plan request."""
    history = history or []
    # Use the recent conversation to resolve short follow-ups like "DSA, 16 Dec".
    previous_user_messages = [turn["content"] for turn in history if turn["role"] == "user"]
    conversation_query = " ".join([*previous_user_messages, question])
    planning = is_study_plan_question(conversation_query)
    if planning and not study_plan_has_exam_details(conversation_query, history):
        # The assistant should ask for the missing subject/date first, without
        # attaching unrelated retrieved pages to a clarification message.
        return [], None

    sources = knowledge_for(session_id).retrieve(conversation_query)
    calendar_events = None
    if planning:
        # The shared AI syllabus is not evidence for a calendar-grounded plan unless
        # the student explicitly says the exam is for that course.
        mentions_ai_course = bool(re.search(r"\b(?:ai|artificial intelligence)\b", conversation_query, re.IGNORECASE))
        if not mentions_ai_course:
            sources = [source for source in sources if source["source"].casefold() != "ai_syllabus.md"]
        _, upload_dir = user_directories(session_id)
        calendar_events = extract_calendar_events(upload_dir)
    return sources, calendar_events

def visible_citations(sources: list[dict]) -> list[dict]:
    """Hide the bundled starter syllabus from citation chips in every reply."""
    return [
        source for source in sources
        if Path(source.get("source", "")).name.casefold() != "ai_syllabus.md"
    ]

def quick_greeting(question: str) -> str | None:
    """Reply instantly to a greeting without calling search or the Gemini API."""
    # Only treat a message as a greeting when the entire message is a greeting.
    # For example, "Hi, when is my exam?" still goes through RAG and Gemini.
    cleaned = question.casefold().strip().translate(str.maketrans("", "", string.punctuation))
    greetings = {
        "hi", "hello", "hey", "hi there", "hello there", "hey there",
        "good morning", "good afternoon", "good evening",
    }
    if cleaned in greetings:
        return "Hi! I’m Campus Compass. Ask me about dates and deadlines in your uploaded academic calendar, or tell me about an exam to make a study plan."
    return None

@app.get("/")
def home(request: Request, response: Response):
    """Serve the chat page and initialize a private browser session."""
    get_session_id(request, response)
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/health")
def health():
    """Simple service check used by Render; it never returns visitor information."""
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY), "indexed_passages": len(shared_knowledge.items), "index_error": index_error or None}

@app.get("/api/calendar")
def calendar_preview(request: Request, response: Response):
    """Return only this browser's extracted calendar events for the sidebar preview."""
    session_id = get_session_id(request, response)
    _, upload_dir = user_directories(session_id)
    events = extract_calendar_events(upload_dir)
    dated = [event for event in events if event.get("start_date")]
    return {"events": dated, "count": len(dated)}

@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request, response: Response):
    """Retrieve from shared course notes and this visitor's private calendar only."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    greeting = quick_greeting(question)
    if greeting:
        # Greetings do not need retrieval or AI generation, so return them immediately.
        return {"answer": greeting, "sources": []}
    try:
        session_id = get_session_id(request, response)
        history = [turn.model_dump() for turn in payload.history]
        sources, calendar_events = answer_context(question, session_id, history)
        generated_answer = answer_question(question, sources, calendar_events=calendar_events, history=history)
        return {"answer": generated_answer, "sources": [
            {"name": item["source"], "page": item.get("page"), "excerpt": item["text"], "relevance": item["score"]}
            for item in visible_citations(sources)
        ]}
    except Exception as error:
        print(f"Chat request failed: {error}")
        if "GEMINI_API_KEY is missing" in str(error):
            raise HTTPException(status_code=503, detail=str(error))
        raise HTTPException(status_code=502, detail="The assistant could not complete this request. Please try again in a moment.")

@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest, request: Request, response: Response):
    """Stream status and Gemini answer tokens to the browser as soon as available."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    greeting = quick_greeting(question)
    session_id = None if greeting else get_session_id(request, response)
    history = [turn.model_dump() for turn in payload.history]

    def event_stream():
        # Newline-delimited JSON lets the browser process each update immediately.
        def send(event: dict) -> str:
            return json.dumps(event, ensure_ascii=False) + "\n"

        if greeting:
            yield send({"type": "token", "text": greeting})
            yield send({"type": "done"})
            return

        request_started = perf_counter()
        try:
            yield send({"type": "status", "message": "Searching your uploaded calendar…"})
            retrieval_started = perf_counter()
            sources, calendar_events = answer_context(question, session_id, history)
            retrieval_ms = round((perf_counter() - retrieval_started) * 1000)
            yield send({"type": "sources", "sources": [
                {"name": item["source"], "page": item.get("page"), "excerpt": item["text"], "relevance": item["score"]}
                for item in visible_citations(sources)
            ]})

            yield send({"type": "status", "message": "Preparing your answer…"})
            generation_started = perf_counter()
            for update in stream_answer(question, sources, calendar_events=calendar_events, history=history):
                yield send(update)
            generation_ms = round((perf_counter() - generation_started) * 1000)
            total_ms = round((perf_counter() - request_started) * 1000)
            # Log phase durations only; never log the student's question or answer.
            print(f"Chat timing: retrieval={retrieval_ms}ms, Gemini={generation_ms}ms, total={total_ms}ms")
            yield send({"type": "done"})
        except Exception as error:
            print(f"Streaming chat request failed: {error}")
            yield send({"type": "error", "message": "I couldn't complete that request. Please try again in a moment."})

    stream_response = StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
    # Preserve a newly-created private workspace cookie on the streaming response.
    for name, value in response.raw_headers:
        if name.lower() == b"set-cookie":
            stream_response.raw_headers.append((name, value))
    return stream_response

@app.post("/api/documents")
async def upload_document(request: Request, response: Response, file: UploadFile = File(...)):
    """Save and index a calendar only inside the current visitor's private folder."""
    allowed = {".pdf", ".md", ".txt", ".csv"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Upload a PDF, CSV, Markdown, or text file.")
    content = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Files must be smaller than {MAX_UPLOAD_MB} MB.")

    session_id = get_session_id(request, response)
    user_root, upload_dir = user_directories(session_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Keep the original file name for source citations, but discard any path supplied by the browser.
    safe_name = Path(file.filename or "calendar").name.replace("..", "_")
    target = upload_dir / safe_name
    target.write_bytes(content)
    try:
        cache_key = str(user_root)
        kb = private_knowledge.get(cache_key)
        if kb is None:
            kb = KnowledgeBase(user_upload_dir=upload_dir, index_file=user_root / "index.json")
            private_knowledge[cache_key] = kb
        count = kb.rebuild()
        event_count = len(extract_calendar_events(upload_dir))
        if event_count:
            message = f"Your calendar uploaded successfully! I found {event_count} dated events for your Coming up panel."
        else:
            message = "Your file uploaded successfully, but I couldn't detect readable calendar dates for the Coming up panel. You can still ask questions about its contents."
        return {"message": message, "indexed_passages": count, "calendar_events": event_count}
    except Exception as error:
        target.unlink(missing_ok=True)
        print(f"Private document indexing failed: {error}")
        raise HTTPException(status_code=502, detail="Calendar indexing failed. Check the Gemini API key and try again.")
