"""FastAPI entry point with a private document space for each browser visitor."""
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
import re
import secrets

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import STATIC_DIR, USER_DATA_DIR, DATA_DIR, MAX_UPLOAD_MB, GEMINI_API_KEY
from .rag import KnowledgeBase
from .gemini import answer_question

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

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1200)

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

@app.get("/")
def home(request: Request, response: Response):
    """Serve the chat page and initialize a private browser session."""
    get_session_id(request, response)
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/health")
def health():
    """Simple service check used by Render; it never returns visitor information."""
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY), "indexed_passages": len(shared_knowledge.items), "index_error": index_error or None}

@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request, response: Response):
    """Retrieve from shared course notes and this visitor's private calendar only."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    try:
        session_id = get_session_id(request, response)
        sources = knowledge_for(session_id).retrieve(question)
        generated_answer = answer_question(question, sources)
        return {"answer": generated_answer, "sources": [
            {"name": item["source"], "page": item.get("page"), "excerpt": item["text"], "relevance": item["score"]}
            for item in sources
        ]}
    except Exception as error:
        print(f"Chat request failed: {error}")
        if "GEMINI_API_KEY is missing" in str(error):
            raise HTTPException(status_code=503, detail=str(error))
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {error}")

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
        return {"message": f"Added {safe_name} to your private browser workspace ({count} passages indexed)."}
    except Exception as error:
        target.unlink(missing_ok=True)
        print(f"Private document indexing failed: {error}")
        raise HTTPException(status_code=502, detail="Calendar indexing failed. Check the Gemini API key and try again.")
