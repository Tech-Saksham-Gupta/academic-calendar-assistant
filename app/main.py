"""FastAPI entry point: serves the user interface and simple JSON API."""
from contextlib import asynccontextmanager
from pathlib import Path
import secrets
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .config import STATIC_DIR, UPLOAD_DIR, MAX_UPLOAD_MB, GEMINI_API_KEY, ADMIN_UPLOAD_TOKEN
from .rag import KnowledgeBase
from .gemini import answer_question

knowledge = KnowledgeBase()
index_error = ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build an index once at startup, while retaining a saved index when possible."""
    global index_error
    if not knowledge.items or knowledge.sources_changed():
        try:
            knowledge.rebuild()
            index_error = ""
        except Exception as error:
            index_error = str(error)
            print(f"Initial indexing skipped: {error}")
    yield

app = FastAPI(title="Campus Compass", description="RAG academic calendar assistant", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1200)

@app.get("/")
def home():
    """Serve the single-page chat experience."""
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/health")
def health():
    """Simple health check used by hosting platforms."""
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY), "indexed_passages": len(knowledge.items), "index_error": index_error or None}

@app.post("/api/chat")
def chat(payload: ChatRequest):
    """Retrieve grounded source passages and generate the assistant response."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    try:
        sources = knowledge.retrieve(question)
        response = answer_question(question, sources)
        return {"answer": response, "sources": [
            {"name": item["source"], "page": item.get("page"), "excerpt": item["text"], "relevance": item["score"]}
            for item in sources
        ]}
    except Exception as error:
        print(f"Chat request failed: {error}")
        if "GEMINI_API_KEY is missing" in str(error):
            raise HTTPException(status_code=503, detail=str(error))
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {error}")

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), x_admin_token: str = Header(default="")):
    """Allow an owner to add a PDF, Markdown, or text source, then reindex it."""
    # Uploads can trigger paid embedding calls, so require an owner token in production.
    if not ADMIN_UPLOAD_TOKEN or not secrets.compare_digest(x_admin_token, ADMIN_UPLOAD_TOKEN):
        raise HTTPException(status_code=403, detail="Document uploads are restricted. Configure the owner upload token.")
    allowed = {".pdf", ".md", ".txt", ".csv"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Upload a PDF, CSV, Markdown, or text file.")
    content = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Files must be smaller than {MAX_UPLOAD_MB} MB.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Use a generated name so an uploaded filename cannot escape the data folder.
    safe_name = Path(file.filename).name.replace("..", "_")
    target = UPLOAD_DIR / safe_name
    target.write_bytes(content)
    try:
        count = knowledge.rebuild()
        return {"message": f"Added {safe_name}; {count} passages are indexed."}
    except Exception as error:
        target.unlink(missing_ok=True)
        print(f"Document indexing failed: {error}")
        raise HTTPException(status_code=502, detail="The document was saved but indexing failed. Check your Gemini key and try again.")
