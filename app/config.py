"""Central settings keep file locations and model choices easy to change."""
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
USER_DATA_DIR = DATA_DIR / "users"
INDEX_FILE = DATA_DIR / "shared_index.json"
STATIC_DIR = ROOT / "static"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
