"""A small, transparent RAG pipeline with persisted Gemini text embeddings."""
import json
import math
from pathlib import Path
from google import genai
from .config import GEMINI_API_KEY, EMBEDDING_MODEL, INDEX_FILE, TOP_K
from .documents import load_documents

class KnowledgeBase:
    """Index source passages, retrieve the closest passages, and cite them."""
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
        self.items = []
        self.load_saved_index()

    def load_saved_index(self):
        """Reuse a previously built index to avoid repeated embedding charges."""
        if INDEX_FILE.exists():
            try:
                self.items = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.items = []

    def sources_changed(self) -> bool:
        """Detect changed or newly added source passages before serving requests."""
        current = load_documents()
        saved = [(item.get("source"), item.get("page"), item.get("text")) for item in self.items]
        latest = [(item.get("source"), item.get("page"), item.get("text")) for item in current]
        return saved != latest

    def rebuild(self) -> int:
        """Read all documents and refresh the vector index when a key is set."""
        docs = load_documents()
        if not self.client:
            # Keep text available for lexical fallback while offline or unconfigured.
            self.items = [{**doc, "embedding": []} for doc in docs]
            return len(self.items)
        fresh = []
        for doc in docs:
            vector = self._embed(doc["text"])
            fresh.append({**doc, "embedding": vector})
        self.items = fresh
        INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        INDEX_FILE.write_text(json.dumps(fresh), encoding="utf-8")
        return len(fresh)

    def _embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return list(result.embeddings[0].values)

    def retrieve(self, query: str, limit: int = TOP_K) -> list[dict]:
        """Return relevant chunks by cosine similarity, or term overlap without a key."""
        if not self.items:
            return []
        if self.client:
            query_vector = self._embed(query)
            ranked = [(self._cosine(query_vector, item.get("embedding", [])), item) for item in self.items]
        else:
            terms = set(query.lower().split())
            ranked = [(len(terms & set(item["text"].lower().split())) / max(len(terms), 1), item) for item in self.items]
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [{**item, "score": round(score, 3)} for score, item in ranked[:limit] if score > 0]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        """Measure the angle-based similarity between two embedding vectors."""
        if not left or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
        return dot / norm if norm else 0.0
