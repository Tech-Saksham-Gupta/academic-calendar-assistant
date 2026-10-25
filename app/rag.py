"""A small, transparent RAG pipeline with persisted Gemini text embeddings."""
import json
import math
import re
from google import genai
from .config import GEMINI_API_KEY, EMBEDDING_MODEL, INDEX_FILE, TOP_K
from .documents import load_documents

class KnowledgeBase:
    """Index source passages, retrieve the closest passages, and cite them."""
    def __init__(self, user_upload_dir=None, index_file=INDEX_FILE):
        self.client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
        self.user_upload_dir = user_upload_dir
        self.index_file = index_file
        self.items = []
        self.load_saved_index()

    def load_saved_index(self):
        """Reuse a previously built index to avoid repeated embedding charges."""
        if self.index_file.exists():
            try:
                self.items = json.loads(self.index_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.items = []

    def sources_changed(self) -> bool:
        """Detect changed or newly added source passages before serving requests."""
        current = load_documents(self.user_upload_dir)
        saved = [(item.get("source"), item.get("page"), item.get("text")) for item in self.items]
        latest = [(item.get("source"), item.get("page"), item.get("text")) for item in current]
        return saved != latest

    def rebuild(self) -> int:
        """Read all documents and refresh the vector index when a key is set."""
        docs = load_documents(self.user_upload_dir)
        if not self.client:
            # Keep text available for lexical fallback while offline or unconfigured.
            self.items = [{**doc, "embedding": []} for doc in docs]
            return len(self.items)
        fresh = []
        for doc in docs:
            try:
                vector = self._embed(doc["text"])
            except Exception as error:
                # Keep documents usable with keyword search if Gemini embeddings are busy.
                print(f"Embedding unavailable for {doc['source']}; using keyword search: {error}")
                vector = []
            fresh.append({**doc, "embedding": vector})
        self.items = fresh
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(json.dumps(fresh), encoding="utf-8")
        return len(fresh)

    def _embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return list(result.embeddings[0].values)

    def retrieve(self, query: str, limit: int = TOP_K) -> list[dict]:
        """Return relevant chunks by cosine similarity, or term overlap without a key."""
        if not self.items:
            return []
        terms = set(re.findall(r"[\w'-]+", query.lower()))
        query_vector = None
        if self.client:
            try:
                query_vector = self._embed(query)
            except Exception as error:
                # A temporary embedding outage should not stop the assistant answering.
                print(f"Query embedding unavailable; using keyword search: {error}")
        ranked = []
        for item in self.items:
            vector = item.get("embedding", [])
            if query_vector and vector:
                score = self._cosine(query_vector, vector)
            else:
                item_terms = set(re.findall(r"[\w'-]+", item["text"].lower()))
                score = len(terms & item_terms) / max(len(terms), 1)
            ranked.append((score, item))
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
