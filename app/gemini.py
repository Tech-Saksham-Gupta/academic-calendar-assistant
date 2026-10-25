"""Create concise, source-grounded answers with Google's Gemini API."""
import time
from google import genai
from google.genai import types
from .config import GEMINI_API_KEY, GEMINI_MODEL

BACKUP_MODELS = ("gemini-3.6-flash", "gemini-3.8-flash")

def _thinking_level(model: str) -> str:
    """Use the lightest supported reasoning level for short calendar answers."""
    return "minimal" if model in {"gemini-3.5-flash-lite", "gemini-3.6-flash"} else "low"

def _is_temporary_error(error: Exception) -> bool:
    """Recognize temporary overload, quota, or server errors from Gemini."""
    message = str(error).lower()
    return any(marker in message for marker in (
        "503", "unavailable", "429", "resource_exhausted", "500", "502", "504",
        "internal error", "high demand", "try again later",
    ))

def _source_fallback(sources: list[dict]) -> str:
    """Show useful retrieved facts if all Gemini models are temporarily unavailable."""
    if not sources:
        return ("Gemini is temporarily busy, and I couldn't find a matching passage in your uploaded "
                "calendar or course notes. Please try again shortly, or check the source document directly.")
    excerpts = "\n\n".join(f"From {item['source']}: {item['text']}" for item in sources[:3])
    return ("Gemini is temporarily busy, so here are the closest passages I found in your sources. "
            "Please verify the wording in the cited document.\n\n" + excerpts)

def answer_question(question: str, sources: list[dict]) -> str:
    """Ask Gemini to answer only from retrieved context and admit uncertainty."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in Render → your service → Environment, then redeploy.")
    # Still call Gemini when retrieval finds nothing. The model can give a natural
    # explanation that the knowledge base is missing the answer instead of returning
    # a canned message that makes the whole app look like a static FAQ.
    context = "\n\n".join(f"[{item['source']}]: {item['text']}" for item in sources)
    if not context:
        context = "No relevant passages were found in the uploaded calendar or course documents."
    prompt = f"""You are Campus Compass, a helpful academic calendar and course assistant.
Use the source excerpts below for institution-specific dates, deadlines, policies, and course facts. Never invent those details.
If no relevant excerpt is available, tell the student the uploaded sources do not contain that answer and suggest uploading the official calendar or contacting the academic office. For general learning questions, you may explain the concept, but label it as a general explanation rather than an institutional fact.
Be concise and student-friendly. Cite claims with the source name in square brackets.

SOURCE EXCERPTS:
{context}

STUDENT QUESTION: {question}
ANSWER:"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    # Try the configured model first, then stable backup models for service overload.
    models = list(dict.fromkeys((GEMINI_MODEL, *BACKUP_MODELS)))
    last_error = None
    for model_index, model in enumerate(models):
        # One short retry for the primary model; move to backups promptly on overload.
        attempts = 2 if model_index == 0 else 1
        for attempt in range(attempts):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level=_thinking_level(model))
                    ),
                )
                if response.text:
                    return response.text
                raise RuntimeError(f"Gemini model {model} returned an empty response.")
            except Exception as error:
                last_error = error
                # Short exponential waits give a busy primary endpoint a chance to recover.
                if _is_temporary_error(error) and attempt < attempts - 1:
                    time.sleep(0.4)
                    continue
                # A model/key configuration error is unlikely to improve by retrying.
                if not _is_temporary_error(error):
                    raise RuntimeError(f"Gemini could not answer with model {model}: {error}") from error
                break
    print(f"All Gemini answer models are temporarily unavailable: {last_error}")
    return _source_fallback(sources)
