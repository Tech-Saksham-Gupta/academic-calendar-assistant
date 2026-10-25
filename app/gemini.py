"""Create concise, source-grounded answers with Google's Gemini API."""
from google import genai
from .config import GEMINI_API_KEY, GEMINI_MODEL

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
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    if not response.text:
        raise RuntimeError("Gemini returned an empty response. Check the model setting and API key access.")
    return response.text
