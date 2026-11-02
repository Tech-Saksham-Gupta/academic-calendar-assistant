"""Create concise, source-grounded answers with Google's Gemini API."""
from datetime import datetime
from zoneinfo import ZoneInfo
from collections.abc import Iterator
from google import genai
from google.genai import types
from .config import GEMINI_API_KEY, GEMINI_MODEL

BACKUP_MODELS = ("gemini-3.6-flash", "gemini-3.8-flash")

def _generation_config(model: str, question: str, history: list[dict] | None = None) -> types.GenerateContentConfig:
    """Keep everyday answers short while allowing enough room for a study plan."""
    conversation_text = " ".join([question, *(turn["content"] for turn in (history or [])[-8:])]).casefold()
    planning_request = any(term in conversation_text for term in (
        "study plan", "study schedule", "make a plan", "prepare for", "preparation plan", "what-if", "what if",
    ))
    # Calendar answers need only a few sentences. Plans get a larger limit so they
    # are not cut off halfway through the schedule.
    output_limit = 1800 if planning_request else 600
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=_thinking_level(model)),
        max_output_tokens=output_limit,
    )

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
    excerpts = "\n\n".join(item["text"] for item in sources[:3])
    return ("Gemini is temporarily busy, so here are the closest passages I found in your sources. "
            "Please verify the wording in the cited document.\n\n" + excerpts)

def _build_prompt(
    question: str,
    sources: list[dict],
    calendar_events: list[dict] | None = None,
    history: list[dict] | None = None,
) -> str:
    """Build the same grounded prompt for both normal and streaming answers."""
    context = "\n\n".join(f"[{item['source']}]: {item['text']}" for item in sources)
    if not context:
        context = "No relevant passages were found in the uploaded calendar or course documents."
    event_context = "No dated events were extracted from the user's uploaded calendar."
    if calendar_events:
        event_lines = []
        for event in calendar_events:
            date_range = event["start_date"]
            if event.get("end_date") and event["end_date"] != event["start_date"]:
                date_range += f" to {event['end_date']}"
            term = f"; {event['term']}" if event.get("term") else ""
            event_lines.append(f"- {date_range}: {event['title']}{term}")
        event_context = "\n".join(event_lines)
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    history = history or []
    recent_conversation = "\n".join(
        f"{turn['role'].upper()}: {turn['content'][-1200:]}"
        for turn in history[-8:]
    ) or "No earlier messages in this chat."
    prompt = f"""You are Campus Compass, a helpful academic calendar and course assistant.
Project information: Campus Compass was made by Saksham Gupta, a second-year CSE student at LPU. If asked who made, created, or developed this website or project, answer clearly: "This website was made by Saksham Gupta, a second-year CSE student at LPU." Do not invent additional biographical details.
Use the recent conversation to understand short follow-up replies and remember details the student has already provided. Do not ask again for information already present in the conversation.
Use the source excerpts below for institution-specific dates, deadlines, policies, and course facts. Never invent those details.
If no relevant excerpt is available, tell the student the uploaded sources do not contain that answer and suggest uploading the official calendar or contacting the academic office. For general learning questions, you may explain the concept, but label it as a general explanation rather than an institutional fact.
For study-plan and what-if requests, first check the whole conversation for the subject and exam date. If either is missing, ask only for the missing required detail(s), optionally ask about daily study time and difficult topics, and do not produce a plan yet. In this clarification, do not list unrelated syllabus details or calendar events. Once subject and date are known, make the plan immediately; daily hours and difficult topics are optional, so use the student's answer if provided or assume about 2 hours per day and say so. Never repeat questions for details already given. Build a practical plan from the current date ({today}) to the exam date. Treat the student's exam details as user-provided, not as calendar facts. Use uploaded calendar events below to ground academic dates and flag nearby deadlines, breaks, classes, or exam periods; do not claim an event conflicts with personal commitments unless the student said so. If the calendar shows only a general final-exam period, state that the exact subject exam date came from the student and only say whether it falls inside that period. For a date without a year, use the next future matching date only when the calendar makes the year clear; otherwise ask briefly. If the student says preparation is zero, start with fundamentals, then guided practice, timed problem solving, and revision. If no syllabus is provided, label the topic sequence as a general suggestion and invite the student to share it. Make the plan specific by weeks or dates, include practice and buffer/revision time, and keep the workload realistic. Mention upcoming calendar dates that may affect the plan; don't invent institutional events. For normal date questions, answer directly and concisely.
Be concise and student-friendly. Do not add inline filename citations; the interface displays source files and page numbers separately below your answer.

RECENT CONVERSATION:
{recent_conversation}

SOURCE EXCERPTS:
{context}

UPLOADED CALENDAR EVENT LIST (use for planning around academic dates):
{event_context}

LATEST STUDENT MESSAGE: {question}
ANSWER:"""
    return prompt

def answer_question(
    question: str,
    sources: list[dict],
    calendar_events: list[dict] | None = None,
    history: list[dict] | None = None,
) -> str:
    """Ask Gemini to answer only from retrieved context and admit uncertainty."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in Render → your service → Environment, then redeploy.")
    prompt = _build_prompt(question, sources, calendar_events, history)
    client = genai.Client(api_key=GEMINI_API_KEY)
    # Try the configured model first, then stable backup models for service overload.
    models = list(dict.fromkeys((GEMINI_MODEL, *BACKUP_MODELS)))
    last_error = None
    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=_generation_config(model, question, history),
            )
            if response.text:
                return response.text
            raise RuntimeError(f"Gemini model {model} returned an empty response.")
        except Exception as error:
            last_error = error
            # If this model is busy, try the next one without repeating the request.
            if not _is_temporary_error(error):
                raise RuntimeError(f"Gemini could not answer with model {model}: {error}") from error
    print(f"All Gemini answer models are temporarily unavailable: {last_error}")
    return _source_fallback(sources)

def stream_answer(
    question: str,
    sources: list[dict],
    calendar_events: list[dict] | None = None,
    history: list[dict] | None = None,
) -> Iterator[dict]:
    """Yield answer tokens as Gemini creates them to reduce perceived waiting time."""
    if not GEMINI_API_KEY:
        yield {"type": "error", "message": "Gemini API key is not configured. Add GEMINI_API_KEY in Render's Environment settings."}
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_prompt(question, sources, calendar_events, history)
    models = list(dict.fromkeys((GEMINI_MODEL, *BACKUP_MODELS)))
    last_error = None

    for model in models:
        received_text = False
        try:
            response_stream = client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=_generation_config(model, question, history),
            )
            for chunk in response_stream:
                if chunk.text:
                    received_text = True
                    yield {"type": "token", "text": chunk.text}
            if received_text:
                return
            raise RuntimeError(f"Gemini model {model} returned an empty response.")
        except Exception as error:
            last_error = error
            # If no usable stream has arrived, move to another model when overloaded.
            if received_text or not _is_temporary_error(error):
                print(f"Gemini streaming failed for {model}: {error}")
                yield {"type": "error", "message": "The assistant could not finish this answer. Please try again."}
                return

    print(f"All Gemini streaming models are temporarily unavailable: {last_error}")
    yield {"type": "token", "text": _source_fallback(sources)}
