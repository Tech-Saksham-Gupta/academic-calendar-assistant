# Campus Compass — Academic Calendar Assistant

A polished, deployable RAG assistant for academic dates and course information. It retrieves relevant passages from the supplied calendar and syllabus, then asks Gemini to answer using those passages and cite its sources.

## Before deployment

1. Each visitor can upload their own institution's official calendar through the app. Calendars are scoped to that visitor's browser workspace and are not shared with other visitors.
2. Review `data/ai_syllabus.md`; it is starter knowledge transcribed from the syllabus image supplied for this project.
3. Create a Gemini API key in Google AI Studio and add it as `GEMINI_API_KEY` in your deployment's environment settings. Never commit the key.

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows; use cp .env.example .env on macOS/Linux
# Put your Gemini key in .env, then:
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. A Gemini API key is required for chat answers.

## Add your own knowledge

Upload `.txt`, `.md`, `.csv`, or text-based `.pdf` files from the app. Each browser receives a random, HttpOnly session cookie, and uploaded files and indexes are stored under a separate hashed folder for that browser. For CSV calendars, use columns such as `date,event,category,details,source`. Scanned PDFs need OCR before adding.

This is anonymous browser-level separation, not a login system: a visitor's workspace is tied to that browser profile and does not follow them to another device. Clearing cookies creates a new workspace. Add account sign-in and managed storage if users need to recover their workspace across devices.

Chat requests call Gemini even when retrieval finds no passages. In that case, Gemini is instructed to say the uploaded documents do not contain the answer rather than inventing an institutional date. `/api/health` reports whether a key is configured and whether indexing reported an error.

## Deploy on Render

1. Push this project folder to a GitHub repository. If it is inside a larger repository, set Render's Root Directory to `academic-calendar-assistant`.
2. In Render, choose **New → Blueprint**, connect the GitHub repository, and select this project's `render.yaml`.
3. In the Blueprint environment form, enter your `GEMINI_API_KEY`. Keep it secret; it stays on the server. No upload password is required because visitor uploads are separated by browser session.
4. Create the service and wait for the first deploy to finish. Open the Render URL and check `/api/health`; it should show `gemini_configured: true` and a nonzero `indexed_passages` count.
5. Each visitor can select **Add a source** and upload their own calendar as a text-based PDF or CSV. For CSV, use columns such as `date,event,category,details,source`.

This project is configured for Render's free web service. Free services do not have persistent disks, so uploaded calendars and browser workspaces can be lost when the service restarts or redeploys; visitors may need to upload their calendar again. The vector index is rebuilt from the bundled syllabus and that visitor's uploaded files when needed. Scanned PDFs need OCR before uploading.

## Project map

- `app/main.py` — API routes and static web serving.
- `app/rag.py` — chunking, Gemini embeddings, local cosine search, and source indexing.
- `app/gemini.py` — grounded answer generation and clear API-key or Gemini errors.
- `app/documents.py` — loading CSV, Markdown, text, and PDF knowledge.
- `app/config.py` — settings and paths.
- `static/` — responsive, accessible chat interface.
- `data/` — replaceable calendar and supplied syllabus starter knowledge.

The local vector index uses Gemini embeddings and is intentionally small and simple for a student project. For a large campus corpus or multi-instance deployment, move vectors and uploads to a managed database/object store and add authentication to document administration.
