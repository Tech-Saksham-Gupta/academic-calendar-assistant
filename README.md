# Campus Compass — Academic Calendar Assistant

A polished, deployable RAG assistant for academic dates and course information. It retrieves relevant passages from the supplied calendar and syllabus, then asks Gemini to answer using those passages and cite its sources.

It also includes a **calendar-grounded exam planner**: students can provide an exam date, subject, preparation level, available study time, and difficult topics. For planning questions, the app passes the uploaded calendar's dated events to Gemini so it can account for nearby academic deadlines and exam periods. Student-provided exam details are clearly distinguished from dates verified by the calendar.

## Before deployment

1. Each visitor can upload their own institution's calendar through the app. Calendars are scoped to that visitor's browser workspace and are not shared with other visitors. Readable PDF and CSV event dates also populate the **Coming up** sidebar and calendar progress bar.
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

Upload `.txt`, `.md`, `.csv`, or `.pdf` files from the app. Each browser receives a random, HttpOnly session cookie, and uploaded files and indexes are stored under a separate hashed folder for that browser. PDF text is extracted in both regular and layout-preserving modes; pages that still have no readable calendar events are also checked with English OCR in the Docker deployment. The sidebar recognizes common month-name and numeric date formats. OCR can still miss text in blurry, rotated, handwriting-heavy, password-protected, or unusually formatted PDFs, so check that the success banner reports dated events. For local Windows use of scanned PDFs, install Tesseract OCR and add it to PATH. For best CSV results, use columns such as `date,event,category,details,source`.

This is anonymous browser-level separation, not a login system: a visitor's workspace is tied to that browser profile and does not follow them to another device. Clearing cookies creates a new workspace. Add account sign-in and managed storage if users need to recover their workspace across devices.

Chat requests call Gemini even when retrieval finds no passages. In that case, Gemini is instructed to say the uploaded documents do not contain the answer rather than inventing an institutional date. The answer generator uses Gemini 3.5 Flash-Lite with minimal thinking for fast calendar replies, then tries Gemini 3.6 Flash and 3.8 Flash if the service is busy. Clear keyword matches are retrieved locally without an extra embedding API request; less direct questions still use Gemini embeddings. If models are unavailable, the app shows relevant source passages instead of a raw error. `/api/health` reports whether a key is configured and whether indexing reported an error.

## Deploy on Render

1. Push this project folder to a GitHub repository. If it is inside a larger repository, set Render's Root Directory to `academic-calendar-assistant`.
2. In Render, choose **New → Blueprint**, connect the GitHub repository, and select this project's `render.yaml`.
3. In the Blueprint environment form, enter your `GEMINI_API_KEY`. Keep it secret; it stays on the server. No upload password is required because visitor uploads are separated by browser session.
4. Create the service and wait for the first deploy to finish. Open the Render URL and check `/api/health`; it should show `gemini_configured: true` and a nonzero `indexed_passages` count. The configured generation model is `gemini-3.5-flash-lite` for low-latency calendar answers.
5. Each visitor can select **Add a source** and upload their own calendar as a PDF or CSV. For scanned PDFs, Docker's Tesseract OCR attempts to read the page. For CSV, use columns such as `date,event,category,details,source`.

This project is configured for Render's free web service. Free services do not have persistent disks, so uploaded calendars and browser workspaces can be lost when the service restarts or redeploys; visitors may need to upload their calendar again. The vector index is rebuilt from the bundled syllabus and that visitor's uploaded files when needed.

## Project map

- `app/main.py` — API routes, calendar-aware study-plan context, and static web serving.
- `app/rag.py` — chunking, Gemini embeddings, local cosine search, and source indexing.
- `app/gemini.py` — grounded answer generation, calendar-aware exam plans, and clear API-key or Gemini errors.
- `app/documents.py` — loading CSV, Markdown, text, and PDF knowledge.
- `app/config.py` — settings and paths.
- `static/` — responsive, accessible chat interface.
- `data/` — replaceable calendar and supplied syllabus starter knowledge.

The local vector index uses Gemini embeddings and is intentionally small and simple for a student project. For a large campus corpus or multi-instance deployment, move vectors and uploads to a managed database/object store and add authentication to document administration.
