# Campus Compass — Academic Calendar Assistant

A polished, deployable RAG assistant for academic dates and course information. It retrieves relevant passages from the supplied calendar and syllabus, then asks Gemini to answer using those passages and cite its sources.

## Before deployment

1. Upload your institution's official calendar through the app after deployment. No institution dates are invented or pre-filled.
2. Review `data/ai_syllabus.md`; it is starter knowledge transcribed from the syllabus image supplied for this project.
3. Create a Gemini API key in Google AI Studio and add it as `GEMINI_API_KEY` in your deployment's environment settings. Never commit the key.
4. Set a long, random `ADMIN_UPLOAD_TOKEN` in deployment settings to authorize document uploads. Keep it private.

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows; use cp .env.example .env on macOS/Linux
# Put your Gemini key and a private ADMIN_UPLOAD_TOKEN in .env, then:
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. A Gemini API key is required for chat answers.

## Add your own knowledge

Put `.txt`, `.md`, `.csv`, or text-based `.pdf` files in `data/`; restart the app to index them. Or upload the official calendar from the app using the private `ADMIN_UPLOAD_TOKEN` set in the server environment. For CSV calendars, use columns such as `date,event,category,details,source`. Scanned PDFs need OCR before adding.

Chat requests call Gemini even when retrieval finds no passages. In that case, Gemini is instructed to say the uploaded documents do not contain the answer rather than inventing an institutional date. `/api/health` reports whether a key is configured and whether indexing reported an error.

## Deploy on Render

1. Push this project folder to a GitHub repository. If it is inside a larger repository, set Render's Root Directory to `academic-calendar-assistant`.
2. In Render, choose **New → Blueprint**, connect the GitHub repository, and select this project's `render.yaml`.
3. In the Blueprint environment form, enter your `GEMINI_API_KEY` and a long private `ADMIN_UPLOAD_TOKEN`. Keep both secret. The Gemini key stays on the server; the upload token authorizes calendar/document uploads.
4. Create the service and wait for the first deploy to finish. Open the Render URL and check `/api/health`; it should show `gemini_configured: true` and a nonzero `indexed_passages` count.
5. On the site, select **Add a source**, choose your official calendar as a text-based PDF or CSV, then enter the same `ADMIN_UPLOAD_TOKEN` when prompted. For CSV, use columns such as `date,event,category,details,source`.

The Render disk keeps uploaded files across deploys. The small vector index is rebuilt from the bundled syllabus and uploaded files when needed. Scanned PDFs need OCR before uploading.

## Project map

- `app/main.py` — API routes and static web serving.
- `app/rag.py` — chunking, Gemini embeddings, local cosine search, and source indexing.
- `app/gemini.py` — grounded answer generation and clear API-key or Gemini errors.
- `app/documents.py` — loading CSV, Markdown, text, and PDF knowledge.
- `app/config.py` — settings and paths.
- `static/` — responsive, accessible chat interface.
- `data/` — replaceable calendar and supplied syllabus starter knowledge.

The local vector index uses Gemini embeddings and is intentionally small and simple for a student project. For a large campus corpus or multi-instance deployment, move vectors and uploads to a managed database/object store and add authentication to document administration.
