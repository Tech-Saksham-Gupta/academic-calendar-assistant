// Browser behavior stays separate from the layout and styling for easy learning.
const form = document.querySelector('#chat-form');
const input = document.querySelector('#question');
const conversation = document.querySelector('#conversation');
const welcome = document.querySelector('#welcome');
const filePicker = document.querySelector('#file-picker');
const toast = document.querySelector('#toast');
const dateFormatter = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' });

function parseLocalDate(isoDate) {
  const [year, month, day] = isoDate.split('-').map(Number);
  return new Date(year, month - 1, day);
}

function formatDateRange(startIso, endIso) {
  const start = parseLocalDate(startIso);
  const end = parseLocalDate(endIso);
  return startIso === endIso ? dateFormatter.format(start) : `${dateFormatter.format(start)} – ${dateFormatter.format(end)}`;
}

async function loadCalendarPreview() {
  const emptyCard = document.querySelector('#calendar-empty');
  const eventList = document.querySelector('#upcoming-events');
  try {
    const response = await fetch('/api/calendar');
    if (!response.ok) throw new Error('Could not load calendar dates.');
    const data = await response.json();
    const events = data.events || [];
    eventList.replaceChildren();
    if (!events.length) {
      emptyCard.hidden = false;
      document.querySelector('#empty-calendar-title').textContent = 'Your dates will live here';
      document.querySelector('#empty-calendar-message').textContent = 'Upload a calendar with readable event dates to see what is coming up.';
      document.querySelector('#calendar-range').textContent = 'YOUR CALENDAR';
      document.querySelector('#progress-label').textContent = '—';
      document.querySelector('#calendar-progress-value').style.width = '0%';
      document.querySelector('#progress-caption').textContent = 'Upload a calendar to see your dates.';
      return;
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const upcoming = events.filter((event) => parseLocalDate(event.end_date) >= today).slice(0, 6);
    const firstDate = parseLocalDate(events[0].start_date);
    const lastIso = events.reduce((latest, event) => event.end_date > latest ? event.end_date : latest, events[0].end_date);
    const lastDate = parseLocalDate(lastIso);
    const rangeYears = firstDate.getFullYear() === lastDate.getFullYear()
      ? String(firstDate.getFullYear())
      : `${firstDate.getFullYear()}–${String(lastDate.getFullYear()).slice(-2)}`;
    document.querySelector('#calendar-range').textContent = rangeYears;

    const totalDuration = lastDate.getTime() - firstDate.getTime();
    const elapsed = today.getTime() - firstDate.getTime();
    const percent = totalDuration > 0 ? Math.max(0, Math.min(100, Math.round((elapsed / totalDuration) * 100))) : 100;
    document.querySelector('#progress-label').textContent = `${percent}%`;
    document.querySelector('#calendar-progress-value').style.width = `${percent}%`;
    document.querySelector('#progress-caption').textContent = `${events.length} dated events in your calendar`;

    if (!upcoming.length) {
      emptyCard.hidden = false;
      document.querySelector('#empty-calendar-title').textContent = 'You’re all caught up';
      document.querySelector('#empty-calendar-message').textContent = 'There are no future dates in this calendar.';
      return;
    }

    emptyCard.hidden = true;
    upcoming.forEach((event) => {
      const row = document.createElement('article');
      row.className = 'upcoming-event';
      const dateBadge = document.createElement('div');
      dateBadge.className = 'event-date';
      dateBadge.textContent = formatDateRange(event.start_date, event.end_date);
      const info = document.createElement('div');
      info.className = 'event-info';
      const title = document.createElement('strong');
      title.textContent = event.title;
      info.append(title);
      if (event.term && event.term !== 'Academic calendar') {
        const term = document.createElement('span');
        term.textContent = event.term;
        info.append(term);
      }
      row.append(dateBadge, info);
      eventList.append(row);
    });
  } catch (error) {
    emptyCard.hidden = false;
    document.querySelector('#empty-calendar-title').textContent = 'Calendar preview unavailable';
    document.querySelector('#empty-calendar-message').textContent = error.message;
  }
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  window.setTimeout(() => toast.classList.remove('show'), 3200);
}

function renderAnswerText(container, text) {
  // Render the small Markdown subset Gemini commonly uses without injecting HTML.
  // Unknown markup is left as plain text, so model output cannot run scripts.
  const lines = String(text).split(/\r?\n/);
  lines.forEach((line, lineIndex) => {
    if (lineIndex > 0) container.append(document.createElement('br'));
    const cleaned = line.replace(/^\s*[-*]\s+/, '');
    const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
    let cursor = 0;
    for (const match of cleaned.matchAll(pattern)) {
      const index = match.index;
      if (index > cursor) container.append(document.createTextNode(cleaned.slice(cursor, index)));
      const isBold = match[0].startsWith('**');
      const emphasis = document.createElement(isBold ? 'strong' : 'em');
      emphasis.textContent = match[0].slice(isBold ? 2 : 1, isBold ? -2 : -1);
      container.append(emphasis);
      cursor = index + match[0].length;
    }
    if (cursor < cleaned.length) container.append(document.createTextNode(cleaned.slice(cursor)));
  });
}

function addMessage(text, role, sources = []) {
  const row = document.createElement('article');
  row.className = `message ${role}`;
  const mark = document.createElement('div');
  mark.className = 'message-mark';
  mark.textContent = role === 'user' ? 'S' : '✳';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (role === 'assistant') renderAnswerText(bubble, text);
  else bubble.textContent = text;
  row.append(mark, bubble);
  conversation.append(row);
  if (sources.length) {
    const sourceRow = document.createElement('div');
    sourceRow.className = 'sources';
    // Collapse multiple retrieved chunks from the same page into one citation chip.
    const bestScore = Math.max(...sources.map((source) => Number(source.relevance) || 0));
    const cutoff = Math.max(0.12, bestScore * 0.7);
    const uniqueSources = new Map();
    sources.forEach((source) => {
      if ((Number(source.relevance) || 0) < cutoff) return;
      const key = `${source.name}|${source.page ?? ''}`;
      const previous = uniqueSources.get(key);
      if (!previous || source.relevance > previous.relevance) uniqueSources.set(key, source);
    });
    uniqueSources.forEach((source) => {
      const chip = document.createElement('span');
      chip.className = 'source-chip';
      chip.textContent = `${source.name}${source.page ? ` · p. ${source.page}` : ''}`;
      chip.title = source.excerpt;
      sourceRow.append(chip);
    });
    conversation.append(sourceRow);
  }
  conversation.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

async function ask(question) {
  const cleanQuestion = question.trim();
  if (!cleanQuestion) return;
  welcome.hidden = true;
  addMessage(cleanQuestion, 'user');
  input.value = '';
  input.style.height = 'auto';
  const pending = document.createElement('article');
  pending.className = 'message';
  pending.innerHTML = '<div class="message-mark">✳</div><div class="bubble typing">Looking through your sources…</div>';
  conversation.append(pending);
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: cleanQuestion }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Please try again.');
    pending.remove();
    addMessage(result.answer, 'assistant', result.sources || []);
  } catch (error) {
    pending.remove();
    addMessage(`I couldn't complete that request. ${error.message}`, 'assistant');
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  ask(input.value);
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
});
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
document.querySelectorAll('[data-question]').forEach((button) => {
  button.addEventListener('click', () => ask(button.dataset.question));
});

function chooseFile() { filePicker.click(); }
document.querySelector('#upload-button').addEventListener('click', chooseFile);
document.querySelector('#add-source').addEventListener('click', chooseFile);
document.querySelector('#rail-upload').addEventListener('click', chooseFile);
filePicker.addEventListener('change', async () => {
  const file = filePicker.files[0];
  if (!file) return;
  const data = new FormData();
  data.append('file', file);
  showToast(`Adding ${file.name}…`);
  try {
    // The server reads the private browser cookie and saves this file in that user's space.
    const response = await fetch('/api/documents', { method: 'POST', body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Upload failed.');
    showToast(result.message);
    loadCalendarPreview();
  } catch (error) {
    showToast(error.message);
  } finally {
    filePicker.value = '';
  }
});
document.querySelector('#calendar-nav').addEventListener('click', () => {
  input.value = 'What important dates are in the academic calendar?';
  input.focus();
});
document.querySelector('#all-dates').addEventListener('click', () => {
  input.value = 'What important dates are in the academic calendar?';
  input.focus();
});

// Show whether the secret key is configured without ever revealing its value.
fetch('/api/health').then((response) => response.json()).then((status) => {
  const statusText = document.querySelector('#status-text');
  if (!status.gemini_configured) statusText.textContent = 'Add a Gemini API key';
  else if (status.index_error) statusText.textContent = 'Knowledge indexing issue';
  else statusText.textContent = 'Gemini key configured';
}).catch(() => {});

// Load this browser's private calendar preview when the page opens.
loadCalendarPreview();
