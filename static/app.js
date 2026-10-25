// Browser behavior stays separate from the layout and styling for easy learning.
const form = document.querySelector('#chat-form');
const input = document.querySelector('#question');
const conversation = document.querySelector('#conversation');
const welcome = document.querySelector('#welcome');
const filePicker = document.querySelector('#file-picker');
const toast = document.querySelector('#toast');

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
