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

function addMessage(text, role, sources = []) {
  const row = document.createElement('article');
  row.className = `message ${role}`;
  const mark = document.createElement('div');
  mark.className = 'message-mark';
  mark.textContent = role === 'user' ? 'S' : '✳';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  // textContent safely displays answers without interpreting model output as HTML.
  bubble.textContent = text;
  row.append(mark, bubble);
  conversation.append(row);
  if (sources.length) {
    const sourceRow = document.createElement('div');
    sourceRow.className = 'sources';
    sources.forEach((source) => {
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
  // Only the owner who configured the server token should upload documents.
  const ownerToken = window.prompt('Enter the private document upload token:');
  if (!ownerToken) return;
  showToast(`Adding ${file.name}…`);
  try {
    const response = await fetch('/api/documents', { method: 'POST', headers: { 'X-Admin-Token': ownerToken }, body: data });
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
