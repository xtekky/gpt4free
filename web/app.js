// Use relative API path so the site works when deployed to Vercel.
// Set env G4F_UPSTREAM on the server to point to your g4f backend.
const API_URL = '/api/chat';

const messagesEl = document.getElementById('messages');
const form = document.getElementById('chatForm');
const input = document.getElementById('prompt');

function appendMessage(text, kind = 'bot'){
  const el = document.createElement('div');
  el.className = `msg ${kind}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

form.addEventListener('submit', async e => {
  e.preventDefault();
  const text = input.value.trim();
  if(!text) return;
  appendMessage(text, 'user');
  input.value = '';
  const loading = document.createElement('div');
  loading.className = 'msg bot loading';
  loading.textContent = '…';
  messagesEl.appendChild(loading);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try{
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    if(!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const reply = data.reply ?? (data.choices?.[0]?.message?.content) ?? JSON.stringify(data);
    loading.remove();
    appendMessage(reply, 'bot');
  }catch(err){
    loading.remove();
    appendMessage('Error: ' + err.message, 'bot');
  }
});
