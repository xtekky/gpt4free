const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const script = fs.readFileSync(path.join(__dirname, "../content/bridge.js"), "utf8");

test("delivers bounded batches and acknowledges the exact sent snapshots", async () => {
  const data = { trackSources: { chatgpt: true } };
  for (let i = 0; i < 12; i++) {
    data[`trackedChat:chatgpt:${i}`] = { source: "chatgpt", id: String(i), messages: [{ role: "user", content: "old" }] };
  }
  const posted = [];
  const acknowledged = [];
  let onMessage;
  let initial;
  const window = {
    location: { origin: "https://g4f.dev" },
    postMessage(message) { posted.push(structuredClone(message)); },
    addEventListener(type, handler) { if (type === "message") onMessage = handler; },
  };
  const chrome = {
    runtime: {
      id: "extension",
      async sendMessage(message) {
        acknowledged.push(structuredClone(message.conversations));
        for (const conv of message.conversations) {
          const key = `trackedChat:${conv.source}:${conv.id}`;
          if (JSON.stringify(data[key]) === JSON.stringify(conv)) delete data[key];
        }
        return { ok: true };
      },
    },
    storage: {
      local: { async get(key) {
        if (key === "trackSources") return structuredClone({ trackSources: data.trackSources });
        return structuredClone(data);
      } },
      onChanged: { addListener() {} },
    },
  };
  vm.runInNewContext(script, {
    chrome, window, console, setTimeout(fn) { initial = fn; }, setInterval() {}, clearTimeout() {}, clearInterval() {},
  });
  await initial();
  assert.equal(posted[0].conversations.length, 10);
  data["trackedChat:chatgpt:0"].messages[0].content = "new";
  await onMessage({ source: window, data: { type: "g4f-tracked-chats-ack", ids: posted[0].conversations.map((c) => c.id) } });
  assert.equal(acknowledged[0][0].messages[0].content, "old");
  assert.equal(posted[1].conversations.length, 3);
  assert.equal(posted[1].conversations[0].messages[0].content, "new");
});
