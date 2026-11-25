const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const script = fs.readFileSync(path.join(__dirname, "../content/chatgpt-history.js"), "utf8");

function harness({ pages, detail, session = { accessToken: "secret" }, initial = {} }) {
  const data = { trackSources: { chatgpt: true }, ...initial };
  const requests = [];
  const saved = [];
  let interval;
  const chrome = {
    storage: { local: {
      async get(keys) {
        if (typeof keys === "string") keys = [keys];
        return Object.fromEntries(keys.map((key) => [key, data[key]]));
      },
      async set(values) { Object.assign(data, values); },
    } },
    runtime: { async sendMessage(message) {
      saved.push(message.conversation);
      data[`chatgptVersion:${message.conversation.id}`] = message.version;
      return { ok: true };
    } },
  };
  const fetch = async (url, options) => {
    requests.push({ url, options });
    let value;
    if (url === "/api/auth/session") value = session;
    else if (url.startsWith("/backend-api/conversations?")) {
      const offset = Number(new URLSearchParams(url.split("?")[1]).get("offset"));
      value = pages[offset];
    } else value = detail(decodeURIComponent(url.split("/").pop()));
    return { ok: true, async json() { return value; } };
  };
  vm.runInNewContext(script, {
    chrome, fetch, location: { hostname: "chatgpt.com" }, URLSearchParams,
    setInterval(fn) { interval = fn; }, console,
  });
  return { data, requests, saved, tick: () => interval() };
}

async function settled(h) {
  for (let i = 0; i < 100; i++) {
    await new Promise((resolve) => setImmediate(resolve));
    if (/sync complete|import failed/.test(h.data.chatgptImportStatus || "")) return;
  }
  assert.fail("Import did not finish");
}

function conversation(id) {
  return {
    title: `Title ${id}`, create_time: 100, update_time: 200, current_node: "answer",
    mapping: {
      root: { parent: null, message: null },
      question: { parent: "root", message: { author: { role: "user" }, content: { parts: ["Question"] } } },
      hidden: { parent: "question", message: { author: { role: "assistant" }, content: { parts: ["Hidden"] }, metadata: { is_visually_hidden_from_conversation: true } } },
      answer: { parent: "hidden", message: { author: { role: "assistant" }, content: { parts: ["Answer"] } } },
      alternate: { parent: "question", message: { author: { role: "assistant" }, content: { parts: ["Alternate"] } } },
    },
  };
}

test("imports every page and the current message branch, then skips unchanged chats", async () => {
  const summaries = Array.from({ length: 21 }, (_, i) => ({ id: `c${i}`, update_time: 200, title: `Title c${i}` }));
  const h = harness({
    pages: { 0: { items: summaries.slice(0, 20), total: 21 }, 20: { items: summaries.slice(20), total: 21 } },
    detail: conversation,
  });
  await settled(h);
  assert.equal(h.saved.length, 21);
  assert.deepEqual(JSON.parse(JSON.stringify(h.saved[0].messages)), [
    { role: "user", content: "Question" }, { role: "assistant", content: "Answer" },
  ]);
  assert.equal(h.saved[0].created, 100000);
  assert.equal(h.saved[0].updated, 200000);
  const listRequests = h.requests.filter((r) => r.url.startsWith("/backend-api/conversations?"));
  assert.equal(listRequests.length, 2);
  assert.equal(new URLSearchParams(listRequests[1].url.split("?")[1]).get("offset"), "20");
  assert.equal(listRequests[0].options.headers.Authorization, "Bearer secret");
  h.data.chatgptImportStatus = "";
  await h.tick();
  await settled(h);
  assert.equal(h.saved.length, 21);
});

test("reports missing authentication without fetching history", async () => {
  const h = harness({ pages: {}, detail: conversation, session: {} });
  await settled(h);
  assert.match(h.data.chatgptImportStatus, /Sign in to ChatGPT/);
  assert.equal(h.requests.length, 1);
});

test("refreshes conversations when the API does not provide an update timestamp", async () => {
  const h = harness({ pages: { 0: { items: [{ id: "c", title: "Test" }], total: 1 } }, detail: conversation });
  await settled(h);
  h.data.chatgptImportStatus = "";
  await h.tick();
  await settled(h);
  assert.equal(h.saved.length, 2);
});

test("does not fetch when tracking is disabled", async () => {
  const h = harness({ pages: {}, detail: conversation, initial: { trackSources: { chatgpt: false } } });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(h.requests.length, 0);
});
