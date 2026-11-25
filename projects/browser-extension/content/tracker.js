/**
 * g4f extension — opt-in account tracker (classic content script).
 * Registered dynamically by the service worker only for sources the user
 * enabled. Reads the visible conversation and hands it to the extension;
 * nothing leaves the browser until the user imports it in g4f.dev/chat.
 */
(async () => {
  const host = location.hostname;
  const params = new URLSearchParams(location.search);
  let source = null;
  if (host === "chatgpt.com" || host === "chat.openai.com") source = "chatgpt";
  else if (host === "gemini.google.com") source = "gemini";
  else if (host === "www.google.com" || host === "google.com") {
    if (params.get("udm") === "50") source = "google_ai";
    else if (location.pathname === "/search" && params.get("q")) source = "google_search";
  }
  if (!source) return;
  if (host === "chatgpt.com") return; // Full conversations are collected through the history API.

  const { trackSources = {} } = await chrome.storage.local.get("trackSources");
  if (!trackSources[source]) return;

  const text = (el) => (el?.innerText || "").trim();
  const idFromUrl = () => location.pathname.split("/").filter(Boolean).pop() || params.get("q") || "";

  function collect() {
    const messages = [];
    let title = document.title.replace(/ - (Google Search|Gemini)$/, "");
    let id = idFromUrl();
    if (source === "chatgpt") {
      document.querySelectorAll("[data-message-author-role]").forEach((el) => {
        const content = text(el);
        if (content) messages.push({ role: el.dataset.messageAuthorRole === "user" ? "user" : "assistant", content });
      });
      if (!location.pathname.startsWith("/c/")) return null;
    } else if (source === "gemini") {
      document.querySelectorAll("user-query, model-response").forEach((el) => {
        const content = text(el);
        if (content) messages.push({ role: el.tagName === "USER-QUERY" ? "user" : "assistant", content });
      });
      if (!location.pathname.includes("/app/")) return null;
    } else {
      const q = params.get("q");
      const answer = text(document.querySelector('[data-attrid="wa:/description"], #rso [data-md], [role="main"]'));
      if (!q) return null;
      messages.push({ role: "user", content: q });
      if (answer) messages.push({ role: "assistant", content: answer.slice(0, 20000) });
      id = q;
      title = q;
    }
    if (!messages.length) return null;
    return { source, id, title, messages, created: Date.now(), updated: Date.now() };
  }

  let last = "";
  let timer = null;
  const push = () => {
    const conv = collect();
    if (!conv) return;
    const sig = conv.id + conv.messages.length + conv.messages.at(-1).content.length;
    if (sig === last) return;
    last = sig;
    chrome.runtime.sendMessage({ type: "g4f:tracked", conversation: conv }).catch(() => {});
  };
  new MutationObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(push, 3000);
  }).observe(document.body, { childList: true, subtree: true, characterData: true });
  timer = setTimeout(push, 3000);
})();
