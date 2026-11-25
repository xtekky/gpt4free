/**
 * Import the signed-in ChatGPT account's history while a ChatGPT tab is open.
 * Authentication stays in this content script and is never stored.
 */
(() => {
  if (location.hostname !== "chatgpt.com") return;

  const interval = 10 * 60 * 1000;
  let running = false;

  const toMillis = (value) => {
    if (typeof value === "number") return value * 1000;
    return Date.parse(value) || 0;
  };
  const text = (value) => {
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return value.map(text).filter(Boolean).join("\n");
    if (value && typeof value === "object") return text(value.parts || value.text || value.content);
    return "";
  };

  function convert(detail, summary) {
    const mapping = detail.mapping;
    if (!mapping || typeof mapping !== "object") throw new Error("ChatGPT returned an invalid conversation");
    let node = detail.current_node || Object.keys(mapping).find((key) => !mapping[key].children?.length);
    const seen = new Set();
    const chain = [];
    while (node && mapping[node] && !seen.has(node)) {
      seen.add(node);
      chain.unshift(mapping[node]);
      node = mapping[node].parent;
    }
    const messages = chain.flatMap(({ message }) => {
      const role = message?.author?.role;
      const content = text(message?.content).trim();
      return ["user", "assistant"].includes(role) && content &&
        !message.metadata?.is_visually_hidden_from_conversation ? [{ role, content }] : [];
    });
    if (!messages.length) return null;
    return {
      source: "chatgpt", id: summary.id, title: detail.title || summary.title || "ChatGPT",
      messages, created: toMillis(detail.create_time || summary.create_time),
      updated: toMillis(detail.update_time || summary.update_time),
    };
  }

  async function getJson(path, token) {
    const response = await fetch(path, {
      credentials: "include",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!response.ok) throw new Error(`ChatGPT request failed (HTTP ${response.status})`);
    return response.json();
  }

  async function enabled() {
    return !!(await chrome.storage.local.get("trackSources")).trackSources?.chatgpt;
  }

  async function sync() {
    if (running || !await enabled()) return;
    running = true;
    try {
      await chrome.storage.local.set({ chatgptImportStatus: "Importing ChatGPT conversations…" });
      const session = await getJson("/api/auth/session");
      if (!session?.accessToken) throw new Error("Sign in to ChatGPT to import conversations");
      let offset = 0;
      let imported = 0;
      const seen = new Set();
      while (await enabled()) {
        const query = new URLSearchParams({
          exclude_conversation_origin: "tpp", expand: "false", hide_snorlax: "false",
          is_archived: "false", is_starred: "false", limit: "20", order: "updated",
          offset: String(offset),
        });
        const page = await getJson(`/backend-api/conversations?${query}`, session.accessToken);
        if (!Array.isArray(page.items)) throw new Error("ChatGPT returned an invalid conversation list");
        if (!page.items.length) break;
        let fresh = 0;
        for (const summary of page.items) {
          if (!await enabled()) return;
          if (typeof summary.id !== "string" || !summary.id) throw new Error("ChatGPT returned a conversation without an ID");
          if (seen.has(summary.id)) continue;
          seen.add(summary.id);
          fresh++;
          const version = summary.update_time == null ? null : JSON.stringify([summary.update_time, summary.title]);
          const versionKey = `chatgptVersion:${summary.id}`;
          if (version !== null && (await chrome.storage.local.get(versionKey))[versionKey] === version) continue;
          const detail = await getJson(`/backend-api/conversation/${encodeURIComponent(summary.id)}`, session.accessToken);
          const conversation = convert(detail, summary);
          if (!conversation) continue;
          const result = await chrome.runtime.sendMessage({ type: "g4f:chatgpt:store", conversation, version });
          if (!result?.ok) throw new Error(result?.error || "Could not save ChatGPT conversation");
          imported++;
        }
        if (!fresh) throw new Error("ChatGPT returned a repeated conversation page");
        offset += page.items.length;
        if (typeof page.total === "number" && offset >= page.total) break;
      }
      if (await enabled()) {
        await chrome.storage.local.set({ chatgptImportStatus: `ChatGPT sync complete: ${imported} new or updated conversations.` });
      }
    } catch (error) {
      console.error("ChatGPT history import failed:", error);
      await chrome.storage.local.set({ chatgptImportStatus: `ChatGPT import failed: ${error.message}` });
    } finally {
      running = false;
    }
  }

  sync();
  setInterval(sync, interval);
})();
