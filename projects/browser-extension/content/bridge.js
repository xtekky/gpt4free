/**
 * g4f extension — bridge on g4f.dev/chat. Relays conversations tracked by the
 * extension to the page (which only imports them if the user opted in) and
 * clears them once acknowledged.
 */
let sending = false;
let stopped = false;
let initialTimer;
let pollTimer;
let sentConversations = [];

const contextInvalidated = (error) =>
  !globalThis.chrome?.runtime?.id ||
  error?.message?.includes("Extension context invalidated");

const handleBridgeError = (operation, error) => {
  if (contextInvalidated(error)) {
    stopped = true;
    clearTimeout(initialTimer);
    clearInterval(pollTimer);
    return;
  }
  console.error(`g4f tracked-chat ${operation} failed:`, error);
};

const send = async (reportEmpty = false) => {
  if (sending || stopped || !globalThis.chrome?.runtime?.id) return;
  sending = true;
  try {
    if (sentConversations.length) {
      const { trackSources = {} } = await chrome.storage.local.get("trackSources");
      sentConversations = sentConversations.filter((conversation) => trackSources[conversation.source]);
    }
    if (sentConversations.length) {
      if (reportEmpty) {
        window.postMessage({ type: "g4f-tracked-chats", conversations: sentConversations }, window.location.origin);
      }
      return;
    }
    const stored = await chrome.storage.local.get(null);
    const { trackedChats = {}, trackSources = {} } = stored;
    const conversations = [...Object.values(trackedChats),
      ...Object.entries(stored).filter(([key]) => key.startsWith("trackedChat:")).map(([, value]) => value)]
      .filter((conversation) => trackSources[conversation.source]).slice(0, 10);
    if (conversations.length) {
      sentConversations = conversations;
      window.postMessage({ type: "g4f-tracked-chats", conversations }, window.location.origin);
    } else if (reportEmpty) {
      window.postMessage({ type: "g4f-tracked-chats-ready" }, window.location.origin);
    }
  } catch (error) {
    handleBridgeError("send", error);
  } finally {
    sending = false;
  }
};
window.addEventListener("message", async (event) => {
  if (stopped || event.source !== window) return;
  if (event.data?.type === "g4f-tracked-chats-request") {
    await send(true);
    return;
  }
  if (event.data?.type !== "g4f-tracked-chats-ack") return;
  try {
    if (!globalThis.chrome?.runtime?.id) {
      handleBridgeError("acknowledgement", new Error("Extension context invalidated"));
      return;
    }
    const ids = new Set(event.data.ids || []);
    const acknowledged = sentConversations.filter((conversation) => ids.has(conversation.id));
    if (!acknowledged.length) return;
    const response = await chrome.runtime.sendMessage({
      type: "g4f:tracked:ack",
      conversations: acknowledged,
    });
    if (!response?.ok) throw new Error(response?.error || "Acknowledgement failed");
    sentConversations = [];
    await send(false);
  } catch (error) {
    handleBridgeError("acknowledgement", error);
  }
});
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.trackSources) {
    const sources = changes.trackSources.newValue || {};
    sentConversations = sentConversations.filter((conversation) => sources[conversation.source]);
  }
  if (!stopped && !sentConversations.length && area === "local" &&
      (changes.trackedChats || changes.trackSources || Object.keys(changes).some((key) => key.startsWith("trackedChat:")))) send(false);
});
initialTimer = setTimeout(send, 2000);
pollTimer = setInterval(send, 60000);
