/**
 * g4f extension — shared constants & defaults.
 */

export const DEFAULT_SETTINGS = {
  // Base URL of the g4f server (default: public instance; run your own with
  // `python -m g4f --port 1337` and point this at http://localhost:1337).
  serverUrl: "https://g4f.space",
  // Optional API key (G4F_API_KEY). Empty = no auth.
  apiKey: "",
  // Default model id (from /v1/models). Empty = server default.
  model: "",
  // Default provider name. Empty = server default / auto.
  provider: "",
  // Streaming responses.
  stream: true,
  // System prompt prepended to every conversation.
  systemPrompt: "You are a helpful assistant running inside a browser extension. Be concise and use markdown.",
  // Temperature (null = provider default).
  temperature: null,
  // Max tokens (null = provider default).
  maxTokens: null,
  // Include the text of the active tab as context when asked from the side panel.
  includePageContext: false,
  // Max characters of page context to send.
  pageContextLimit: 6000,
  // Image generation defaults.
  imageModel: "",
  imageCount: 1,
  // UI
  fontSize: "medium",
  sendOnEnter: true,
  // Quick actions shown in the popup.
  quickActions: [
    { id: "summarize", label: "Summarize page", prompt: "Summarize the following page content in a few bullet points:\n\n" },
    { id: "explain", label: "Explain selection", prompt: "Explain this in simple terms:\n\n" },
    { id: "translate", label: "Translate to English", prompt: "Translate the following text to English. Only output the translation:\n\n" },
    { id: "reply", label: "Draft a reply", prompt: "Draft a short, friendly reply to this message:\n\n" },
  ],
};

export const STORAGE_KEYS = {
  settings: "settings",
  conversations: "conversations",
  activeConversation: "activeConversationId",
  // OAuth session (chrome.storage.local)
  user: "g4fUser",
  token: "g4fToken",
  expires: "g4fExpires",
  workspaceSecret: "g4fWorkspaceSecret",
};

export const MESSAGE_TYPES = {
  // popup/sidepanel -> background
  CHAT: "g4f:chat",
  CHAT_ABORT: "g4f:chat:abort",
  IMAGE: "g4f:image",
  MODELS: "g4f:models",
  PROVIDERS: "g4f:providers",
  HEALTH: "g4f:health",
  GET_PAGE: "g4f:get-page",
  GET_SELECTION: "g4f:get-selection",
  OPEN_SIDE_PANEL: "g4f:open-side-panel",
  // account & cloud sync
  OAUTH_START: "g4f:oauth:start",
  OAUTH_LOGOUT: "g4f:oauth:logout",
  OAUTH_STATUS: "g4f:oauth:status",
  SYNC_PUSH: "g4f:sync:push",
  SYNC_PULL: "g4f:sync:pull",
  SYNC_NOW: "g4f:sync:now",
  // background -> UI (stream chunks)
  STREAM_CHUNK: "g4f:stream:chunk",
  STREAM_DONE: "g4f:stream:done",
  STREAM_ERROR: "g4f:stream:error",
};

/** Parse a server base URL, trimming trailing slashes. */
export function normalizeBaseUrl(url) {
  let u = (url || "").trim();
  if (!u) return DEFAULT_SETTINGS.serverUrl;
  if (!/^https?:\/\//i.test(u)) u = "http://" + u;
  return u.replace(/\/+$/, "");
}
