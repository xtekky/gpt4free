# Feature Status

## Shipped

- [x] Side panel with the **full g4f.dev chat embedded** (iframe, postMessage bridge)
- [x] Conversations persist **locally** — IndexedDB (`chat-db`) in the embedded
      chat, `chrome.storage.local` (newest 50) in lite mode
- [x] **Native lite chat** fallback with conversation history drawer,
      markdown, streaming, abort, copy buttons
- [x] **Quick actions** — context menus (Ask / Explain / Translate / Summarize)
      inject prompts into the embedded chat; popup quick actions hand off to the panel
- [x] **Page context** — attach active-tab text or selection to lite-chat prompts;
      page summaries via context menu
- [x] Image generation via `/img <prompt>` (lite chat)
- [x] Model & provider pickers, server URL / API key, system prompt,
      temperature / max tokens, streaming toggle
- [x] CDP bridge (browser-as-provider for local g4f servers)
- [x] Keyboard shortcuts: `Alt+Shift+G`, `Alt+Shift+S`
## Planned

- [ ] Firefox (MV3) port — side panel via `sidebar_action`
- [ ] Optional sync of lite-mode conversations to a user-specified server
      (opt-in, end-to-end encrypted)
- [ ] Per-site page-context allowlist
- [ ] Voice input in the lite chat
