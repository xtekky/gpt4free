# Roadmap — Next Features

Prioritized list of planned features for the gpt4free browser extension.
✅ = shipped in v0.1 · 🚧 = in progress · ⬜ = planned

## v0.2 — Accounts & Sync (current)

- ✅ g4f.space OAuth sign-in (PKCE, GitHub / Discord / HuggingFace)
- ✅ Conversation cloud sync via `/v1/secret/conversations` (encrypted at rest)
- ✅ Auto-push after each response + manual two-way sync
- ⬜ Conversation delete propagation (local → remote tombstones)
- ⬜ Sync status indicator (last-synced timestamp, pending changes count)
- ⬜ Cross-device workspace-secret request flow (`/v1/secret/request`)

## v0.3 — Chat power features

- ⬜ **Conversation search & history browser** — full-text search across past chats
- ⬜ **Prompt library** — save / reuse / share prompt templates with variables
- ⬜ **Provider picker per chat** — override the provider per conversation, not just globally
- ⬜ **Regenerate & edit** — re-run the last exchange, edit a sent message
- ⬜ **Multi-model compare** — send one prompt to two models side by side
- ⬜ **Token/usage stats** — per-conversation token estimates, usage via members API

## v0.4 — Rich input & output

- ⬜ **Voice input** — Web Speech API dictation into the panel
- ⬜ **Vision / image attach** — paste or drop images for vision models (`m.vision`)
- ⬜ **Text-to-speech** — read answers aloud via `/v1/audio` endpoints
- ⬜ **YouTube / PDF context** — transcript & document ingestion as page context
- ⬜ **LaTeX rendering** — KaTeX in markdown output

## v0.5 — Workflow integration

- ⬜ **Page Q&A mode** — auto-RAG over the current page, no checkbox needed
- ⬜ **Selection inline actions** — floating "Ask g4f" bubble on text selection
- ⬜ **Custom context menus** — user-defined right-click actions from the prompt library
- ⬜ **Keyboard-first mode** — command palette (Ctrl+K) for actions & model switching
- ⬜ **Export** — copy conversation as Markdown / JSON, download images

## v1.0 — Platform

- ⬜ **Firefox port** — WebExtension with `browser.*` polyfill
- ⬜ **Edge/Brave store packaging** — CRX packaging + store assets
- ⬜ **Options import/export** — backup settings & prompts as JSON
- ⬜ **i18n** — UI translations (locales/)
- ⬜ **Accessibility pass** — ARIA roles, focus management, reduced motion
- ⬜ **CI** — lint + manifest validation + icon build on push

## Ideas (unscheduled)

- MCP integration: expose extension page-context as an MCP tool for the g4f server
- Shared team prompt packs via secret workspace
- Side-panel notebook mode (multi-turn scratchpad with cells)
- Auto-model routing (cheap model for short prompts, big model for code)
