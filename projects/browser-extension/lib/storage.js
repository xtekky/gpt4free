/**
 * g4f extension — chrome.storage wrapper with defaults merging.
 *
 * Storage areas:
 *   - settings            -> chrome.storage.sync  (small, roams across devices)
 *   - conversations       -> chrome.storage.local (large; sync has an 8KB
 *                            per-item quota that full chat histories exceed)
 *   - activeConversation  -> chrome.storage.local
 */
import { DEFAULT_SETTINGS, STORAGE_KEYS } from "./constants.js";

const syncArea = chrome.storage.sync;
const localArea = chrome.storage.local;

export async function getSettings() {
  const result = await syncArea.get(STORAGE_KEYS.settings);
  const stored = result[STORAGE_KEYS.settings] || {};
  return { ...DEFAULT_SETTINGS, ...stored };
}

export async function saveSettings(patch) {
  const current = await getSettings();
  const next = { ...current, ...patch };
  await syncArea.set({ [STORAGE_KEYS.settings]: next });
  return next;
}

export async function resetSettings() {
  await syncArea.set({ [STORAGE_KEYS.settings]: { ...DEFAULT_SETTINGS } });
  return { ...DEFAULT_SETTINGS };
}

/* ------------------------- conversations ------------------------- */

export async function getConversations() {
  const result = await localArea.get(STORAGE_KEYS.conversations);
  return result[STORAGE_KEYS.conversations] || [];
}

export async function saveConversations(list) {
  // Keep storage bounded: newest 50 conversations. chrome.storage.local
  // allows ~10 MB total with no per-item limit (sync would throw
  // QUOTA_BYTES_PER_ITEM at ~8 KB).
  await localArea.set({ [STORAGE_KEYS.conversations]: list.slice(0, 50) });
}

export async function getActiveConversationId() {
  const result = await localArea.get(STORAGE_KEYS.activeConversation);
  return result[STORAGE_KEYS.activeConversation] || null;
}

export async function setActiveConversationId(id) {
  await localArea.set({ [STORAGE_KEYS.activeConversation]: id });
}

export function newConversation(title = "New chat") {
  return {
    id: "c_" + Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
    title,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
}
