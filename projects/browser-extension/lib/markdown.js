/**
 * g4f extension — tiny markdown renderer (no dependencies).
 * Supports: headings, bold, italic, inline code, fenced code blocks,
 * links, blockquotes, unordered/ordered lists, hr, line breaks.
 * Escapes HTML first — safe against injection from model output.
 */

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function renderMarkdown(text) {
  if (!text) return "";
  const codeBlocks = [];

  // Extract fenced code blocks first so their content isn't transformed.
  let src = String(text).replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const placeholder = `\u0000CODE${codeBlocks.length}\u0000`;
    codeBlocks.push(
      `<pre class="md-code"><code data-lang="${escapeHtml(lang || "")}">${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`
    );
    return placeholder;
  });

  src = escapeHtml(src);

  // Headings (up to ######)
  src = src.replace(/^######\s+(.+)$/gm, "<h6>$1</h6>")
           .replace(/^#####\s+(.+)$/gm, "<h5>$1</h5>")
           .replace(/^####\s+(.+)$/gm, "<h4>$1</h4>")
           .replace(/^###\s+(.+)$/gm, "<h3>$1</h3>")
           .replace(/^##\s+(.+)$/gm, "<h2>$1</h2>")
           .replace(/^#\s+(.+)$/gm, "<h1>$1</h1>");

  // Horizontal rule
  src = src.replace(/^\s*(---+|\*\*\*+)\s*$/gm, "<hr>");

  // Blockquote
  src = src.replace(/^&gt;\s?(.+)$/gm, "<blockquote>$1</blockquote>");

  // Unordered list (consecutive lines)
  src = src.replace(/(?:^[-*]\s+.+\n?)+/gm, (block) => {
    const items = block.trim().split("\n").map((l) => `<li>${l.replace(/^[-*]\s+/, "")}</li>`).join("");
    return `<ul>${items}</ul>`;
  });

  // Ordered list
  src = src.replace(/(?:^\d+\.\s+.+\n?)+/gm, (block) => {
    const items = block.trim().split("\n").map((l) => `<li>${l.replace(/^\d+\.\s+/, "")}</li>`).join("");
    return `<ol>${items}</ol>`;
  });

  // Bold / italic / strikethrough
  src = src.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
  src = src.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  src = src.replace(/(^|\W)\*([^*\n]+)\*(?=\W|$)/g, "$1<em>$2</em>");
  src = src.replace(/~~(.+?)~~/g, "<del>$1</del>");

  // Inline code
  src = src.replace(/`([^`\n]+)`/g, "<code>$1</code>");

  // Links [text](url) — only http(s) to be safe
  src = src.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  // Bare URLs
  src = src.replace(/(?<!["'>=])(https?:\/\/[^\s<]+)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');

  // Paragraphs: split on double newlines not already inside a tag
  src = src.replace(/\n{2,}/g, "<br><br>").replace(/\n/g, "<br>");

  // Restore code blocks
  src = src.replace(/\u0000CODE(\d+)\u0000/g, (_, i) => codeBlocks[Number(i)]);

  return src;
}

/** Extract the first code block from a message, if any. */
export function extractCodeBlock(text) {
  const m = String(text || "").match(/```(?:\w*)\n?([\s\S]*?)```/);
  return m ? m[1] : null;
}
