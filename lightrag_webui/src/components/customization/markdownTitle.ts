/**
 * The title of a Markdown document, read out of the document itself.
 *
 * Used by the login page's agreement dialog: the document renders as-is, so
 * its own first heading IS the visible title, and the dialog must not print a
 * second one above it. What is still needed is an ACCESSIBLE name for the
 * dialog (a Radix `Dialog.Title` is required, and a screen reader announces
 * it on open) — that name is this function's job, so the announced title and
 * the printed one are the same words instead of two independent strings that
 * can drift apart.
 *
 * Pure and separately tested because it is parsing, not rendering: the repo's
 * frontend tests cover logic and there is no DOM test setup.
 */

/** Fence openers/closers: at least three backticks or tildes, indented < 4. */
const FENCE_RE = /^ {0,3}(`{3,}|~{3,})/
/** ATX heading: `#`…`######` plus a space (or the empty `#` line). */
const ATX_RE = /^ {0,3}(#{1,6})(?:\s+(.*))?$/
/** Setext underline: `===` (h1) or `---` (h2) under a paragraph line. */
const SETEXT_RE = /^ {0,3}(=+|-+)\s*$/

/**
 * Strips the inline Markdown a heading may carry, so the announced name is
 * words rather than syntax. Deliberately small: emphasis, inline code,
 * strikethrough, links/images and backslash escapes are what headings
 * actually contain — anything else is left alone rather than guessed at.
 */
function stripInlineMarkdown(text: string): string {
  return text
    // Images before links: `![alt](src)` keeps its alt text.
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    // Reference links: `[text][id]` and the shortcut `[text]`.
    .replace(/\[([^\]]*)\]\[[^\]]*\]/g, '$1')
    .replace(/\[([^\]]*)\]/g, '$1')
    .replace(/`+([^`]*)`+/g, '$1')
    .replace(/(\*\*\*|___)(.+?)\1/g, '$2')
    .replace(/(\*\*|__)(.+?)\1/g, '$2')
    .replace(/(\*|_)(.+?)\1/g, '$2')
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/\\([\\`*_{}[\]()#+\-.!~>|])/g, '$1')
    .trim()
}

/** A line that can carry a setext underline: a plain paragraph line, not a
 * quote, list item, table row or heading — where `---` means something else
 * entirely (a horizontal rule, or a list item's own bullet). */
function isSetextCandidate(line: string): boolean {
  const trimmed = line.trim()
  if (!trimmed) return false
  if (/^ {4,}/.test(line)) return false // indented code block
  return !/^([>#|]|[-*+]\s|\d+[.)]\s)/.test(trimmed)
}

/**
 * The document's first heading, as plain text, or null when it has none.
 *
 * Fenced code blocks are skipped, so a `#` comment inside an example block is
 * never mistaken for the title. Both heading spellings count: ATX (`# Title`)
 * and setext (`Title` over `===`), because a bundle author writes whichever
 * their editor produces.
 */
export function extractMarkdownTitle(markdown: string | null | undefined): string | null {
  if (!markdown) return null
  const lines = markdown.split(/\r?\n/)
  let fence: string | null = null
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    const fenceMatch = FENCE_RE.exec(line)
    if (fence !== null) {
      // Inside a fence: only a closer of the SAME character ends it.
      if (fenceMatch && fenceMatch[1][0] === fence[0] && fenceMatch[1].length >= fence.length) {
        fence = null
      }
      continue
    }
    if (fenceMatch) {
      fence = fenceMatch[1]
      continue
    }
    const atx = ATX_RE.exec(line)
    if (atx) {
      // Closing sequence (`## Title ##`) is decoration, not text.
      const text = stripInlineMarkdown((atx[2] ?? '').replace(/\s+#+\s*$/, ''))
      // An empty heading is a heading with no title: keep looking rather
      // than announcing a blank name.
      if (text) return text
      continue
    }
    const next = lines[index + 1]
    if (next !== undefined && SETEXT_RE.test(next) && isSetextCandidate(line)) {
      const text = stripInlineMarkdown(line)
      if (text) return text
    }
  }
  return null
}
