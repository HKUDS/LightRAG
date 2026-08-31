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

interface MarkdownHeading {
  level: number
  text: string
}

/** Every non-empty heading, in document order, ignoring fenced code blocks so
 * a `#` comment inside an example is never read as one. Both spellings count:
 * ATX (`# Title`) and setext (`Title` over `===` / `---`), because a bundle
 * author writes whichever their editor produces. */
function collectHeadings(markdown: string): MarkdownHeading[] {
  const headings: MarkdownHeading[] = []
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
      // Closing sequence (`## Title ##`) is decoration, not text; an empty
      // heading is a heading with no title and is not collected.
      const text = stripInlineMarkdown((atx[2] ?? '').replace(/\s+#+\s*$/, ''))
      if (text) headings.push({ level: atx[1].length, text })
      continue
    }
    const next = lines[index + 1]
    if (next !== undefined && SETEXT_RE.test(next) && isSetextCandidate(line)) {
      const text = stripInlineMarkdown(line)
      if (text) headings.push({ level: next.trim().startsWith('=') ? 1 : 2, text })
    }
  }
  return headings
}

/**
 * The document's own title, as plain text, or null when it has none.
 *
 * NOT simply "the first heading". A heading counts as the document's title
 * only when it both OPENS the document and is the ONLY heading at its level,
 * with nothing shallower anywhere — i.e. it stands over the whole file rather
 * than over the first part of it.
 *
 * The rejected case is the one that matters, and it is a real shape: an
 * agreement written as `## Privacy Policy` … `## Model Service Agreement` has
 * no title, only two peer sections, and the docs shipped exactly that example
 * before the dialog rendered the file as-is. Taking the first of the two would
 * name the dialog "Privacy Policy" for a document that also carries the model
 * service agreement — announcing a NARROWER consent scope than the visitor is
 * being asked for, which is worse than having no title at all. Null sends the
 * caller to the checkbox's own link text, which names the whole document.
 *
 * A single `## Only Heading` is still a title: one section is the document.
 */
export function extractMarkdownTitle(markdown: string | null | undefined): string | null {
  if (!markdown) return null
  const headings = collectHeadings(markdown)
  if (headings.length === 0) return null
  const shallowest = Math.min(...headings.map((heading) => heading.level))
  if (headings[0].level !== shallowest) return null
  if (headings.filter((heading) => heading.level === shallowest).length > 1) return null
  return headings[0].text
}
