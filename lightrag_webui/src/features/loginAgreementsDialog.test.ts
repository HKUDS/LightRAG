import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

const source = readFileSync(join(import.meta.dir, 'LoginPage.tsx'), 'utf8')

/**
 * The agreement dialog's shape, asserted on the source: the repo has no DOM
 * test setup, and these are exactly the details a well-meaning edit would
 * undo — reinstating a printed title above the document, or dropping the
 * hidden one that gives the dialog its accessible name.
 */
describe('login agreements dialog', () => {
  const header = source.match(/<DialogHeader[\s\S]*?<\/DialogHeader>/)?.[0] ?? ''

  test('prints no title above the document', () => {
    // The visible title belongs to the file. The dialog's own title element
    // exists only to give the dialog an accessible name, so it lives inside
    // the hidden header and nowhere else — a second one on screen would sit
    // above the document's own heading and disagree with it.
    expect(header).toContain('className="sr-only"')
    expect(source.match(/<DialogTitle>/g)).toHaveLength(1)
  })

  test('names the dialog with the link text, not with parsed Markdown', () => {
    // The accessible name is what the visitor just ticked, and the bundle
    // maintains it beside the file it points at. Nothing reads the document
    // to name the dialog — the dialog renders the document.
    expect(header).toContain('<DialogTitle>{consentDocuments}</DialogTitle>')
    expect(source).not.toContain('extractMarkdownTitle')
  })

  test('renders the document with the document typography', () => {
    expect(source).toContain('variant="document"')
  })

  test('takes the link wording from the bundle, translation as fallback', () => {
    expect(source).toContain('resolveConsentDocuments(')
    expect(source).toContain('content.consentDocuments')
    expect(source).toContain('t(\'login.consentDocuments\')')
  })
})
