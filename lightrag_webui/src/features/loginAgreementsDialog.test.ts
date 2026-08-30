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
  test('prints no title above the document', () => {
    // The link's wording must not become a second, different heading over
    // the file's own — the document supplies its title.
    expect(source).not.toContain('<DialogTitle>{consentDocuments}</DialogTitle>')
    expect(source).toContain('<DialogHeader className="sr-only">')
  })

  test('names the dialog from the document itself', () => {
    expect(source).toContain(
      'extractMarkdownTitle(content.agreementsMarkdown) ?? consentDocuments'
    )
    expect(source).toContain('<DialogTitle>{agreementsTitle}</DialogTitle>')
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
