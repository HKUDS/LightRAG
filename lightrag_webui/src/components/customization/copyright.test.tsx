import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'
import type { ReactElement } from 'react'
import CustomizedCopyright from './CustomizedCopyright'

const hookSource = readFileSync(join(import.meta.dir, 'useCustomizedContent.ts'), 'utf8')
const welcomeSource = readFileSync(
  join(import.meta.dir, '..', '..', 'features', 'workspace', 'WorkspaceWelcome.tsx'),
  'utf8'
)
const loginSource = readFileSync(
  join(import.meta.dir, '..', '..', 'features', 'LoginPage.tsx'),
  'utf8'
)

// The component uses no hooks, so calling it returns the element tree
// directly — enough to assert what it renders without a DOM.
const render = (copyright: string) =>
  CustomizedCopyright({ copyright }) as ReactElement<{
    children?: string
  }> | null

describe('customized copyright line', () => {
  test('renders the bundle text', () => {
    const element = render('© 2025 ACME Inc.')
    expect(element).not.toBeNull()
    expect(element!.type).toBe('footer')
    expect(element!.props.children).toBe('© 2025 ACME Inc.')
  })

  test('renders nothing when there is no text to show', () => {
    // Undeclared, empty and whitespace-only are ONE state: no line, and no
    // footer element drawing padding at the foot of the page either.
    expect(render('')).toBeNull()
    expect(render('   ')).toBeNull()
    expect(render('\n\t')).toBeNull()
  })

  test('has no built-in text for an uncustomized deployment', () => {
    // The line is the deployment's own legal assertion: it comes from the
    // bundle manifest or it does not exist. No i18n default, no LightRAG
    // notice printed on a customer's page.
    const uncustomized = hookSource.slice(hookSource.indexOf('customized: false'))
    expect(uncustomized).toContain('copyright: \'\'')
    expect(hookSource).toContain(
      'copyright: snapshot.brand.copyright?.trim() ?? \'\''
    )
    expect(hookSource).not.toContain('welcome.copyright')
    expect(welcomeSource).not.toContain('welcome.copyright')
  })

  test('sits outside the card, at the foot of both pre-login pages', () => {
    for (const source of [welcomeSource, loginSource]) {
      expect(source).toContain('copyright={content.copyright}')
      const footer = source.indexOf('<CustomizedCopyright')
      expect(footer).toBeGreaterThan(0)
      // Rendered as a sibling AFTER the card, never inside it: nothing that
      // belongs to the card's own subtree may follow the footer.
      const tail = source.slice(footer)
      expect(tail).not.toContain('</Card>')
      expect(tail).not.toContain('max-w-[520px]')
      expect(tail).not.toContain('max-w-[480px]')
    }
    // The card wrapper takes the leftover height so the footer keeps the
    // page's bottom edge instead of overlapping the card.
    expect(welcomeSource).toContain('flex min-h-0 w-full flex-1 items-center justify-center')
    expect(loginSource).toContain('flex w-full flex-1 items-center justify-center')
  })
})
