import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

const retrievalSource = readFileSync(join(import.meta.dir, 'RetrievalView.tsx'), 'utf8')
const documentManagerSource = readFileSync(join(import.meta.dir, 'DocumentManager.tsx'), 'utf8')
const cardSource = readFileSync(
  join(import.meta.dir, '..', 'components', 'ui', 'Card.tsx'),
  'utf8'
)
const statusIndicatorSource = readFileSync(
  join(import.meta.dir, '..', 'components', 'status', 'StatusIndicator.tsx'),
  'utf8'
)

describe('admin retrieval layout', () => {
  test('matches the document panel bottom clearance and avoids the status indicator', () => {
    expect(cardSource).toMatch(/cn\('p-6 pt-0'/)
    expect(documentManagerSource).toContain('min-h-0 mb-2')
    expect(statusIndicatorSource).toContain('right-4 bottom-4')
    expect(retrievalSource).toContain('px-2 pb-8')
    expect(retrievalSource).not.toContain('pb-12')
  })
})
