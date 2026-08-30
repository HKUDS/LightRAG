import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

const source = readFileSync(join(import.meta.dir, 'WorkspaceWelcome.tsx'), 'utf8')

describe('workspace welcome branding', () => {
  test('does not render the deployment description', () => {
    expect(source).not.toContain('content.brandDescription')
  })
})
