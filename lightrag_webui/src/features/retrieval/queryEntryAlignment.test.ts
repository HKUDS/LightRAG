import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

const adminSource = readFileSync(join(import.meta.dir, '..', 'RetrievalView.tsx'), 'utf8')
const workspaceSource = readFileSync(
  join(import.meta.dir, '..', 'workspace', 'WorkspaceQueryView.tsx'),
  'utf8'
)

describe('query entry input alignment', () => {
  test.each([
    ['webui', adminSource],
    ['workspace', workspaceSource]
  ])('%s uses shared prefix parsing and preserves the displayed input', (_entry, source) => {
    expect(source).toContain('prepareQueryInput(input, getQuerySettingsSnapshot().mode)')
    expect(source).toContain('modeOverride: prepared.modeOverride')
    expect(source).toContain('displayedInput: input')
  })
})
