/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

/**
 * `bypassEntryEquivalence.test.ts` proves the shared pipeline produces the same
 * request from both entries. What a behavioural test cannot see is an entry
 * QUITTING that pipeline — reimplementing prefix parsing locally, or dropping a
 * field on the way into it. That is what these source-level assertions guard,
 * so keep them about routing, not about formatting.
 */

const ENTRIES = [
  ['webui', readFileSync(join(import.meta.dir, '..', 'RetrievalView.tsx'), 'utf8')],
  [
    'workspace',
    readFileSync(join(import.meta.dir, '..', 'workspace', 'WorkspaceQueryView.tsx'), 'utf8')
  ]
] as const

describe('query settings entry alignment', () => {
  const workspace = ENTRIES.find(([entry]) => entry === 'workspace')![1]

  test('workspace derives its snapshot through the shared policy', () => {
    // The contract is that this entry USES /webui's configuration. Inlining
    // the spread again is how it would drift: the policy's own tests would
    // keep passing while this entry quietly stopped obeying them.
    expect(workspace).toMatch(
      /import\s*\{[^}]*\bworkspaceQuerySettingsSnapshot\b[^}]*\}\s*from/
    )
    expect(workspace).toMatch(/workspaceQuerySettingsSnapshot\(/)
  })

  test('workspace does not re-clamp settings locally', () => {
    // Any override belongs in WORKSPACE_CLAMPED_SETTINGS, where a test pins
    // the size of the exemption list.
    expect(workspace).not.toMatch(/only_need_context:\s*false/)
    expect(workspace).not.toMatch(/only_need_prompt:\s*false/)
    expect(workspace).not.toMatch(/disable_user_prompt_prefix:/)
  })
})

describe('query entry input alignment', () => {
  test.each(ENTRIES)('%s delegates prefix parsing to the shared helper', (_entry, source) => {
    expect(source).toMatch(/import\s*\{[^}]*\bprepareQueryInput\b[^}]*\}\s*from/)
    expect(source).toMatch(/prepareQueryInput\(\s*input\s*,/)
  })

  test.each(ENTRIES)('%s does not reimplement prefix parsing', (_entry, source) => {
    // A local `/mode ` regex is how the two entries drifted apart before.
    expect(source).not.toMatch(/\/\^\\\/|match\(\s*\/\^/)
    expect(source).not.toContain('allowedModes')
  })

  test.each(ENTRIES)('%s forwards the parsed mode and the raw input', (_entry, source) => {
    // The request carries the prefix-stripped query; the transcript shows what
    // the user actually typed, prefix and all.
    expect(source).toMatch(/modeOverride:\s*prepared\.modeOverride/)
    expect(source).toMatch(/displayedInput:\s*input/)
    expect(source).toMatch(/submitQuery\(\s*prepared\.query/)
  })

  test.each(ENTRIES)('%s surfaces every rejection reason it can get', (_entry, source) => {
    // Interpolating the error key means a new `QueryInputError` member is
    // rendered by both entries without either being touched.
    expect(source).toMatch(/retrievePanel\.retrieval\.\$\{prepared\.error\}/)
  })
})
