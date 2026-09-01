/// <reference types="bun" />
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'
import type { ComponentType } from 'react'
import { act, cleanup, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import { resetCustomization, seedCustomization } from '@/test/customization'
import { defaultQuerySettings, useQuerySettingsStore } from '@/stores/querySettings'
import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'
import { useWorkspaceRetrievalHistoryStore } from '@/stores/workspaceRetrievalHistory'
import type { QuerySettings } from '@/stores/querySettings'
import type { QueryRequest } from '@/api/lightrag'

/**
 * The two query entries (`/webui`'s RetrievalView and the workspace's
 * WorkspaceQueryView) must send the SAME request for the same input. This file
 * pins that by driving both of them through their real composer and reading the
 * request that reaches the API layer — `bypassEntryEquivalence.test.ts` replays
 * the three preparation steps in isolation, which cannot see an entry that
 * stopped taking them.
 *
 * The routing assertions at the bottom stay source-level on purpose: an entry
 * that reimplements prefix parsing IDENTICALLY is invisible to a behavioural
 * test, and reimplementing it is exactly how the two entries drifted apart
 * before.
 */

const requests: QueryRequest[] = []
let realApiModule: Record<string, unknown>
let entries: { webui: ComponentType; workspace: ComponentType }

beforeAll(async () => {
  realApiModule = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({
    ...realApiModule,
    queryTextStream: mock(async (params: QueryRequest) => {
      requests.push(params)
    }),
    queryText: mock(async (params: QueryRequest) => {
      requests.push(params)
      return { response: '' }
    })
  }))

  // Dynamic, and AFTER the mock: `useQuerySession` binds `queryTextStream` at
  // import time, so a static import here would send the query for real.
  entries = {
    webui: (await import('@/features/RetrievalView')).default,
    workspace: (await import('@/features/workspace/WorkspaceQueryView')).default
  }
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
})

const setQuerySettings = (overrides: Partial<QuerySettings> = {}): void => {
  act(() => {
    useQuerySettingsStore.setState({
      querySettings: { ...defaultQuerySettings, ...overrides }
    })
  })
}

beforeEach(() => {
  requests.length = 0
  // The workspace empty state sits behind `useCustomizedContent`, which renders
  // a placeholder — and fires a real request — until a snapshot has settled.
  seedCustomization()
  setQuerySettings()
  act(() => {
    useWebuiRetrievalHistoryStore.getState().clearHistory()
    useWorkspaceRetrievalHistoryStore.getState().clearHistory()
  })
})

afterEach(() => {
  // First, so the store resets below land on an UNMOUNTED tree: the preload's
  // own `cleanup()` runs after this hook, too late to stop a teardown re-render
  // from starting a request that lands during the next test.
  cleanup()
  resetCustomization()
})

const entryNames = ['webui', 'workspace'] as const
type EntryName = (typeof entryNames)[number]

const renderEntry = async (entry: EntryName): Promise<void> => {
  const Entry = entries[entry]
  renderWithProviders(<Entry />)
  await act(async () => {})
}

/** The composer's input, scoped to the send form so the /webui sidebar's own
 * textareas can never be picked up instead. */
const composerInput = (): HTMLElement =>
  within(screen.getByRole('search')).getByRole('textbox')

const send = async (entry: EntryName, input: string): Promise<void> => {
  const user = userEvent.setup()
  await renderEntry(entry)
  await user.type(composerInput(), input)
  await user.keyboard('{Enter}')
  await act(async () => {})
}

describe('query entry input handling', () => {
  // A plain loop rather than `test.each`: Bun's table typings widen every
  // parameter to `unknown`, and the entry name is a key into `entries`.
  for (const entry of entryNames) {
    test(`${entry} rejects an unknown mode prefix with a readable reason`, async () => {
      await send(entry, '/nope hello')

      // The translated sentence, not the error KEY: an entry that interpolated
      // the key into the wrong namespace would still "show an error".
      const alert = screen.getByRole('alert')
      expect(alert.textContent).toContain('Only supports the following query modes')
      expect(alert.textContent).toContain('bypass')

      // And it stops there. A rejected input that still reached the server would
      // run the query under the stored mode, silently ignoring what was typed.
      expect(requests).toHaveLength(0)
    })

    test(`${entry} sends the stripped query under the prefixed mode`, async () => {
      setQuerySettings({ mode: 'mix' })

      await send(entry, '/local hello there')

      expect(requests).toHaveLength(1)
      expect(requests[0].mode).toBe('local')
      expect(requests[0].query).toBe('hello there')

      // The transcript keeps what the user actually typed, prefix and all.
      expect(screen.queryAllByText('/local hello there')).toHaveLength(1)
      expect(screen.queryAllByText('hello there')).toHaveLength(0)
    })

    test(`${entry} falls back to the stored mode with no prefix`, async () => {
      setQuerySettings({ mode: 'naive' })

      await send(entry, 'plain question')

      expect(requests).toHaveLength(1)
      expect(requests[0].mode).toBe('naive')
      expect(requests[0].query).toBe('plain question')
    })
  }
})

describe('query entry settings snapshot', () => {
  test('the workspace entry clamps the debug-only switches and inherits the rest', async () => {
    // Both entries read the SAME stored settings; only the workspace overrides
    // anything. Values here are deliberately non-default so an entry that
    // rebuilt the snapshot from defaults would show up as a difference.
    setQuerySettings({
      mode: 'global',
      only_need_context: true,
      only_need_prompt: true,
      top_k: 7,
      chunk_top_k: 3,
      user_prompt: 'be terse',
      enable_rerank: false,
      history_turns: 0
    })

    await send('webui', 'question')
    const admin = requests[0]
    cleanup()

    requests.length = 0
    await send('workspace', 'question')
    const workspace = requests[0]

    const differing = (Object.keys(admin) as (keyof QueryRequest)[])
      .filter((key) => JSON.stringify(admin[key]) !== JSON.stringify(workspace[key]))
      .sort()

    // Exactly the clamped pair — not a subset, not a superset. Clamping one
    // more key would narrow an agreed contract, and inheriting one fewer would
    // leave the workspace chat with context instead of an answer.
    expect(differing).toEqual(['only_need_context', 'only_need_prompt'])
    expect(workspace.only_need_context).toBe(false)
    expect(workspace.only_need_prompt).toBe(false)
    expect(admin.only_need_context).toBe(true)
    expect(admin.only_need_prompt).toBe(true)
  })
})

/**
 * Source-level on purpose. An entry that reimplemented prefix parsing with the
 * same behaviour would pass every test above; what these guard is that neither
 * entry LEAVES the shared pipeline, which is how they drifted apart before.
 * Keep them about routing, not about formatting.
 */
const ENTRY_SOURCES = [
  ['webui', readFileSync(join(import.meta.dir, '..', 'RetrievalView.tsx'), 'utf8')],
  [
    'workspace',
    readFileSync(join(import.meta.dir, '..', 'workspace', 'WorkspaceQueryView.tsx'), 'utf8')
  ]
] as const

describe('query entry routing', () => {
  test('workspace derives its snapshot through the shared policy', () => {
    const workspace = ENTRY_SOURCES.find(([entry]) => entry === 'workspace')![1]

    expect(workspace).toMatch(
      /import\s*\{[^}]*\bworkspaceQuerySettingsSnapshot\b[^}]*\}\s*from/
    )
    expect(workspace).toMatch(/workspaceQuerySettingsSnapshot\(/)

    // Any override belongs in WORKSPACE_CLAMPED_SETTINGS, where a test pins
    // the size of the exemption list.
    expect(workspace).not.toMatch(/only_need_context:\s*false/)
    expect(workspace).not.toMatch(/only_need_prompt:\s*false/)
    expect(workspace).not.toMatch(/disable_user_prompt_prefix:/)
  })

  test.each(ENTRY_SOURCES)('%s delegates prefix parsing to the shared helper', (_entry, source) => {
    expect(source).toMatch(/import\s*\{[^}]*\bprepareQueryInput\b[^}]*\}\s*from/)
    expect(source).toMatch(/prepareQueryInput\(\s*input\s*,/)
  })

  test.each(ENTRY_SOURCES)('%s does not reimplement prefix parsing', (_entry, source) => {
    // A local `/mode ` regex is how the two entries drifted apart before.
    expect(source).not.toMatch(/\/\^\\\/|match\(\s*\/\^/)
    expect(source).not.toContain('allowedModes')
  })
})
