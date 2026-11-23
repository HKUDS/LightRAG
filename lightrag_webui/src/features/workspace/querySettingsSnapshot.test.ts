/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import {
  WORKSPACE_CLAMPED_SETTINGS,
  workspaceQuerySettingsSnapshot
} from './querySettingsSnapshot'
import { defaultQuerySettings } from '@/stores/querySettings'
import type { QuerySettings } from '@/stores/querySettings'

/**
 * The agreed contract: the workspace entry USES /webui's configuration.
 *
 * These tests are deliberately written over the whole settings object rather
 * than over a list of interesting fields, so a setting added to
 * `QuerySettings` is covered by the contract without anyone editing this file
 * — which is what "inherits everything" has to mean to be worth asserting.
 */

/** Every setting at a value distinguishable from its default. */
const NON_DEFAULT_SETTINGS: QuerySettings = {
  mode: 'local',
  top_k: 11,
  chunk_top_k: 7,
  max_entity_tokens: 1234,
  max_relation_tokens: 2345,
  max_total_tokens: 4567,
  only_need_context: true,
  only_need_prompt: true,
  stream: false,
  history_turns: 3,
  user_prompt: 'answer as a table',
  disable_user_prompt_prefix: true,
  enable_rerank: false
}

type ClampedKey = keyof typeof WORKSPACE_CLAMPED_SETTINGS

const settingKeys = Object.keys(defaultQuerySettings) as (keyof QuerySettings)[]
const clampedKeys = Object.keys(WORKSPACE_CLAMPED_SETTINGS) as ClampedKey[]
const inheritedKeys = settingKeys.filter(
  (key) => !(clampedKeys as (keyof QuerySettings)[]).includes(key)
)

describe('the NON_DEFAULT_SETTINGS fixture', () => {
  // Guards the two tests below from silently going blind. Without this, adding
  // a setting to `defaultQuerySettings` and forgetting it here would leave the
  // new field untested while every assertion still passed.
  test('covers every setting', () => {
    expect(Object.keys(NON_DEFAULT_SETTINGS).sort()).toEqual([...settingKeys].sort())
  })

  test('differs from the default for every setting', () => {
    // The keys that do NOT differ, so a failure names them.
    const sameAsDefault = settingKeys.filter(
      (key) => JSON.stringify(NON_DEFAULT_SETTINGS[key]) === JSON.stringify(defaultQuerySettings[key])
    )
    expect(sameAsDefault).toEqual([])
  })
})

describe('workspace query settings snapshot', () => {
  test('inherits every other setting verbatim from /webui', () => {
    const snapshot = workspaceQuerySettingsSnapshot(NON_DEFAULT_SETTINGS)
    // The keys that were NOT inherited, so a failure names them.
    const notInherited = inheritedKeys.filter(
      (key) => JSON.stringify(snapshot[key]) !== JSON.stringify(NON_DEFAULT_SETTINGS[key])
    )
    expect(notInherited).toEqual([])
  })

  test.each(clampedKeys)('overrides %s instead of inheriting it', (key) => {
    const snapshot = workspaceQuerySettingsSnapshot(NON_DEFAULT_SETTINGS)
    expect(snapshot[key]).toEqual(WORKSPACE_CLAMPED_SETTINGS[key])
    expect(snapshot[key]).not.toEqual(NON_DEFAULT_SETTINGS[key])
  })

  test('clamps only the debug outlets', () => {
    // Pins the SIZE of the exemption list. A new clamp narrows the contract,
    // so it should have to be added here deliberately.
    expect([...clampedKeys].sort()).toEqual(['only_need_context', 'only_need_prompt'])
  })

  test('does not mutate the stored settings', () => {
    const stored = { ...NON_DEFAULT_SETTINGS }
    workspaceQuerySettingsSnapshot(stored)
    expect(stored).toEqual(NON_DEFAULT_SETTINGS)
  })
})
