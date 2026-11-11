/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import {
  WORKSPACE_CLAMPED_SETTINGS,
  workspaceQuerySettingsSnapshot
} from './querySettingsSnapshot'
import { defaultQuerySettings } from '@/stores/querySettings'
import type { QuerySettings } from '@/stores/querySettings'
import { serializeQueryRequest } from '@/features/retrieval/serializeQueryRequest'

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

  test.each(settingKeys)('differs from the default for %s', (key) => {
    expect(NON_DEFAULT_SETTINGS[key]).not.toEqual(defaultQuerySettings[key])
  })
})

describe('workspace query settings snapshot', () => {
  test.each(inheritedKeys)('inherits %s verbatim from /webui', (key) => {
    const snapshot = workspaceQuerySettingsSnapshot(NON_DEFAULT_SETTINGS)
    expect(snapshot[key]).toEqual(NON_DEFAULT_SETTINGS[key])
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

describe('disable_user_prompt_prefix reaches the request body', () => {
  // The end of the chain: inheriting the field is only useful if the shared
  // serializer then forwards it, which is what the server reads.
  test.each([true, false])('admin set %s, workspace request carries it', (value) => {
    const snapshot = workspaceQuerySettingsSnapshot({
      ...defaultQuerySettings,
      disable_user_prompt_prefix: value
    })
    const body = serializeQueryRequest(snapshot, 'who is Tesla', [])
    expect(body.disable_user_prompt_prefix).toBe(value)
  })

  test('survives alongside a clamped debug outlet', () => {
    const snapshot = workspaceQuerySettingsSnapshot({
      ...defaultQuerySettings,
      only_need_context: true,
      disable_user_prompt_prefix: true
    })
    const body = serializeQueryRequest(snapshot, 'who is Tesla', [])
    expect(body.only_need_context).toBe(false)
    expect(body.disable_user_prompt_prefix).toBe(true)
  })

  test('a workspace user who never opened /webui gets the server prefix', () => {
    // The inheritance rides localStorage, so it is per-browser, not
    // per-account. An untouched browser applies the operator's prefix.
    const snapshot = workspaceQuerySettingsSnapshot({ ...defaultQuerySettings })
    const body = serializeQueryRequest(snapshot, 'who is Tesla', [])
    expect(body.disable_user_prompt_prefix).toBe(false)
  })
})
