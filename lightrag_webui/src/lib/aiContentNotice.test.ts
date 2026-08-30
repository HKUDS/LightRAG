import { describe, expect, test } from 'bun:test'

import {
  producesAiGeneratedOutput,
  restoredAiGeneratedFlag,
  shouldShowAiContentNotice
} from './aiContentNotice'
import type { MessageWithError } from '@/types/retrieval'

const assistant = (extra: Partial<MessageWithError> = {}): MessageWithError => ({
  id: 'm1',
  role: 'assistant',
  content: 'answer',
  ...extra
})

describe('producesAiGeneratedOutput', () => {
  test('a normal query produces model-written text', () => {
    expect(producesAiGeneratedOutput({})).toBe(true)
    expect(
      producesAiGeneratedOutput({ only_need_context: false, only_need_prompt: false })
    ).toBe(true)
  })

  test('the debug switches short-circuit the answering LLM', () => {
    // only_need_context returns the retrieved context and only_need_prompt the
    // constructed prompt, both BEFORE the answering call — labelling either as
    // AI-generated would be false.
    expect(producesAiGeneratedOutput({ only_need_context: true })).toBe(false)
    expect(producesAiGeneratedOutput({ only_need_prompt: true })).toBe(false)
  })
})

describe('restoredAiGeneratedFlag', () => {
  test('an explicit flag is preserved in both directions', () => {
    expect(restoredAiGeneratedFlag(assistant({ aiGenerated: true }))).toBe(true)
    // A context-only answer persisted with false must NOT be relabelled as
    // generated when the conversation is restored.
    expect(restoredAiGeneratedFlag(assistant({ aiGenerated: false }))).toBe(false)
  })

  test('a pre-flag conversation falls back to assistant-and-not-an-error', () => {
    expect(restoredAiGeneratedFlag(assistant())).toBe(true)
    expect(restoredAiGeneratedFlag(assistant({ isError: true }))).toBe(false)
    expect(restoredAiGeneratedFlag({ id: 'u1', role: 'user', content: 'hi' })).toBe(false)
  })
})

describe('shouldShowAiContentNotice', () => {
  const base = { enabled: true, role: 'assistant' as const, aiGenerated: true, hasContent: true }

  test('shows on an answer that carries model-written text', () => {
    expect(shouldShowAiContentNotice(base)).toBe(true)
  })

  test('a partial streamed answer that then failed keeps the notice', () => {
    // The bubble is an error bubble, but the text in it was written by the
    // model — that degraded answer is exactly where the label matters.
    expect(shouldShowAiContentNotice({ ...base, aiGenerated: true })).toBe(true)
  })

  test('never on the user, on unlabelled output, or before the first token', () => {
    expect(shouldShowAiContentNotice({ ...base, role: 'user' })).toBe(false)
    expect(shouldShowAiContentNotice({ ...base, aiGenerated: false })).toBe(false)
    expect(shouldShowAiContentNotice({ ...base, aiGenerated: undefined })).toBe(false)
    expect(shouldShowAiContentNotice({ ...base, hasContent: false })).toBe(false)
  })

  test('the deployment switch gates everything', () => {
    expect(shouldShowAiContentNotice({ ...base, enabled: false })).toBe(false)
  })
})
