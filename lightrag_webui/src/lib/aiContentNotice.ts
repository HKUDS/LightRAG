import type { QuerySettings } from '@/stores/querySettings'
import type { MessageWithError } from '@/types/retrieval'

/**
 * Who may wear the "AI-generated content" label (ENABLE_AI_CONTENT_NOTICE).
 *
 * The label is a claim about where the text in a bubble came from, so it is
 * decided at query time — by the session controller, which knows what it
 * asked the server for — and NOT re-derived at render time from the message's
 * shape. Two cases make that necessary, and neither is visible to the
 * renderer:
 *
 * - `only_need_context` / `only_need_prompt` (the admin panel's debug
 *   switches; the workspace entry always sends them false) return the
 *   retrieved context or the constructed prompt *instead of* calling the
 *   answering LLM. Labelling those as AI-generated would be false.
 * - A stream that emits real answer chunks and then fails ends up as an
 *   `isError` bubble that still holds the generated text. Keying the label on
 *   `isError` would drop it exactly where a partial, degraded answer needs it
 *   most.
 */

/**
 * Whether a query with these settings produces model-written text at all.
 *
 * Both debug switches short-circuit the answering LLM server-side — EXCEPT in
 * `bypass` mode, which skips retrieval and sends the question straight to the
 * LLM without ever consulting them (`LightRAG.aquery_llm`). A bypass answer is
 * always model-written, and a streamed one carries no server verdict to
 * correct a wrong prediction, so the prediction has to get this case right on
 * its own.
 */
export function producesAiGeneratedOutput(
  settings: Partial<Pick<QuerySettings, 'mode' | 'only_need_context' | 'only_need_prompt'>>
): boolean {
  if (settings.mode === 'bypass') return true
  return !settings.only_need_context && !settings.only_need_prompt
}

/**
 * Reconcile the client's prediction with what the server reported.
 *
 * The server is authoritative: only it knows whether the answering LLM ran —
 * a query that finds no context gets the canned `PROMPTS["fail_response"]`
 * back without one, which no client can tell from the text. The prediction
 * above is the fallback for servers older than the `llm_generated` field.
 */
export function resolveAiGenerated(predicted: boolean, reported: unknown): boolean {
  return typeof reported === 'boolean' ? reported : predicted
}

/**
 * The flag for a message restored from the retrieval history.
 *
 * Messages written since the flag exists always carry an explicit boolean, so
 * this only fills in conversations persisted before it — where the origin of
 * the text is genuinely unknown and "an assistant bubble that is not an
 * error" is the best available reading (and what the notice showed when it
 * was first introduced).
 */
export function restoredAiGeneratedFlag(message: MessageWithError): boolean {
  if (typeof message.aiGenerated === 'boolean') return message.aiGenerated
  return message.role === 'assistant' && message.isError !== true
}

/**
 * Whether a rendered message shows the notice. `hasContent` is the renderer's
 * own "there is visible answer text" test — a bubble still waiting for its
 * first token has nothing to label yet.
 */
export function shouldShowAiContentNotice(params: {
  enabled: boolean
  role: MessageWithError['role']
  aiGenerated?: boolean
  hasContent: boolean
}): boolean {
  return (
    params.enabled &&
    params.role === 'assistant' &&
    params.hasContent &&
    params.aiGenerated === true
  )
}
