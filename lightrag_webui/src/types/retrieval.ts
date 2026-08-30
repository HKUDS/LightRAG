/**
 * Retrieval domain types shared by the two query pages (`RetrievalView`,
 * `WorkspaceQueryView`), the shared query-session layer and `ChatMessage`.
 *
 * `MessageWithError` used to be exported by `ChatMessage.tsx`; it lives here
 * so the UI-free session controller does not have to depend on a message
 * rendering component. `ChatMessage` re-exports it for compatibility.
 */

import type { Message } from '@/api/lightrag'

export type MessageWithError = Message & {
  id: string // Unique identifier for stable React keys
  isError?: boolean
  isThinking?: boolean // Flag to indicate if the message is in a "thinking" state
  /**
   * Whether this message's text was written by the answering LLM. Decided by
   * the query session (see `lib/aiContentNotice.ts`), never re-derived from
   * the message's shape: a context-only debug response is not generated even
   * though it is a normal assistant bubble, and a stream that failed midway
   * IS generated even though it is an error bubble. Drives the optional
   * AI-generated content notice; persisted with the message so a restored
   * conversation keeps the distinction.
   */
  aiGenerated?: boolean
  isAborted?: boolean // Flag to indicate the user terminated this query (response may be incomplete)
  /**
   * Indicates if the mermaid diagram in this message has been rendered.
   * Used to persist the rendering state across updates and prevent flickering.
   */
  mermaidRendered?: boolean
  /**
   * Indicates if the LaTeX formulas in this message are complete and ready for rendering.
   * Used to prevent red error text during streaming of incomplete LaTeX formulas.
   */
  latexRendered?: boolean
}
