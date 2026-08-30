/**
 * The text to APPEND to an assistant message when a streaming query fails.
 *
 * The message updater is append-only (`assistantMessage.content += chunk`), so
 * whatever the stream already delivered is still in the bubble. Passing the
 * partial answer back in — as `content + '\n' + error` — therefore rendered
 * the whole answer twice before the error. Only the separator and the error
 * belong here.
 *
 * The separator is a blank line: a single newline leaves the error glued to
 * the answer's last sentence in the rendered markdown.
 */
export function streamErrorChunk(existingContent: string, error: string): string {
  if (!error) return ''
  return existingContent ? `\n\n${error}` : error
}
