import type { MessageWithError } from '@/types/retrieval'

/**
 * A value that changes exactly when STREAMED CONTENT changes, and not when
 * the session merely re-stamps the elapsed time.
 *
 * `useQuerySession`'s response timer replaces the whole `messages` array
 * every 100 ms while a query is in flight, touching only `responseTime`.
 * Anything that follows the array identity therefore fires on every tick —
 * and anything that DEBOUNCES the array never fires at all, because a 150 ms
 * debounce restarted every 100 ms can never elapse. That is what left a long
 * streamed answer scrolling off-screen until the query finished. Only the
 * text lengths are read here, so a tick leaves this string untouched.
 */
export const streamedContentKey = (messages: MessageWithError[]): string => {
  const last = messages[messages.length - 1]
  if (!last) return '0'
  return [
    messages.length,
    last.content?.length ?? 0,
    last.displayContent?.length ?? 0,
    last.thinkingContent?.length ?? 0
  ].join(':')
}
