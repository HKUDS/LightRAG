import { ReactNode, useCallback, useEffect, useRef, useState } from 'react'
import Button from '@/components/ui/Button'
import { ChatMessage } from '@/components/retrieval/ChatMessage'
import type { MessageWithError } from '@/types/retrieval'
import { ChevronDownIcon, CopyIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { copyToClipboard } from '@/utils/clipboard'
import { useDebounce } from '@/hooks/useDebounce'
import { streamedContentKey } from './streamedContentKey'

/**
 * Shared message list layer: iterates messages, renders `ChatMessage`, the
 * copy buttons, scroll-follow, jump-to-bottom and text-selection behavior.
 * Contains no query parameters, no welcome page and no page header — those
 * belong to the page composition layer.
 */

// Distance from the bottom (px) within which a user's downward scroll re-enables
// auto-follow. Wide enough to absorb streaming content growth between the user's
// release of the scrollbar and the scroll event being processed.
const NEAR_BOTTOM_PX = 100
// Upward scrollTop deltas at (or within) this distance from the bottom are treated
// as browser clamping after content shrinkage (e.g. a thinking block collapsing),
// not as user intent to scroll up.
const BOTTOM_CLAMP_EPSILON_PX = 2

export interface MessageListProps {
  messages: MessageWithError[]
  isLoading: boolean
  queryProgress: string | null
  /** Whether this page is currently visible (the admin page passes its tab
   * state; the workspace message area is ALWAYS active). */
  isActive?: boolean
  /** Rendered centered when there are no messages. */
  emptyState?: ReactNode
}

export default function MessageList({
  messages,
  isLoading,
  queryProgress,
  isActive = true,
  emptyState = null
}: MessageListProps) {
  const { t } = useTranslation()

  // Reference to track if we should follow scroll during streaming (ref for
  // synchronous updates); isFollowing mirrors it for rendering.
  const shouldFollowScrollRef = useRef(true)
  const [isFollowing, setIsFollowing] = useState(true)
  // Reference to track if user interaction is from the form area
  const isFormInteractionRef = useRef(false)
  // Direction baseline for the scroll handler: last observed scrollTop.
  const lastScrollTopRef = useRef(0)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)

  const setFollowScroll = useCallback((value: boolean) => {
    // Bail out early: the scroll handler runs un-throttled on every scroll event
    if (shouldFollowScrollRef.current === value) return
    shouldFollowScrollRef.current = value
    setIsFollowing(value)
  }, [])

  // Scroll to bottom. Single choke point for all programmatic scrolling.
  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      // Re-check at execution time: queued scrolls must not fire after the
      // user has detached from the bottom.
      if (!shouldFollowScrollRef.current) return
      const container = messagesContainerRef.current
      messagesEndRef.current?.scrollIntoView({ behavior: 'auto' })
      if (container) {
        // Absorb the programmatic displacement into the direction baseline so
        // the scroll handler's delta only ever reflects user movement.
        lastScrollTopRef.current = container.scrollTop
      }
    })
  }, [])

  // A newly submitted query (a user message appended at position length-2)
  // re-enables follow and forces a scroll.
  const previousLengthRef = useRef(messages.length)
  useEffect(() => {
    const previous = previousLengthRef.current
    previousLengthRef.current = messages.length
    if (
      messages.length > previous &&
      messages[messages.length - 2]?.role === 'user'
    ) {
      setFollowScroll(true)
      setTimeout(() => scrollToBottom(), 0)
    }
  }, [messages, setFollowScroll, scrollToBottom])

  // Add event listeners to detect when user manually interacts with the container.
  // User intent is inferred from the direction of user-initiated movement only:
  // scrolling up detaches from the bottom, scrolling down near the bottom re-attaches.
  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return

    const handleWheel = (e: WheelEvent) => {
      if (isFormInteractionRef.current) return
      // The 10px debounce only makes sense for pixel-mode deltas (trackpad
      // jitter). Line/page-mode events (e.g. Firefox wheel: deltaMode=1,
      // deltaY=±3) are discrete, deliberate steps — a raw pixel threshold
      // would swallow them; only ignore pure horizontal scrolling there.
      const insignificant =
        e.deltaMode === WheelEvent.DOM_DELTA_PIXEL
          ? Math.abs(e.deltaY) <= 10
          : e.deltaY === 0
      if (insignificant) return
      if (e.deltaY < 0) {
        // Scrolling up: the user wants to read back. Detach synchronously,
        // before the resulting scroll event, so streaming auto-scrolls stop
        // immediately.
        setFollowScroll(false)
      } else if (
        container.scrollHeight - container.scrollTop - container.clientHeight <
        NEAR_BOTTOM_PX
      ) {
        // Scrolling down at the bottom produces no scroll event (scrollTop
        // cannot move), so this wheel gesture is the only signal to re-attach.
        setFollowScroll(true)
      }
    }

    // Sample direction synchronously on every raw scroll event. Deferring this
    // (e.g. via throttle) would let a queued programmatic scroll overwrite the
    // user's upward displacement before it is ever observed.
    const handleScroll = () => {
      const scrollTop = container.scrollTop
      const delta = scrollTop - lastScrollTopRef.current
      lastScrollTopRef.current = scrollTop
      const distance = container.scrollHeight - scrollTop - container.clientHeight
      if (delta < 0) {
        // Upward user movement detaches — unless it is browser clamping after
        // the content shrank (still at the bottom), which carries no intent.
        if (distance > BOTTOM_CLAMP_EPSILON_PX && !isFormInteractionRef.current) {
          setFollowScroll(false)
        }
      } else if (delta > 0 && distance < NEAR_BOTTOM_PX) {
        // Downward user movement reaching the bottom region re-attaches.
        // Covers scrollbar drags, touch flicks, PageDown/End.
        setFollowScroll(true)
      }
      // delta === 0: leave the follow state untouched.
    }

    // Text selection starting inside the container detaches immediately, so
    // auto-scroll cannot destroy an in-progress selection during streaming.
    const handleSelectStart = () => setFollowScroll(false)

    container.addEventListener('wheel', handleWheel as EventListener)
    container.addEventListener('scroll', handleScroll as EventListener)
    container.addEventListener('selectstart', handleSelectStart)

    return () => {
      container.removeEventListener('wheel', handleWheel as EventListener)
      container.removeEventListener('scroll', handleScroll as EventListener)
      container.removeEventListener('selectstart', handleSelectStart)
    }
  }, [setFollowScroll])

  // Interacting with the page's form (the query composer) must not disable
  // auto-scroll; the composer is the page's single <form>.
  useEffect(() => {
    const form = document.querySelector('form')
    if (!form) return

    const handleFormMouseDown = () => {
      isFormInteractionRef.current = true
      setTimeout(() => {
        isFormInteractionRef.current = false
      }, 500) // Give enough time for the form interaction to complete
    }

    form.addEventListener('mousedown', handleFormMouseDown)
    return () => {
      form.removeEventListener('mousedown', handleFormMouseDown)
    }
  }, [])

  // Follow the answer as it streams. Keyed on the CONTENT, never on the
  // `messages` array: the response timer's 100 ms tick replaces that array
  // and would restart the debounce below forever (100 < 150), which is why
  // the debounce alone left a long answer scrolling off-screen until the
  // query finished. `scrollToBottom` re-checks follow state inside its own
  // rAF, so a queued scroll cannot fire after the user detaches.
  const contentKey = streamedContentKey(messages)
  useEffect(() => {
    if (shouldFollowScrollRef.current) {
      scrollToBottom()
    }
  }, [contentKey, scrollToBottom])

  // Settle-up pass: content whose height lands AFTER its commit (markdown,
  // KaTeX, a mermaid block finishing) is not covered by the effect above.
  // This one does follow the array, so the timer keeps it deferred while the
  // query runs and it fires once the ticks stop.
  const debouncedMessages = useDebounce(messages, 150)
  useEffect(() => {
    if (shouldFollowScrollRef.current) {
      scrollToBottom()
    }
  }, [debouncedMessages, scrollToBottom])

  // Handle copying message content with robust clipboard support
  const handleCopyMessage = useCallback(
    async (message: MessageWithError) => {
      const contentToCopy =
        message.role === 'user'
          ? message.content || ''
          : message.displayContent !== undefined
            ? message.displayContent
            : message.content || ''

      if (!contentToCopy.trim()) {
        toast.error(t('retrievePanel.chatMessage.copyEmpty', 'No content to copy'))
        return
      }

      try {
        const result = await copyToClipboard(contentToCopy)

        if (result.success) {
          const methodMessages: Record<string, string> = {
            'clipboard-api': t(
              'retrievePanel.chatMessage.copySuccess',
              'Content copied to clipboard'
            ),
            execCommand: t(
              'retrievePanel.chatMessage.copySuccessLegacy',
              'Content copied (legacy method)'
            ),
            'manual-select': t(
              'retrievePanel.chatMessage.copySuccessManual',
              'Content copied (manual method)'
            ),
            fallback: t(
              'retrievePanel.chatMessage.copySuccess',
              'Content copied to clipboard'
            )
          }

          toast.success(
            methodMessages[result.method] ||
              t('retrievePanel.chatMessage.copySuccess', 'Content copied to clipboard')
          )
        } else {
          if (result.method === 'fallback') {
            toast.error(
              result.error ||
                t('retrievePanel.chatMessage.copyFailed', 'Failed to copy content'),
              {
                description: t(
                  'retrievePanel.chatMessage.copyManualInstruction',
                  'Please select and copy the text manually'
                )
              }
            )
          } else {
            toast.error(
              t('retrievePanel.chatMessage.copyFailed', 'Failed to copy content'),
              {
                description: result.error
              }
            )
          }
        }
      } catch (err) {
        console.error('Clipboard operation failed:', err)
        toast.error(t('retrievePanel.chatMessage.copyError', 'Copy operation failed'), {
          description: err instanceof Error ? err.message : 'Unknown error occurred'
        })
      }
    },
    [t]
  )

  // Jump to the latest message and re-enable follow scroll.
  const handleJumpToBottom = useCallback(() => {
    setFollowScroll(true)
    scrollToBottom()
  }, [setFollowScroll, scrollToBottom])

  return (
    <div className="relative grow">
      <div
        ref={messagesContainerRef}
        // Vertical scrolling only. The list must NEVER scroll horizontally:
        // its children legitimately paint outside the content box (the copy
        // buttons' ::after touch targets extend 10px past the message row),
        // and wide answer content scrolls INSIDE its own block (code, tables)
        // rather than widening the conversation. Plain `overflow-auto` turned
        // both into a horizontal scrollbar on narrow screens — visible from
        // the very first short message.
        className="bg-primary-foreground/60 absolute inset-0 flex flex-col overflow-y-auto overflow-x-hidden rounded-lg border p-2"
      >
        <div className="flex min-h-0 flex-1 flex-col gap-2">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              {emptyState}
            </div>
          ) : (
            messages.map((message, idx) => {
              return (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} items-end gap-2`}
                >
                  {message.role === 'user' && (
                    <Button
                      onClick={() => handleCopyMessage(message)}
                      // 24px visual; ::after widens the TOUCH target to 44px
                      // (PRD ≥44px) without inflating the message layout.
                      className="relative mb-2 size-6 shrink-0 rounded-md opacity-60 transition-opacity after:absolute after:-inset-2.5 after:content-[''] hover:opacity-100"
                      tooltip={t('retrievePanel.chatMessage.copyTooltip')}
                      aria-label={t('retrievePanel.chatMessage.copyTooltip')}
                      variant="ghost"
                      size="icon"
                    >
                      <CopyIcon className="size-4" aria-hidden="true" />
                    </Button>
                  )}
                  <ChatMessage
                    message={message}
                    isTabActive={isActive}
                    activeProgress={
                      idx === messages.length - 1 && message.role === 'assistant'
                        ? queryProgress
                        : null
                    }
                    isQuerying={
                      idx === messages.length - 1 &&
                      message.role === 'assistant' &&
                      isLoading
                    }
                  />
                  {message.role === 'assistant' && (
                    <Button
                      onClick={() => handleCopyMessage(message)}
                      className="relative mb-2 size-6 shrink-0 rounded-md opacity-60 transition-opacity after:absolute after:-inset-2.5 after:content-[''] hover:opacity-100"
                      tooltip={t('retrievePanel.chatMessage.copyTooltip')}
                      aria-label={t('retrievePanel.chatMessage.copyTooltip')}
                      variant="ghost"
                      size="icon"
                    >
                      <CopyIcon className="size-4" aria-hidden="true" />
                    </Button>
                  )}
                </div>
              )
            })
          )}
          <div ref={messagesEndRef} className="pb-1" />
        </div>
      </div>
      {!isFollowing && messages.length > 0 && (
        <Button
          type="button"
          variant="outline"
          size="icon"
          aria-label={t('retrievePanel.retrieval.scrollToBottom')}
          tooltip={t('retrievePanel.retrieval.scrollToBottom')}
          // 32px visual; ::after widens the TOUCH target to 44px (PRD ≥44px).
          className="bg-background/70 absolute bottom-4 left-1/2 z-10 size-8 -translate-x-1/2 rounded-full opacity-70 shadow-md backdrop-blur transition-opacity after:absolute after:-inset-1.5 after:content-[''] hover:opacity-100"
          onClick={handleJumpToBottom}
        >
          <ChevronDownIcon className="size-4" aria-hidden="true" />
        </Button>
      )}
    </div>
  )
}
