import { useCallback, useEffect, useRef, useState } from 'react'
import { queryText, queryTextStream } from '@/api/lightrag'
import type { QueryMode } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useTranslation } from 'react-i18next'
import type { MessageWithError } from '@/types/retrieval'
import type { QuerySettings } from '@/stores/querySettings'
import type { RetrievalHistoryStore } from '@/stores/retrievalHistory'
import { serializeQueryRequest } from './serializeQueryRequest'

/**
 * Shared query-session controller used by BOTH query pages (`RetrievalView`
 * and `WorkspaceQueryView`): message state, streaming increments, COT/LaTeX
 * completeness tracking, progress, timing, stop and cleanup.
 *
 * Boundary rules (workspace-entry PRD §7.2):
 * - The history storage is INJECTED — this layer never decides which
 *   localStorage key a page's history lives in.
 * - The settings snapshot is BUILT AND PASSED by the page composition layer
 *   (per submit); this layer contains no entry detection, no clamping of
 *   `only_need_*`, no mode-prefix parsing, and never reads the admin tab
 *   state.
 * - Two pages create two independent instances: streaming/stop/timer state
 *   machines share the implementation, never the state.
 */

// Helper function to generate unique IDs with browser compatibility
const generateUniqueId = () => {
  // Use crypto.randomUUID() if available
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  // Fallback to timestamp + random string for browsers without crypto.randomUUID
  return `id-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

// LaTeX completeness detection function
const detectLatexCompleteness = (content: string): boolean => {
  // Check for unclosed block-level LaTeX formulas ($$...$$)
  const blockLatexMatches = content.match(/\$\$/g) || []
  const hasUnclosedBlock = blockLatexMatches.length % 2 !== 0

  // Check for unclosed inline LaTeX formulas ($...$, but not $$)
  // Remove all block formulas first to avoid interference
  const contentWithoutBlocks = content.replace(/\$\$[\s\S]*?\$\$/g, '')
  const inlineLatexMatches = contentWithoutBlocks.match(/(?<!\$)\$(?!\$)/g) || []
  const hasUnclosedInline = inlineLatexMatches.length % 2 !== 0

  // LaTeX is complete if there are no unclosed formulas
  return !hasUnclosedBlock && !hasUnclosedInline
}

// Robust COT parsing function to handle multiple think blocks and edge cases
const parseCOTContent = (content: string) => {
  const thinkStartTag = '<think>'
  const thinkEndTag = '</think>'

  // Find all <think> and </think> tag positions
  const startMatches: number[] = []
  const endMatches: number[] = []

  let startIndex = 0
  while ((startIndex = content.indexOf(thinkStartTag, startIndex)) !== -1) {
    startMatches.push(startIndex)
    startIndex += thinkStartTag.length
  }

  let endIndex = 0
  while ((endIndex = content.indexOf(thinkEndTag, endIndex)) !== -1) {
    endMatches.push(endIndex)
    endIndex += thinkEndTag.length
  }

  // Analyze COT state
  const hasThinkStart = startMatches.length > 0
  const hasThinkEnd = endMatches.length > 0
  const isThinking = hasThinkStart && startMatches.length > endMatches.length

  let thinkingContent = ''
  let displayContent = content

  if (hasThinkStart) {
    if (hasThinkEnd && startMatches.length === endMatches.length) {
      // Complete thinking blocks: extract the last complete thinking content
      const lastStartIndex = startMatches[startMatches.length - 1]
      const lastEndIndex = endMatches[endMatches.length - 1]

      if (lastEndIndex > lastStartIndex) {
        thinkingContent = content
          .substring(lastStartIndex + thinkStartTag.length, lastEndIndex)
          .trim()

        // Remove all thinking blocks, keep only the final display content
        displayContent = content.substring(lastEndIndex + thinkEndTag.length).trim()
      }
    } else if (isThinking) {
      // Currently thinking: extract current thinking content
      const lastStartIndex = startMatches[startMatches.length - 1]
      thinkingContent = content.substring(lastStartIndex + thinkStartTag.length)
      displayContent = ''
    }
  }

  return {
    isThinking,
    thinkingContent,
    displayContent,
    hasValidThinkBlock:
      hasThinkStart && hasThinkEnd && startMatches.length === endMatches.length
  }
}

export interface QuerySessionOptions {
  /** This page's own history storage — injected, never chosen here. */
  historyStore: RetrievalHistoryStore
  /**
   * Builds the settings snapshot for one submit. The page composition layer
   * owns any snapshot processing (workspace clamps `only_need_*` here).
   */
  getQuerySettingsSnapshot: () => QuerySettings
  /** Optional: called with a non-empty user_prompt on submit (admin page
   * records it into the prompt history; the workspace passes nothing). */
  onUserPromptUsed?: (prompt: string) => void
}

export interface SubmitOptions {
  /** Resolved by the page from its own input conventions (the admin page's
   * `/mode` prefix). Applied onto the settings snapshot before serializing. */
  modeOverride?: QueryMode
  /** What to show as the user's message (the admin page shows the original
   * input including its `/mode` prefix while sending the stripped query). */
  displayedInput?: string
}

export function useQuerySession({
  historyStore,
  getQuerySettingsSnapshot,
  onUserPromptUsed
}: QuerySessionOptions) {
  const { t } = useTranslation()

  const [messages, setMessages] = useState<MessageWithError[]>(() => {
    try {
      const history = historyStore.getState().history || []
      // Ensure each message from history has a unique ID and mermaidRendered status
      return history.map((msg, index) => {
        try {
          return {
            ...msg,
            id: msg.id || `hist-${Date.now()}-${index}`, // Add ID if missing
            mermaidRendered: msg.mermaidRendered ?? true, // Assume historical mermaid is rendered
            latexRendered: msg.latexRendered ?? true // Assume historical LaTeX is rendered
          }
        } catch (error) {
          console.error('Error processing message:', error)
          // Return a default message if there's an error
          return {
            role: 'system' as const,
            content: 'Error loading message',
            id: `error-${Date.now()}-${index}`,
            isError: true,
            mermaidRendered: true
          }
        }
      })
    } catch (error) {
      console.error('Error loading history:', error)
      return [] // Return an empty array if there's an error
    }
  })
  const [isLoading, setIsLoading] = useState(false)
  // Current retrieval pipeline step (e.g. "extracting_keywords") — shown to
  // the user while the query is in flight so they see live progress.
  const [queryProgress, setQueryProgress] = useState<string | null>(null)
  // Briefly disable the Stop button right after a query starts so a fast
  // double-click on Send (which morphs into Stop at the same position) can't
  // accidentally abort the query it just launched.
  const [stopDisabled, setStopDisabled] = useState(false)
  const stopCooldownTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Live response timer: ticks every 100ms while a query is in flight so the
  // user sees a running stopwatch.
  const responseTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const responseStartRef = useRef<number | null>(null)
  // Authoritative server-side duration returned by the backend.
  const serverResponseTimeRef = useRef<number | null>(null)
  // Whether time-to-first-token was already recorded for the current query.
  const firstTokenRecordedRef = useRef(false)
  // Whether the current query streams (TTFT is only meaningful then).
  const isStreamingRef = useRef(false)
  const thinkingStartTime = useRef<number | null>(null)
  const thinkingProcessed = useRef(false)
  // Abort controller for the in-flight query (streaming or non-streaming)
  const abortControllerRef = useRef<AbortController | null>(null)
  // Id of the assistant message currently receiving a response
  const activeAssistantIdRef = useRef<string | null>(null)

  // Unmount cleanup — abort in-flight work and drop timers.
  useEffect(() => {
    return () => {
      // Relinquish ownership before aborting so the request's deferred
      // `finally` cannot update state after this hook's owner unmounted.
      const controller = abortControllerRef.current
      abortControllerRef.current = null
      controller?.abort()

      if (responseTimerRef.current) {
        clearInterval(responseTimerRef.current)
        responseTimerRef.current = null
      }
      responseStartRef.current = null
      serverResponseTimeRef.current = null
      firstTokenRecordedRef.current = false
      isStreamingRef.current = false
      activeAssistantIdRef.current = null

      if (thinkingStartTime.current) {
        thinkingStartTime.current = null
      }
      if (stopCooldownTimerRef.current) {
        clearTimeout(stopCooldownTimerRef.current)
        stopCooldownTimerRef.current = null
      }
    }
  }, [])

  const submitQuery = useCallback(
    async (query: string, options: SubmitOptions = {}) => {
      if (!query.trim() || isLoading) return

      // Reset thinking timer state for new query to prevent confusion
      thinkingStartTime.current = null
      thinkingProcessed.current = false

      const userMessage: MessageWithError = {
        id: generateUniqueId(),
        content: options.displayedInput ?? query,
        role: 'user'
      }

      const assistantMessage: MessageWithError = {
        id: generateUniqueId(),
        content: '',
        role: 'assistant',
        mermaidRendered: false,
        latexRendered: false,
        thinkingTime: null,
        thinkingContent: undefined,
        displayContent: undefined,
        isThinking: false
      }

      const prevMessages = [...messages]

      // Create an abort controller so the user can terminate this query via
      // the Stop button. Track the active assistant message for handleStop.
      const controller = new AbortController()
      abortControllerRef.current = controller
      activeAssistantIdRef.current = assistantMessage.id

      setMessages([...prevMessages, userMessage, assistantMessage])
      setIsLoading(true)
      // Disable the Stop button for a short cooldown so a fast double-click
      // on Send doesn't immediately abort this query.
      setStopDisabled(true)
      if (stopCooldownTimerRef.current) clearTimeout(stopCooldownTimerRef.current)
      stopCooldownTimerRef.current = setTimeout(() => setStopDisabled(false), 500)

      // One consistent settings snapshot per submit. The page composition
      // layer owns snapshot processing; the mode override resolved by the
      // page's input conventions is applied here onto the snapshot, so the
      // serializer sees the final settings.
      const baseSnapshot = getQuerySettingsSnapshot()
      const snapshot: QuerySettings = options.modeOverride
        ? { ...baseSnapshot, mode: options.modeOverride }
        : baseSnapshot

      // Start the live response timer.
      responseStartRef.current = Date.now()
      serverResponseTimeRef.current = null
      firstTokenRecordedRef.current = false
      isStreamingRef.current = snapshot.stream ?? false
      assistantMessage.responseTime = 0
      assistantMessage.firstTokenTime = null
      if (responseTimerRef.current) clearInterval(responseTimerRef.current)
      responseTimerRef.current = setInterval(() => {
        const elapsed = (Date.now() - (responseStartRef.current ?? Date.now())) / 1000
        const rounded = parseFloat(elapsed.toFixed(1))
        assistantMessage.responseTime = rounded
        setMessages((prev) => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage && lastMessage.id === assistantMessage.id) {
            lastMessage.responseTime = rounded
          }
          return newMessages
        })
      }, 100)

      // Create a function to update the assistant's message
      const updateAssistantMessage = (chunk: string, isError?: boolean) => {
        // Record time-to-first-token on the very first content chunk.
        if (
          !isError &&
          !firstTokenRecordedRef.current &&
          isStreamingRef.current &&
          responseStartRef.current &&
          chunk
        ) {
          firstTokenRecordedRef.current = true
          const ttft = (Date.now() - responseStartRef.current) / 1000
          assistantMessage.firstTokenTime = parseFloat(ttft.toFixed(1))
        }
        assistantMessage.content += chunk

        // Start thinking timer on first sight of think tag
        if (assistantMessage.content.includes('<think>') && !thinkingStartTime.current) {
          thinkingStartTime.current = Date.now()
        }

        // Use the robust COT parsing function
        const cotResult = parseCOTContent(assistantMessage.content)

        // Update thinking state
        assistantMessage.isThinking = cotResult.isThinking

        // Only calculate time and extract thinking content once when thinking is complete
        if (cotResult.hasValidThinkBlock && !thinkingProcessed.current) {
          if (thinkingStartTime.current && !assistantMessage.thinkingTime) {
            const duration = (Date.now() - thinkingStartTime.current) / 1000
            assistantMessage.thinkingTime = parseFloat(duration.toFixed(2))
          }
          thinkingProcessed.current = true
        }

        // Update content based on parsing results
        assistantMessage.thinkingContent = cotResult.thinkingContent
        // Only fallback to full content if not in a thinking state.
        if (cotResult.isThinking) {
          assistantMessage.displayContent = ''
        } else {
          assistantMessage.displayContent =
            cotResult.displayContent || assistantMessage.content
        }

        // Detect if the assistant message contains a complete mermaid code block
        // Simple heuristic: look for ```mermaid ... ```
        const mermaidBlockRegex = /```mermaid\s+([\s\S]+?)```/g
        let mermaidRendered = false
        let match
        while ((match = mermaidBlockRegex.exec(assistantMessage.content)) !== null) {
          // If the block is not too short, consider it complete
          if (match[1] && match[1].trim().length > 10) {
            mermaidRendered = true
            break
          }
        }
        assistantMessage.mermaidRendered = mermaidRendered

        // Detect if the assistant message contains complete LaTeX formulas
        assistantMessage.latexRendered = detectLatexCompleteness(assistantMessage.content)

        // Single unified update to avoid race conditions
        setMessages((prev) => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage && lastMessage.id === assistantMessage.id) {
            // Update all properties at once to maintain consistency
            Object.assign(lastMessage, {
              content: assistantMessage.content,
              thinkingContent: assistantMessage.thinkingContent,
              displayContent: assistantMessage.displayContent,
              isThinking: assistantMessage.isThinking,
              isError: isError,
              mermaidRendered: assistantMessage.mermaidRendered,
              latexRendered: assistantMessage.latexRendered,
              thinkingTime: assistantMessage.thinkingTime,
              responseTime: assistantMessage.responseTime,
              firstTokenTime: assistantMessage.firstTokenTime
            })
          }
          return newMessages
        })
      }

      // Record a non-empty user prompt (page-injected behavior).
      if (snapshot.user_prompt && snapshot.user_prompt.trim()) {
        onUserPromptUsed?.(snapshot.user_prompt.trim())
      }

      // The shared serializer: same snapshot + same history + same question
      // ⇒ identical request body, on both pages.
      const queryParams = serializeQueryRequest(snapshot, query, prevMessages)

      try {
        // Run query
        if (snapshot.stream) {
          let streamError = ''
          await queryTextStream(
            queryParams,
            updateAssistantMessage,
            (error) => {
              streamError += error
            },
            controller.signal,
            // Capture the authoritative server-side duration (emitted as the
            // final NDJSON metadata line).
            (seconds) => {
              serverResponseTimeRef.current = seconds
            },
            (event) => {
              setQueryProgress(event)
            }
          )
          if (streamError) {
            if (assistantMessage.content) {
              streamError = assistantMessage.content + '\n' + streamError
            }
            updateAssistantMessage(streamError, true)
          }
        } else {
          const response = await queryText(queryParams, controller.signal)
          if (typeof response.response_time === 'number') {
            serverResponseTimeRef.current = response.response_time
          }
          updateAssistantMessage(response.response)
        }
      } catch (err) {
        // If the user terminated the query, handleStop already finalized the
        // message state; don't render it as an error.
        if (!controller.signal.aborted) {
          updateAssistantMessage(
            `${t('retrievePanel.retrieval.error')}\n${errorMessage(err)}`,
            true
          )
        }
      } finally {
        // Only the owning, non-terminated query runs global cleanup — see
        // handleStop for the ownership contract.
        if (abortControllerRef.current === controller) {
          setIsLoading(false)
          setQueryProgress(null)
          abortControllerRef.current = null

          // Stop the live response timer and stamp the final duration.
          if (responseTimerRef.current) {
            clearInterval(responseTimerRef.current)
            responseTimerRef.current = null
          }
          const authoritativeTime = serverResponseTimeRef.current
          if (authoritativeTime !== null) {
            assistantMessage.responseTime = authoritativeTime
          } else if (responseStartRef.current) {
            const finalElapsed = (Date.now() - responseStartRef.current) / 1000
            assistantMessage.responseTime = parseFloat(finalElapsed.toFixed(1))
          }
          responseStartRef.current = null
          serverResponseTimeRef.current = null
          firstTokenRecordedRef.current = false
          // Sync the finalized time into the rendered message
          setMessages((prev) => {
            const newMessages = [...prev]
            const lastMessage = newMessages[newMessages.length - 1]
            if (lastMessage && lastMessage.id === assistantMessage.id) {
              lastMessage.responseTime = assistantMessage.responseTime
            }
            return newMessages
          })

          // Enhanced cleanup with error handling to prevent memory leaks
          try {
            // Final COT state validation and cleanup
            const finalCotResult = parseCOTContent(assistantMessage.content)

            // Force set final state - stream ended so thinking must be false
            assistantMessage.isThinking = false

            // If we have a complete thinking block but time wasn't calculated, finalize
            if (
              finalCotResult.hasValidThinkBlock &&
              thinkingStartTime.current &&
              !assistantMessage.thinkingTime
            ) {
              const duration = (Date.now() - thinkingStartTime.current) / 1000
              assistantMessage.thinkingTime = parseFloat(duration.toFixed(2))
            }

            // Ensure display content is correctly set based on final parsing
            if (finalCotResult.displayContent !== undefined) {
              assistantMessage.displayContent = finalCotResult.displayContent
            }
          } catch (error) {
            console.error('Error in final COT state validation:', error)
            // Force reset state on error
            assistantMessage.isThinking = false
          } finally {
            // Ensure cleanup happens regardless of errors
            thinkingStartTime.current = null
          }

          // Save history into THIS page's injected storage
          try {
            historyStore
              .getState()
              .setHistory([...prevMessages, userMessage, assistantMessage])
          } catch (error) {
            console.error('Error saving retrieval history:', error)
          }
        }
      }
    },
    [isLoading, messages, historyStore, getQuerySettingsSnapshot, onUserPromptUsed, t]
  )

  const clearMessages = useCallback(() => {
    // Stop any running response timer so it doesn't keep ticking after clear.
    if (responseTimerRef.current) {
      clearInterval(responseTimerRef.current)
      responseTimerRef.current = null
    }
    responseStartRef.current = null
    serverResponseTimeRef.current = null
    firstTokenRecordedRef.current = false
    setQueryProgress(null)
    setMessages([])
    historyStore.getState().setHistory([])
  }, [historyStore])

  // Stop the in-flight query. Frees the UI immediately so the user can start
  // a new query without waiting for the aborted request to unwind.
  const handleStop = useCallback(() => {
    const controller = abortControllerRef.current
    if (!controller) return
    controller.abort()
    // Relinquish ownership so the aborted query's deferred `finally` skips
    // its cleanup — otherwise it would write the stale conversation back into
    // history, undoing a Clear the user performs after stopping.
     
    abortControllerRef.current = null

    // Finalize the terminated assistant message and persist immediately so
    // the terminated state is the authoritative saved history.
    const activeId = activeAssistantIdRef.current
    let stoppedResponseTime: number | null = null
    if (responseStartRef.current) {
      stoppedResponseTime = parseFloat(
        ((Date.now() - responseStartRef.current) / 1000).toFixed(1)
      )
    }
    const finalizedMessages = messages.map((m) => {
      if (m.id !== activeId) return m
      // Terminated mid-thinking: finalize the COT block so the partial
      // reasoning stays visible instead of vanishing once isThinking clears.
      let thinkingTime = m.thinkingTime ?? null
      if (m.isThinking && thinkingTime === null && thinkingStartTime.current) {
        thinkingTime = parseFloat(
          ((Date.now() - thinkingStartTime.current) / 1000).toFixed(2)
        )
      }
      return {
        ...m,
        isThinking: false,
        isAborted: true,
        thinkingTime,
        responseTime: stoppedResponseTime ?? m.responseTime
      }
    })
    setMessages(finalizedMessages)
    try {
      historyStore.getState().setHistory(finalizedMessages)
    } catch (error) {
      console.error('Error saving retrieval history:', error)
    }

    // The skipped `finally` won't reset these shared thinking refs.
     
    thinkingStartTime.current = null
     
    thinkingProcessed.current = false

    // Stop the live response timer (the skipped finally won't clear it).
    if (responseTimerRef.current) {
      clearInterval(responseTimerRef.current)
      responseTimerRef.current = null
    }
    responseStartRef.current = null
    serverResponseTimeRef.current = null
    firstTokenRecordedRef.current = false

    setIsLoading(false)
    setQueryProgress(null)
    // Cancel any pending Stop-button cooldown and reset it for the next query.
    if (stopCooldownTimerRef.current) {
      clearTimeout(stopCooldownTimerRef.current)
      stopCooldownTimerRef.current = null
    }
    setStopDisabled(false)
  }, [messages, historyStore])

  return {
    messages,
    isLoading,
    queryProgress,
    stopDisabled,
    submitQuery,
    clearMessages,
    handleStop
  }
}
