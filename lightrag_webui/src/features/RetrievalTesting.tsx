import { CopyIcon, EraserIcon, SendIcon } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import type { CitationsMetadata, QueryMode } from '@/api/lightrag'
import { queryText, queryTextStream } from '@/api/lightrag'
import { ChatMessage, type MessageWithError } from '@/components/retrieval/ChatMessage'
import QuerySettings from '@/components/retrieval/QuerySettings'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import { useDebounce } from '@/hooks/useDebounce'
import { errorMessage, throttle } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { copyToClipboard } from '@/utils/clipboard'

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

  let startIndex = content.indexOf(thinkStartTag)
  while (startIndex !== -1) {
    startMatches.push(startIndex)
    startIndex = content.indexOf(thinkStartTag, startIndex + thinkStartTag.length)
  }

  let endIndex = content.indexOf(thinkEndTag)
  while (endIndex !== -1) {
    endMatches.push(endIndex)
    endIndex = content.indexOf(thinkEndTag, endIndex + thinkEndTag.length)
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
    hasValidThinkBlock: hasThinkStart && hasThinkEnd && startMatches.length === endMatches.length,
  }
}

export default function RetrievalTesting() {
  const { t } = useTranslation()
  // Get current tab to determine if this tab is active (for performance optimization)
  const currentTab = useSettingsStore.use.currentTab()
  const isRetrievalTabActive = currentTab === 'retrieval'

  const [messages, setMessages] = useState<MessageWithError[]>(() => {
    try {
      const history = useSettingsStore.getState().retrievalHistory || []
      // Ensure each message from history has a unique ID and mermaidRendered status
      return history.map((msg, index) => {
        try {
          const msgWithError = msg as MessageWithError // Cast to access potential properties
          return {
            ...msg,
            id: msgWithError.id || `hist-${Date.now()}-${index}`, // Add ID if missing
            mermaidRendered: msgWithError.mermaidRendered ?? true, // Assume historical mermaid is rendered
            latexRendered: msgWithError.latexRendered ?? true, // Assume historical LaTeX is rendered
          }
        } catch (error) {
          console.error('Error processing message:', error)
          // Return a default message if there's an error
          return {
            role: 'system',
            content: 'Error loading message',
            id: `error-${Date.now()}-${index}`,
            isError: true,
            mermaidRendered: true,
          }
        }
      })
    } catch (error) {
      console.error('Error loading history:', error)
      return [] // Return an empty array if there's an error
    }
  })
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [inputError, setInputError] = useState('') // Error message for input
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null)

  // Smart switching logic: use Input for single line, Textarea for multi-line
  const hasMultipleLines = inputValue.includes('\n')

  // Enhanced event handlers for smart switching
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setInputValue(e.target.value)
      if (inputError) setInputError('')
    },
    [inputError]
  )

  // Unified height adjustment function for textarea
  const adjustTextareaHeight = useCallback((element: HTMLTextAreaElement) => {
    requestAnimationFrame(() => {
      element.style.height = 'auto'
      element.style.height = `${Math.min(element.scrollHeight, 120)}px`
    })
  }, [])

  // Scroll to bottom function - restored smooth scrolling with better handling
  const scrollToBottom = useCallback(() => {
    // Set flag to indicate this is a programmatic scroll
    programmaticScrollRef.current = true
    // Use requestAnimationFrame for better performance
    requestAnimationFrame(() => {
      if (messagesEndRef.current) {
        // Use smooth scrolling for better user experience
        messagesEndRef.current.scrollIntoView({ behavior: 'auto' })
      }
    })
  }, [])

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      if (!inputValue.trim() || isLoading) return

      // Parse query mode prefix
      const allowedModes: QueryMode[] = ['naive', 'local', 'global', 'hybrid', 'mix', 'bypass']
      const prefixMatch = inputValue.match(/^\/(\w+)\s+([\s\S]+)/)
      let modeOverride: QueryMode | undefined
      let actualQuery = inputValue

      // If input starts with a slash, but does not match the valid prefix pattern, treat as error
      if (/^\/\S+/.test(inputValue) && !prefixMatch) {
        setInputError(t('retrievePanel.retrieval.queryModePrefixInvalid'))
        return
      }

      if (prefixMatch) {
        const mode = prefixMatch[1] as QueryMode
        const query = prefixMatch[2]
        if (!allowedModes.includes(mode)) {
          setInputError(
            t('retrievePanel.retrieval.queryModeError', {
              modes: 'naive, local, global, hybrid, mix, bypass',
            })
          )
          return
        }
        modeOverride = mode
        actualQuery = query
      }

      // Clear error message
      setInputError('')

      // Reset thinking timer state for new query to prevent confusion
      thinkingStartTime.current = null
      thinkingProcessed.current = false

      // Create messages
      // Save the original input (with prefix if any) in userMessage.content for display
      const userMessage: MessageWithError = {
        id: generateUniqueId(), // Use browser-compatible ID generation
        content: inputValue,
        role: 'user',
      }

      const assistantMessage: MessageWithError = {
        id: generateUniqueId(), // Use browser-compatible ID generation
        content: '',
        role: 'assistant',
        mermaidRendered: false,
        latexRendered: false, // Explicitly initialize to false
        thinkingTime: null, // Explicitly initialize to null
        thinkingContent: undefined, // Explicitly initialize to undefined
        displayContent: undefined, // Explicitly initialize to undefined
        isThinking: false, // Explicitly initialize to false
        citationsProcessed: false, // Prevent finally block from overwriting citation content
      }

      const prevMessages = [...messages]

      // Add messages to chatbox
      setMessages([...prevMessages, userMessage, assistantMessage])

      // Reset scroll following state for new query
      shouldFollowScrollRef.current = true
      // Set flag to indicate we're receiving a response
      isReceivingResponseRef.current = true

      // Force scroll to bottom after messages are rendered
      setTimeout(() => {
        scrollToBottom()
      }, 0)

      // Clear input and set loading
      setInputValue('')
      setIsLoading(true)

      // Reset input height to minimum after clearing input
      if (inputRef.current) {
        if ('style' in inputRef.current) {
          inputRef.current.style.height = '40px'
        }
      }

      // Create a function to update the assistant's message
      const updateAssistantMessage = (chunk: string, isError?: boolean) => {
        assistantMessage.content += chunk

        // Start thinking timer on first sight of think tag
        if (assistantMessage.content.includes('<think>') && !thinkingStartTime.current) {
          thinkingStartTime.current = Date.now()
        }

        // Use the new robust COT parsing function
        const cotResult = parseCOTContent(assistantMessage.content)

        // Update thinking state
        assistantMessage.isThinking = cotResult.isThinking

        // Only calculate time and extract thinking content once when thinking is complete
        if (cotResult.hasValidThinkBlock && !thinkingProcessed.current) {
          if (thinkingStartTime.current && !assistantMessage.thinkingTime) {
            const duration = (Date.now() - thinkingStartTime.current) / 1000
            assistantMessage.thinkingTime = Number.parseFloat(duration.toFixed(2))
          }
          thinkingProcessed.current = true
        }

        // Update content based on parsing results
        assistantMessage.thinkingContent = cotResult.thinkingContent
        // Only fallback to full content if not in a thinking state.
        if (cotResult.isThinking) {
          assistantMessage.displayContent = ''
        } else {
          assistantMessage.displayContent = cotResult.displayContent || assistantMessage.content
        }

        // Detect if the assistant message contains a complete mermaid code block
        // Simple heuristic: look for ```mermaid ... ```
        const mermaidBlockRegex = /```mermaid\s+([\s\S]+?)```/g
        let mermaidRendered = false
        let match: RegExpExecArray | null = mermaidBlockRegex.exec(assistantMessage.content)
        while (match !== null) {
          // If the block is not too short, consider it complete
          if (match[1] && match[1].trim().length > 10) {
            mermaidRendered = true
            break
          }
          match = mermaidBlockRegex.exec(assistantMessage.content)
        }
        assistantMessage.mermaidRendered = mermaidRendered

        // Detect if the assistant message contains complete LaTeX formulas
        const latexRendered = detectLatexCompleteness(assistantMessage.content)
        assistantMessage.latexRendered = latexRendered

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
            })
          }
          return newMessages
        })

        // After updating content, scroll to bottom if auto-scroll is enabled
        // Use a longer delay to ensure DOM has updated
        if (shouldFollowScrollRef.current) {
          setTimeout(() => {
            scrollToBottom()
          }, 30)
        }
      }

      // Prepare query parameters
      const state = useSettingsStore.getState()

      // Add user prompt to history if it exists and is not empty
      if (state.querySettings.user_prompt?.trim()) {
        state.addUserPromptToHistory(state.querySettings.user_prompt.trim())
      }

      // Determine the effective mode
      const effectiveMode = modeOverride || state.querySettings.mode

      // Determine effective history turns with bypass override
      const configuredHistoryTurns = state.querySettings.history_turns || 0
      const effectiveHistoryTurns =
        effectiveMode === 'bypass' && configuredHistoryTurns === 0 ? 3 : configuredHistoryTurns

      const queryParams = {
        ...state.querySettings,
        query: actualQuery,
        response_type: 'Multiple Paragraphs',
        conversation_history:
          effectiveHistoryTurns > 0
            ? prevMessages
                .filter((m) => m.isError !== true)
                .slice(-effectiveHistoryTurns * 2)
                .map((m) => ({ role: m.role, content: m.content }))
            : [],
        ...(modeOverride ? { mode: modeOverride } : {}),
      }

      try {
        // Run query
        if (state.querySettings.stream) {
          let errorMessage = ''
          await queryTextStream(
            queryParams,
            updateAssistantMessage,
            (error) => {
              errorMessage += error
            },
            // Citation callback - use position markers to insert citations client-side
            // NEW: No longer receives annotated_response (which duplicated payload)
            // Instead receives position metadata for client-side marker insertion
            (() => {
              let citationsApplied = false
              return (metadata: CitationsMetadata) => {
                // Guard against multiple invocations
                if (citationsApplied || !metadata.markers || metadata.markers.length === 0) return
                citationsApplied = true

                // Insert markers into the accumulated response using position data
                // Sort by position descending so we can insert from end to start (preserves positions)
                const sortedMarkers = [...metadata.markers].sort(
                  (a, b) => b.insert_position - a.insert_position
                )

                let annotatedContent = assistantMessage.content
                for (const marker of sortedMarkers) {
                  // Insert marker at the specified position
                  if (marker.insert_position <= annotatedContent.length) {
                    annotatedContent =
                      annotatedContent.slice(0, marker.insert_position) +
                      marker.marker +
                      annotatedContent.slice(marker.insert_position)
                  }
                }

                // Append footnotes if provided
                let finalContent = annotatedContent
                if (metadata.footnotes && metadata.footnotes.length > 0) {
                  finalContent += `\n\n---\n\n**References:**\n${metadata.footnotes.join('\n')}`
                }

                // Update message with annotated content and store citation metadata for HoverCards
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessage.id
                      ? {
                          ...msg,
                          content: finalContent,
                          displayContent: finalContent,
                          citationsProcessed: true,
                          citationsMetadata: metadata, // Store for HoverCard rendering
                        }
                      : msg
                  )
                )
                // Also update the local reference for final cleanup operations
                assistantMessage.content = finalContent
                assistantMessage.displayContent = finalContent
                assistantMessage.citationsProcessed = true
              }
            })()
          )
          if (errorMessage) {
            if (assistantMessage.content) {
              errorMessage = `${assistantMessage.content}\n${errorMessage}`
            }
            updateAssistantMessage(errorMessage, true)
          }
        } else {
          const response = await queryText(queryParams)
          updateAssistantMessage(response.response)
        }
      } catch (err) {
        // Handle error
        updateAssistantMessage(`${t('retrievePanel.retrieval.error')}\n${errorMessage(err)}`, true)
      } finally {
        // Clear loading and add messages to state
        setIsLoading(false)
        isReceivingResponseRef.current = false

        // Enhanced cleanup with error handling to prevent memory leaks
        try {
          // Final COT state validation and cleanup
          const finalCotResult = parseCOTContent(assistantMessage.content)

          // Force set final state - stream ended so thinking must be false
          assistantMessage.isThinking = false

          // If we have a complete thinking block but time wasn't calculated, do final calculation
          if (
            finalCotResult.hasValidThinkBlock &&
            thinkingStartTime.current &&
            !assistantMessage.thinkingTime
          ) {
            const duration = (Date.now() - thinkingStartTime.current) / 1000
            assistantMessage.thinkingTime = Number.parseFloat(duration.toFixed(2))
          }

          // Ensure display content is correctly set based on final parsing
          // BUT skip if citations were processed (they already set displayContent)
          if (!assistantMessage.citationsProcessed && finalCotResult.displayContent !== undefined) {
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

        // Save history with error handling
        try {
          useSettingsStore
            .getState()
            .setRetrievalHistory([...prevMessages, userMessage, assistantMessage])
        } catch (error) {
          console.error('Error saving retrieval history:', error)
        }
      }
    },
    [inputValue, isLoading, messages, t, scrollToBottom]
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && e.shiftKey) {
        // Shift+Enter: Insert newline
        e.preventDefault()
        const target = e.target as HTMLInputElement | HTMLTextAreaElement
        const start = target.selectionStart || 0
        const end = target.selectionEnd || 0
        const newValue = `${inputValue.slice(0, start)}\n${inputValue.slice(end)}`
        setInputValue(newValue)

        // Set cursor position after the newline and adjust height if needed
        setTimeout(() => {
          if (target.setSelectionRange) {
            target.setSelectionRange(start + 1, start + 1)
          }

          // Manually trigger height adjustment for textarea after component switch
          if (inputRef.current && inputRef.current.tagName === 'TEXTAREA') {
            adjustTextareaHeight(inputRef.current as HTMLTextAreaElement)
          }
        }, 0)
      } else if (e.key === 'Enter' && !e.shiftKey) {
        // Enter: Submit form
        e.preventDefault()
        handleSubmit(e as unknown as React.FormEvent)
      }
    },
    [inputValue, handleSubmit, adjustTextareaHeight]
  )

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      // Get pasted text content
      const pastedText = e.clipboardData.getData('text')

      // Check if it contains newlines
      if (pastedText.includes('\n')) {
        e.preventDefault() // Prevent default paste behavior

        // Get current cursor position
        const target = e.target as HTMLInputElement | HTMLTextAreaElement
        const start = target.selectionStart || 0
        const end = target.selectionEnd || 0

        // Build new value
        const newValue = inputValue.slice(0, start) + pastedText + inputValue.slice(end)

        // Update state (this will trigger component switch to Textarea)
        setInputValue(newValue)

        // Set cursor position to end of pasted content
        setTimeout(() => {
          if (inputRef.current?.setSelectionRange) {
            const newCursorPosition = start + pastedText.length
            inputRef.current.setSelectionRange(newCursorPosition, newCursorPosition)
          }
        }, 0)
      }
      // If no newlines, let default paste behavior continue
    },
    [inputValue]
  )

  // Effect to handle component switching and maintain focus
  useEffect(() => {
    if (inputRef.current) {
      // When component type changes, restore focus and cursor position
      const currentElement = inputRef.current
      const cursorPosition = currentElement.selectionStart || inputValue.length

      // Use requestAnimationFrame to ensure DOM update is complete
      requestAnimationFrame(() => {
        currentElement.focus()
        if (currentElement.setSelectionRange) {
          currentElement.setSelectionRange(cursorPosition, cursorPosition)
        }
      })
    }
  }, [inputValue.length]) // Include inputValue.length dependency

  // Effect to adjust textarea height when switching to multi-line mode
  useEffect(() => {
    if (hasMultipleLines && inputRef.current && inputRef.current.tagName === 'TEXTAREA') {
      adjustTextareaHeight(inputRef.current as HTMLTextAreaElement)
    }
  }, [hasMultipleLines, adjustTextareaHeight])

  // Reference to track if we should follow scroll during streaming (using ref for synchronous updates)
  const shouldFollowScrollRef = useRef(true)
  const thinkingStartTime = useRef<number | null>(null)
  const thinkingProcessed = useRef(false)
  // Reference to track if user interaction is from the form area
  const isFormInteractionRef = useRef(false)
  // Reference to track if scroll was triggered programmatically
  const programmaticScrollRef = useRef(false)
  // Reference to track if we're currently receiving a streaming response
  const isReceivingResponseRef = useRef(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)

  // Add cleanup effect for memory leak prevention
  useEffect(() => {
    // Component cleanup - reset timer state to prevent memory leaks
    return () => {
      if (thinkingStartTime.current) {
        thinkingStartTime.current = null
      }
    }
  }, [])

  // Add event listeners to detect when user manually interacts with the container
  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return

    // Handle significant mouse wheel events - only disable auto-scroll for deliberate scrolling
    const handleWheel = (e: WheelEvent) => {
      // Only consider significant wheel movements (more than 10px)
      if (Math.abs(e.deltaY) > 10 && !isFormInteractionRef.current) {
        shouldFollowScrollRef.current = false
      }
    }

    // Handle scroll events - only disable auto-scroll if not programmatically triggered
    // and if it's a significant scroll
    const handleScroll = throttle(() => {
      // If this is a programmatic scroll, don't disable auto-scroll
      if (programmaticScrollRef.current) {
        programmaticScrollRef.current = false
        return
      }

      // Check if scrolled to bottom or very close to bottom
      const container = messagesContainerRef.current
      if (container) {
        const isAtBottom =
          container.scrollHeight - container.scrollTop - container.clientHeight < 20

        // If at bottom, enable auto-scroll, otherwise disable it
        if (isAtBottom) {
          shouldFollowScrollRef.current = true
        } else if (!isFormInteractionRef.current && !isReceivingResponseRef.current) {
          shouldFollowScrollRef.current = false
        }
      }
    }, 30)

    // Add event listeners - only listen for wheel and scroll events
    container.addEventListener('wheel', handleWheel as EventListener)
    container.addEventListener('scroll', handleScroll as EventListener)

    return () => {
      container.removeEventListener('wheel', handleWheel as EventListener)
      container.removeEventListener('scroll', handleScroll as EventListener)
    }
  }, [])

  // Add event listeners to the form area to prevent disabling auto-scroll when interacting with form
  useEffect(() => {
    const form = document.querySelector('form')
    if (!form) return

    const handleFormMouseDown = () => {
      // Set flag to indicate form interaction
      isFormInteractionRef.current = true

      // Reset the flag after a short delay
      setTimeout(() => {
        isFormInteractionRef.current = false
      }, 500) // Give enough time for the form interaction to complete
    }

    form.addEventListener('mousedown', handleFormMouseDown)

    return () => {
      form.removeEventListener('mousedown', handleFormMouseDown)
    }
  }, [])

  // Use a longer debounce time for better performance with large message updates
  const debouncedMessages = useDebounce(messages, 150)
  useEffect(() => {
    // Only auto-scroll if enabled and there are messages
    if (shouldFollowScrollRef.current && debouncedMessages.length > 0) {
      // Force scroll to bottom when messages change
      scrollToBottom()
    }
  }, [debouncedMessages, scrollToBottom])

  const clearMessages = useCallback(() => {
    setMessages([])
    useSettingsStore.getState().setRetrievalHistory([])
  }, [])

  // Handle copying message content with robust clipboard support
  const handleCopyMessage = useCallback(
    async (message: MessageWithError) => {
      let contentToCopy = ''

      if (message.role === 'user') {
        // User messages: copy original content
        contentToCopy = message.content || ''
      } else {
        // Assistant messages: prefer processed display content, fallback to original content
        const finalDisplayContent =
          message.displayContent !== undefined ? message.displayContent : message.content || ''
        contentToCopy = finalDisplayContent
      }

      if (!contentToCopy.trim()) {
        toast.error(t('retrievePanel.chatMessage.copyEmpty', 'No content to copy'))
        return
      }

      try {
        const result = await copyToClipboard(contentToCopy)

        if (result.success) {
          // Show success message with method used
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
            fallback: t('retrievePanel.chatMessage.copySuccess', 'Content copied to clipboard'),
          }

          toast.success(
            methodMessages[result.method] ||
              t('retrievePanel.chatMessage.copySuccess', 'Content copied to clipboard')
          )
        } else {
          // Show error with fallback instructions
          if (result.method === 'fallback') {
            toast.error(
              result.error || t('retrievePanel.chatMessage.copyFailed', 'Failed to copy content'),
              {
                description: t(
                  'retrievePanel.chatMessage.copyManualInstruction',
                  'Please select and copy the text manually'
                ),
              }
            )
          } else {
            toast.error(t('retrievePanel.chatMessage.copyFailed', 'Failed to copy content'), {
              description: result.error,
            })
          }
        }
      } catch (err) {
        console.error('Clipboard operation failed:', err)
        toast.error(t('retrievePanel.chatMessage.copyError', 'Copy operation failed'), {
          description: err instanceof Error ? err.message : 'Unknown error occurred',
        })
      }
    },
    [t]
  )

  return (
    <div className="flex size-full gap-2 px-2 pb-12 overflow-hidden">
      <div className="flex grow flex-col gap-4">
        <div className="relative grow">
          <div
            ref={messagesContainerRef}
            role="log"
            aria-live="polite"
            className="bg-primary-foreground/60 absolute inset-0 flex flex-col overflow-auto rounded-lg border p-2"
            onClick={() => {
              if (shouldFollowScrollRef.current) {
                shouldFollowScrollRef.current = false
              }
            }}
            onKeyDown={(e) => {
              if (e.key === 'Escape' && shouldFollowScrollRef.current) {
                shouldFollowScrollRef.current = false
              }
            }}
          >
            <div className="flex min-h-0 flex-1 flex-col gap-2">
              {messages.length === 0 ? (
                <div className="text-muted-foreground flex h-full items-center justify-center text-lg">
                  {t('retrievePanel.retrieval.startPrompt')}
                </div>
              ) : (
                messages.map((message) => {
                  // Remove unused idx
                  // isComplete logic is now handled internally based on message.mermaidRendered
                  return (
                    <div
                      key={message.id} // Use stable ID for key
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} items-end gap-2`}
                    >
                      {message.role === 'user' && (
                        <Button
                          onClick={() => handleCopyMessage(message)}
                          className="mb-2 size-6 rounded-md opacity-60 transition-opacity hover:opacity-100 shrink-0"
                          tooltip={t('retrievePanel.chatMessage.copyTooltip')}
                          variant="ghost"
                          size="icon"
                        >
                          <CopyIcon className="size-4" />
                        </Button>
                      )}
                      <ChatMessage message={message} isTabActive={isRetrievalTabActive} />
                      {message.role === 'assistant' && (
                        <Button
                          onClick={() => handleCopyMessage(message)}
                          className="mb-2 size-6 rounded-md opacity-60 transition-opacity hover:opacity-100 shrink-0"
                          tooltip={t('retrievePanel.chatMessage.copyTooltip')}
                          variant="ghost"
                          size="icon"
                        >
                          <CopyIcon className="size-4" />
                        </Button>
                      )}
                    </div>
                  )
                })
              )}
              <div ref={messagesEndRef} className="pb-1" />
            </div>
          </div>
        </div>

        <search>
          <form
            onSubmit={handleSubmit}
            className="flex shrink-0 items-center gap-2"
            autoComplete="on"
            method="post"
            action="#"
          >
            {/* Hidden submit button to ensure form meets HTML standards */}
            <input type="submit" style={{ display: 'none' }} tabIndex={-1} />
            <Button
              type="button"
              variant="outline"
              onClick={clearMessages}
              disabled={isLoading}
              size="sm"
            >
              <EraserIcon />
              {t('retrievePanel.retrieval.clear')}
            </Button>
            <div className="flex-1 relative">
              <label htmlFor="query-input" className="sr-only">
                {t('retrievePanel.retrieval.placeholder')}
              </label>
              {hasMultipleLines ? (
                <Textarea
                  ref={inputRef as React.RefObject<HTMLTextAreaElement>}
                  id="query-input"
                  autoComplete="on"
                  className="w-full min-h-[40px] max-h-[120px] overflow-y-auto"
                  value={inputValue}
                  onChange={handleChange}
                  onKeyDown={handleKeyDown}
                  onPaste={handlePaste}
                  placeholder={t('retrievePanel.retrieval.placeholder')}
                  disabled={isLoading}
                  rows={1}
                  style={{
                    resize: 'none',
                    height: 'auto',
                    minHeight: '40px',
                    maxHeight: '120px',
                  }}
                  onInput={(e: React.FormEvent<HTMLTextAreaElement>) => {
                    const target = e.target as HTMLTextAreaElement
                    requestAnimationFrame(() => {
                      target.style.height = 'auto'
                      target.style.height = `${Math.min(target.scrollHeight, 120)}px`
                    })
                  }}
                />
              ) : (
                <Input
                  ref={inputRef as React.RefObject<HTMLInputElement>}
                  id="query-input"
                  autoComplete="on"
                  className="w-full"
                  value={inputValue}
                  onChange={handleChange}
                  onKeyDown={handleKeyDown}
                  onPaste={handlePaste}
                  placeholder={t('retrievePanel.retrieval.placeholder')}
                  disabled={isLoading}
                />
              )}
              {/* Error message below input */}
              {inputError && (
                <div className="absolute left-0 top-full mt-1 text-xs text-red-500">
                  {inputError}
                </div>
              )}
            </div>
            <Button type="submit" variant="default" disabled={isLoading} size="sm">
              <SendIcon />
              {t('retrievePanel.retrieval.send')}
            </Button>
          </form>
        </search>
      </div>
      <QuerySettings />
    </div>
  )
}
