import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { useCallback, useEffect, useRef, useState } from 'react'
import { throttle } from '@/lib/utils'
import { queryText, queryTextStream } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { useDebounce } from '@/hooks/useDebounce'
import QuerySettings from '@/components/retrieval/QuerySettings'
import { ChatMessage, MessageWithError } from '@/components/retrieval/ChatMessage'
import { EraserIcon, SendIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { QueryMode } from '@/api/lightrag'

// Helper function to generate unique IDs with browser compatibility
const generateUniqueId = () => {
  // Use crypto.randomUUID() if available
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  // Fallback to timestamp + random string for browsers without crypto.randomUUID
  return `id-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
};

export default function RetrievalTesting() {
  const { t } = useTranslation()
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
            mermaidRendered: msgWithError.mermaidRendered ?? true // Assume historical mermaid is rendered
          }
        } catch (error) {
          console.error('Error processing message:', error)
          // Return a default message if there's an error
          return {
            role: 'system',
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
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [inputError, setInputError] = useState('') // Error message for input
  // Reference to track if we should follow scroll during streaming (using ref for synchronous updates)
  const shouldFollowScrollRef = useRef(true)
  // Reference to track if user interaction is from the form area
  const isFormInteractionRef = useRef(false)
  // Reference to track if scroll was triggered programmatically
  const programmaticScrollRef = useRef(false)
  // Reference to track if we're currently receiving a streaming response
  const isReceivingResponseRef = useRef(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)

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
      const prefixMatch = inputValue.match(/^\/(\w+)\s+(.+)/)
      let modeOverride: QueryMode | undefined = undefined
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

      // Create messages
      // Save the original input (with prefix if any) in userMessage.content for display
      const userMessage: MessageWithError = {
        id: generateUniqueId(), // Use browser-compatible ID generation
        content: inputValue,
        role: 'user'
      }

      const assistantMessage: MessageWithError = {
        id: generateUniqueId(), // Use browser-compatible ID generation
        content: '',
        role: 'assistant',
        mermaidRendered: false
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

      // Create a function to update the assistant's message
      const updateAssistantMessage = (chunk: string, isError?: boolean) => {
        assistantMessage.content += chunk

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

        setMessages((prev) => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage.role === 'assistant') {
            lastMessage.content = assistantMessage.content
            lastMessage.isError = isError
            lastMessage.mermaidRendered = assistantMessage.mermaidRendered
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
      const queryParams = {
        ...state.querySettings,
        query: actualQuery,
        conversation_history: prevMessages
          .filter((m) => m.isError !== true)
          .slice(-(state.querySettings.history_turns || 0) * 2)
          .map((m) => ({ role: m.role, content: m.content })),
        ...(modeOverride ? { mode: modeOverride } : {})
      }

      try {
        // Run query
        if (state.querySettings.stream) {
          let errorMessage = ''
          await queryTextStream(queryParams, updateAssistantMessage, (error) => {
            errorMessage += error
          })
          if (errorMessage) {
            if (assistantMessage.content) {
              errorMessage = assistantMessage.content + '\n' + errorMessage
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
        useSettingsStore
          .getState()
          .setRetrievalHistory([...prevMessages, userMessage, assistantMessage])
      }
    },
    [inputValue, isLoading, messages, setMessages, t, scrollToBottom]
  )

  // Add event listeners to detect when user manually interacts with the container
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    // Handle significant mouse wheel events - only disable auto-scroll for deliberate scrolling
    const handleWheel = (e: WheelEvent) => {
      // Only consider significant wheel movements (more than 10px)
      if (Math.abs(e.deltaY) > 10 && !isFormInteractionRef.current) {
        shouldFollowScrollRef.current = false;
      }
    };

    // Handle scroll events - only disable auto-scroll if not programmatically triggered
    // and if it's a significant scroll
    const handleScroll = throttle(() => {
      // If this is a programmatic scroll, don't disable auto-scroll
      if (programmaticScrollRef.current) {
        programmaticScrollRef.current = false;
        return;
      }

      // Check if scrolled to bottom or very close to bottom
      const container = messagesContainerRef.current;
      if (container) {
        const isAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 20;

        // If at bottom, enable auto-scroll, otherwise disable it
        if (isAtBottom) {
          shouldFollowScrollRef.current = true;
        } else if (!isFormInteractionRef.current && !isReceivingResponseRef.current) {
          shouldFollowScrollRef.current = false;
        }
      }
    }, 30);

    // Add event listeners - only listen for wheel and scroll events
    container.addEventListener('wheel', handleWheel as EventListener);
    container.addEventListener('scroll', handleScroll as EventListener);

    return () => {
      container.removeEventListener('wheel', handleWheel as EventListener);
      container.removeEventListener('scroll', handleScroll as EventListener);
    };
  }, []);

  // Add event listeners to the form area to prevent disabling auto-scroll when interacting with form
  useEffect(() => {
    const form = document.querySelector('form');
    if (!form) return;

    const handleFormMouseDown = () => {
      // Set flag to indicate form interaction
      isFormInteractionRef.current = true;

      // Reset the flag after a short delay
      setTimeout(() => {
        isFormInteractionRef.current = false;
      }, 500); // Give enough time for the form interaction to complete
    };

    form.addEventListener('mousedown', handleFormMouseDown);

    return () => {
      form.removeEventListener('mousedown', handleFormMouseDown);
    };
  }, []);

  // Use a longer debounce time for better performance with large message updates
  const debouncedMessages = useDebounce(messages, 150)
  useEffect(() => {
    // Only auto-scroll if enabled
    if (shouldFollowScrollRef.current) {
      // Force scroll to bottom when messages change
      scrollToBottom()
    }
  }, [debouncedMessages, scrollToBottom])


  const clearMessages = useCallback(() => {
    setMessages([])
    useSettingsStore.getState().setRetrievalHistory([])
  }, [setMessages])

  return (
    <div className="flex size-full gap-2 px-2 pb-12 overflow-hidden">
      <div className="flex grow flex-col gap-4">
        <div className="relative grow">
          <div
            ref={messagesContainerRef}
            className="bg-primary-foreground/60 absolute inset-0 flex flex-col overflow-auto rounded-lg border p-2"
            onClick={() => {
              if (shouldFollowScrollRef.current) {
                shouldFollowScrollRef.current = false;
              }
            }}
          >
            <div className="flex min-h-0 flex-1 flex-col gap-2">
              {messages.length === 0 ? (
                <div className="text-muted-foreground flex h-full items-center justify-center text-lg">
                  {t('retrievePanel.retrieval.startPrompt')}
                </div>
              ) : (
                messages.map((message) => { // Remove unused idx
                  // isComplete logic is now handled internally based on message.mermaidRendered
                  return (
                    <div
                      key={message.id} // Use stable ID for key
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      {<ChatMessage message={message} />}
                    </div>
                  );
                })
              )}
              <div ref={messagesEndRef} className="pb-1" />
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="flex shrink-0 items-center gap-2">
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
            <Input
              id="query-input"
              className="w-full"
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value)
                if (inputError) setInputError('')
              }}
              placeholder={t('retrievePanel.retrieval.placeholder')}
              disabled={isLoading}
            />
            {/* Error message below input */}
            {inputError && (
              <div className="absolute left-0 top-full mt-1 text-xs text-red-500">{inputError}</div>
            )}
          </div>
          <Button type="submit" variant="default" disabled={isLoading} size="sm">
            <SendIcon />
            {t('retrievePanel.retrieval.send')}
          </Button>
        </form>
      </div>
      <QuerySettings />
    </div>
  )
}
