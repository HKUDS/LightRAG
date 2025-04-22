import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { useCallback, useEffect, useRef, useState } from 'react'
import { queryText, queryTextStream, Message } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { useDebounce } from '@/hooks/useDebounce'
import QuerySettings from '@/components/retrieval/QuerySettings'
import { ChatMessage, MessageWithError } from '@/components/retrieval/ChatMessage'
import { EraserIcon, SendIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export default function RetrievalTesting() {
  const { t } = useTranslation()
  const [messages, setMessages] = useState<MessageWithError[]>(
    () => useSettingsStore.getState().retrievalHistory || []
  )
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  // Reference to track if we should follow scroll during streaming (using ref for synchronous updates)
  const shouldFollowScrollRef = useRef(true)
  // Reference to track if this is the first chunk of a streaming response
  const isFirstChunkRef = useRef(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)

  // Check if the container is near the bottom
  const isNearBottom = useCallback(() => {
    const container = messagesContainerRef.current
    if (!container) return true // Default to true if no container reference

    // Calculate distance to bottom
    const { scrollTop, scrollHeight, clientHeight } = container
    const distanceToBottom = scrollHeight - scrollTop - clientHeight

    // Consider near bottom if less than 100px from bottom
    return distanceToBottom < 100
  }, [])

  const scrollToBottom = useCallback((force = false) => {
    // Only scroll if forced or user is already near bottom
    if (force || isNearBottom()) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [isNearBottom])

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      if (!inputValue.trim() || isLoading) return

      // Create messages
      const userMessage: Message = {
        content: inputValue,
        role: 'user'
      }

      const assistantMessage: Message = {
        content: '',
        role: 'assistant'
      }

      const prevMessages = [...messages]

      // Add messages to chatbox
      setMessages([...prevMessages, userMessage, assistantMessage])
      
      // Reset first chunk flag for new streaming response
      isFirstChunkRef.current = true
      // Enable follow scroll for new query
      shouldFollowScrollRef.current = true
      
      // Force scroll to bottom after messages are rendered
      setTimeout(() => {
        scrollToBottom(true)
      }, 0)

      // Clear input and set loading
      setInputValue('')
      setIsLoading(true)

      // Create a function to update the assistant's message
      const updateAssistantMessage = (chunk: string, isError?: boolean) => {
        // Check if this is the first chunk of the streaming response
        if (isFirstChunkRef.current) {
          // Determine scroll behavior based on initial position
          shouldFollowScrollRef.current = isNearBottom();
          isFirstChunkRef.current = false;
        }
        
        // Save current scroll position before updating content
        const container = messagesContainerRef.current;
        const currentScrollPosition = container ? container.scrollTop : 0;
        
        assistantMessage.content += chunk
        setMessages((prev) => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage.role === 'assistant') {
            lastMessage.content = assistantMessage.content
            lastMessage.isError = isError
          }
          return newMessages
        })
        
        // After updating content, check if we should scroll
        // Use consistent scrolling behavior throughout the streaming response
        if (shouldFollowScrollRef.current) {
          scrollToBottom(true);
        } else if (container) {
          // If user was not near bottom, restore their scroll position
          // This needs to be in a setTimeout to work after React updates the DOM
          setTimeout(() => {
            if (container) {
              container.scrollTop = currentScrollPosition;
            }
          }, 0);
        }
      }

      // Prepare query parameters
      const state = useSettingsStore.getState()
      const queryParams = {
        ...state.querySettings,
        query: userMessage.content,
        conversation_history: prevMessages
          .filter((m) => m.isError !== true)
          .slice(-(state.querySettings.history_turns || 0) * 2)
          .map((m) => ({ role: m.role, content: m.content }))
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
        useSettingsStore
          .getState()
          .setRetrievalHistory([...prevMessages, userMessage, assistantMessage])
      }
    },
    [inputValue, isLoading, messages, setMessages, t, scrollToBottom]
  )

  // Add scroll event listener to detect when user manually scrolls
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    
    const handleScroll = () => {
      const isNearBottomNow = isNearBottom();
      
      // If user scrolls away from bottom while in auto-scroll mode, disable it
      if (shouldFollowScrollRef.current && !isNearBottomNow) {
        shouldFollowScrollRef.current = false;
      }
      // If user scrolls back to bottom while not in auto-scroll mode, re-enable it
      else if (!shouldFollowScrollRef.current && isNearBottomNow) {
        shouldFollowScrollRef.current = true;
      }
    };
    
    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, [isNearBottom]); // Remove shouldFollowScroll from dependencies since we're using ref now

  const debouncedMessages = useDebounce(messages, 100)
  useEffect(() => scrollToBottom(false), [debouncedMessages, scrollToBottom])

  const clearMessages = useCallback(() => {
    setMessages([])
    useSettingsStore.getState().setRetrievalHistory([])
  }, [setMessages])

  return (
    <div className="flex size-full gap-2 px-2 pb-12 overflow-hidden">
      <div className="flex grow flex-col gap-4">
        <div className="relative grow">
          <div ref={messagesContainerRef} className="bg-primary-foreground/60 absolute inset-0 flex flex-col overflow-auto rounded-lg border p-2">
            <div className="flex min-h-0 flex-1 flex-col gap-2">
              {messages.length === 0 ? (
                <div className="text-muted-foreground flex h-full items-center justify-center text-lg">
                  {t('retrievePanel.retrieval.startPrompt')}
                </div>
              ) : (
                messages.map((message, idx) => (
                  <div
                    key={idx}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {<ChatMessage message={message} />}
                  </div>
                ))
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
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={t('retrievePanel.retrieval.placeholder')}
              disabled={isLoading}
            />
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
