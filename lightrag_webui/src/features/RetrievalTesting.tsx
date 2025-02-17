import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { useCallback, useEffect, useRef, useState } from 'react'
import { queryText, queryTextStream, Message as ChatMessage } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { useDebounce } from '@/hooks/useDebounce'
import QuerySettings from '@/components/retrieval/QuerySettings'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeReact from 'rehype-react'
import remarkMath from 'remark-math'

import { EraserIcon, SendIcon, LoaderIcon } from 'lucide-react'

type Message = ChatMessage & {
  isError?: boolean
}

const ChatMessageComponent = ({ message }: { message: Message }) => {
  return (
    <ReactMarkdown
      className="prose lg:prose-xs dark:prose-invert max-w-none text-base"
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeReact]}
      skipHtml={false}
    >
      {message.content}
    </ReactMarkdown>
  )
}

export default function RetrievalTesting() {
  const [messages, setMessages] = useState<Message[]>(
    () => useSettingsStore.getState().retrievalHistory || []
  )
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

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

      // Clear input and set loading
      setInputValue('')
      setIsLoading(true)

      // Create a function to update the assistant's message
      const updateAssistantMessage = (chunk: string, isError?: boolean) => {
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
      }

      // Prepare query parameters
      const state = useSettingsStore.getState()
      const queryParams = {
        ...state.querySettings,
        query: userMessage.content,
        conversation_history: prevMessages
          .filter((m) => m.isError !== true)
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
        updateAssistantMessage(`Error: Failed to get response\n${errorMessage(err)}`, true)
      } finally {
        // Clear loading and add messages to state
        setIsLoading(false)
        useSettingsStore
          .getState()
          .setRetrievalHistory([...prevMessages, userMessage, assistantMessage])
      }
    },
    [inputValue, isLoading, messages, setMessages]
  )

  const debouncedMessages = useDebounce(messages, 100)
  useEffect(() => scrollToBottom(), [debouncedMessages, scrollToBottom])

  const clearMessages = useCallback(() => {
    setMessages([])
    useSettingsStore.getState().setRetrievalHistory([])
  }, [setMessages])

  return (
    <div className="flex size-full gap-2 px-2 pb-12">
      <div className="flex grow flex-col gap-4">
        <div className="relative grow">
          <div className="bg-primary-foreground/60 absolute inset-0 flex flex-col overflow-auto rounded-lg border p-2">
            <div className="flex min-h-0 flex-1 flex-col gap-2">
              {messages.length === 0 ? (
                <div className="text-muted-foreground flex h-full items-center justify-center text-lg">
                  Start a retrieval by typing your query below
                </div>
              ) : (
                messages.map((message, idx) => (
                  <div
                    key={idx}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg px-4 py-2 ${
                        message.role === 'user'
                          ? 'bg-primary text-primary-foreground'
                          : message.isError
                            ? 'bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400'
                            : 'bg-muted'
                      }`}
                    >
                      <pre className="break-words whitespace-pre-wrap">
                        {<ChatMessageComponent message={message} />}
                      </pre>
                      {message.content.length === 0 && (
                        <LoaderIcon className="animate-spin duration-2000" />
                      )}
                    </div>
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
            Clear
          </Button>
          <Input
            className="flex-1"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type your query..."
            disabled={isLoading}
          />
          <Button type="submit" variant="default" disabled={isLoading} size="sm">
            <SendIcon />
            Send
          </Button>
        </form>
      </div>
      <QuerySettings />
    </div>
  )
}
