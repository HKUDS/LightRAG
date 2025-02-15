import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { useCallback, useEffect, useRef, useState } from 'react'
import { queryTextStream, QueryMode } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { useDebounce } from '@/hooks/useDebounce'
import { EraserIcon, SendIcon, LoaderIcon } from 'lucide-react'

type Message = {
  id: string
  content: string
  role: 'User' | 'LightRAG'
}

export default function RetrievalTesting() {
  const [messages, setMessages] = useState<Message[]>(
    () => useSettingsStore.getState().retrievalHistory || []
  )
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [mode, setMode] = useState<QueryMode>('mix')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      if (!inputValue.trim() || isLoading) return

      const userMessage: Message = {
        id: Date.now().toString(),
        content: inputValue,
        role: 'User'
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: '',
        role: 'LightRAG'
      }

      setMessages((prev) => {
        const newMessages = [...prev, userMessage, assistantMessage]
        return newMessages
      })

      setInputValue('')
      setIsLoading(true)

      // Create a function to update the assistant's message
      const updateAssistantMessage = (chunk: string) => {
        assistantMessage.content += chunk
        setMessages((prev) => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage.role === 'LightRAG') {
            lastMessage.content = assistantMessage.content
          }
          return newMessages
        })
      }

      try {
        await queryTextStream(
          {
            query: userMessage.content,
            mode: mode,
            stream: true
          },
          updateAssistantMessage
        )
      } catch (err) {
        updateAssistantMessage(`Error: Failed to get response\n${errorMessage(err)}`)
      } finally {
        setIsLoading(false)
        useSettingsStore
          .getState()
          .setRetrievalHistory([
            ...useSettingsStore.getState().retrievalHistory,
            userMessage,
            assistantMessage
          ])
      }
    },
    [inputValue, isLoading, mode, setMessages]
  )

  const debouncedMessages = useDebounce(messages, 100)
  useEffect(() => scrollToBottom(), [debouncedMessages, scrollToBottom])

  const clearMessages = useCallback(() => {
    setMessages([])
    useSettingsStore.getState().setRetrievalHistory([])
  }, [setMessages])

  return (
    <div className="flex size-full flex-col gap-4 px-32 py-6">
      <div className="relative grow">
        <div className="bg-primary-foreground/60 absolute inset-0 flex flex-col overflow-auto rounded-lg border p-2">
          <div className="flex min-h-0 flex-1 flex-col gap-2">
            {messages.length === 0 ? (
              <div className="text-muted-foreground flex h-full items-center justify-center text-lg">
                Start a retrieval by typing your query below
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'User' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-2 ${
                      message.role === 'User' ? 'bg-primary text-primary-foreground' : 'bg-muted'
                    }`}
                  >
                    <pre className="break-words whitespace-pre-wrap">{message.content}</pre>
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

      <form onSubmit={handleSubmit} className="flex shrink-0 items-center gap-2 pb-2">
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
        <select
          className="border-input bg-background ring-offset-background h-9 rounded-md border px-3 py-1 text-sm"
          value={mode}
          onChange={(e) => setMode(e.target.value as QueryMode)}
          disabled={isLoading}
        >
          <option value="naive">Naive</option>
          <option value="local">Local</option>
          <option value="global">Global</option>
          <option value="hybrid">Hybrid</option>
          <option value="mix">Mix</option>
        </select>
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
  )
}
