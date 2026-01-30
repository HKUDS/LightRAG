import { useState, useRef, useEffect } from 'react'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { SendIcon, BotIcon, UserIcon, Loader2 } from 'lucide-react'
import { queryTextStream, ChatMessage } from '@/api/lightrag'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'

interface ChatInterfaceProps {
    sessionId: string | null
    initialMessages: ChatMessage[]
    onMessageSent: () => void
}

export default function ChatInterface({ sessionId, initialMessages, onMessageSent }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setMessages(initialMessages)
  }, [initialMessages, sessionId])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!inputValue.trim() || !sessionId || isStreaming) return

    const userMessage = inputValue.trim()
    setInputValue('')

    // Optimistically add user message
    const newMessages: ChatMessage[] = [
      ...messages,
      { role: 'user', content: userMessage, created_at: new Date().toISOString() }
    ]
    setMessages(newMessages)
    setIsStreaming(true)

    // Placeholder for assistant message
    const assistantMsg: ChatMessage = { role: 'assistant', content: '', created_at: new Date().toISOString() }
    setMessages([...newMessages, assistantMsg])

    try {
      await queryTextStream({
        query: userMessage,
        mode: 'hybrid', // Default mode
        stream: true,
        session_id: sessionId
      }, (chunk) => {
        setMessages(prev => {
          const updated = [...prev]
          if (updated[updated.length - 1].role === 'assistant') {
            updated[updated.length - 1].content += chunk
          }
          return updated
        })
      }, (error) => {
        toast.error(`Error: ${error}`)
      })

      // Refresh list in parent to update timestamps/previews if needed
      onMessageSent()

    } catch (err) {
      console.error(err)
      toast.error('Failed to send message')
    } finally {
      setIsStreaming(false)
    }
  }

  if (!sessionId) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground">
        Select a chat or create a new one to start messaging.
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full relative">
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="space-y-4 pb-4">
          {messages.map((msg, idx) => (
            <div key={idx} className={cn(
              'flex w-full',
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            )}>
              <div className={cn(
                'max-w-[80%] rounded-lg p-3',
                msg.role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground'
              )}>
                <div className="flex items-center gap-2 mb-1">
                  {msg.role === 'user' ? <UserIcon size={14} /> : <BotIcon size={14} />}
                  <span className="text-xs opacity-70">
                    {msg.role === 'user' ? 'You' : 'Assistant'}
                  </span>
                </div>
                <div className="whitespace-pre-wrap text-sm">
                  {msg.content}
                </div>
              </div>
            </div>
          ))}
          {isStreaming && (
            <div className="flex justify-start w-full">
              <div className="max-w-[80%] rounded-lg p-3 bg-muted text-muted-foreground flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-xs">Thinking...</span>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
      <div className="p-4 border-t bg-background">
        <form onSubmit={handleSend} className="flex gap-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type a message..."
            disabled={isStreaming}
            className="flex-1"
          />
          <Button type="submit" disabled={isStreaming || !inputValue.trim()}>
            <SendIcon size={16} />
          </Button>
        </form>
      </div>
    </div>
  )
}
