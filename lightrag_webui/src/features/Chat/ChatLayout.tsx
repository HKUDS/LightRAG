import { useState, useEffect, useCallback } from 'react'
import { Card } from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { PlusIcon, MessageSquareIcon, TrashIcon } from 'lucide-react'
import { listChats, createChat, deleteChat, getChatMessages, ChatSession, ChatMessage } from '@/api/lightrag'
import ChatInterface from './ChatInterface'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'

export default function ChatLayout() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoadingSessions, setIsLoadingSessions] = useState(false)
  const [isLoadingMessages, setIsLoadingMessages] = useState(false)

  const fetchSessions = useCallback(async () => {
    setIsLoadingSessions(true)
    try {
      const data = await listChats()
      setSessions(data)
    } catch (error) {
      console.error(error)
      toast.error('Failed to load chat sessions')
    } finally {
      setIsLoadingSessions(false)
    }
  }, [])

  const fetchMessages = useCallback(async (id: string) => {
    setIsLoadingMessages(true)
    try {
      const data = await getChatMessages(id)
      setMessages(data)
      setSelectedSessionId(id)
    } catch (error) {
      console.error(error)
      toast.error('Failed to load messages')
    } finally {
      setIsLoadingMessages(false)
    }
  }, [])

  useEffect(() => {
    fetchSessions()
  }, [fetchSessions])

  const handleCreateChat = async () => {
    try {
      const newSession = await createChat('New Chat')
      setSessions([newSession, ...sessions])
      setSelectedSessionId(newSession.id)
      setMessages([])
    } catch (error) {
      console.error(error)
      toast.error('Failed to create chat')
    }
  }

  const handleDeleteChat = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    if (!confirm('Are you sure you want to delete this chat?')) return

    try {
      await deleteChat(id)
      setSessions(sessions.filter(s => s.id !== id))
      if (selectedSessionId === id) {
        setSelectedSessionId(null)
        setMessages([])
      }
      toast.success('Chat deleted')
    } catch (error) {
      console.error(error)
      toast.error('Failed to delete chat')
    }
  }

  const handleSelectSession = (id: string) => {
    if (selectedSessionId === id) return
    fetchMessages(id)
  }

  return (
    <div className="flex h-full w-full gap-4 pt-4">
      {/* Sidebar */}
      <Card className="flex flex-col w-[300px] h-full overflow-hidden">
        <div className="p-4 border-b">
          <Button onClick={handleCreateChat} className="w-full flex gap-2">
            <PlusIcon size={16} /> New Chat
          </Button>
        </div>
        <ScrollArea className="flex-1">
          <div className="flex flex-col p-2 gap-1">
            {isLoadingSessions ? (
              <div className="text-center p-4 text-sm text-muted-foreground">Loading chats...</div>
            ) : sessions.length === 0 ? (
              <div className="text-center p-4 text-sm text-muted-foreground">No chat sessions</div>
            ) : (
              sessions.map(session => (
                <div
                  key={session.id}
                  onClick={() => handleSelectSession(session.id)}
                  className={cn(
                    'flex items-center justify-between p-3 rounded-md cursor-pointer transition-colors group',
                    selectedSessionId === session.id
                      ? 'bg-accent/50 text-accent-foreground'
                      : 'hover:bg-muted'
                  )}
                >
                  <div className="flex items-center gap-2 overflow-hidden">
                    <MessageSquareIcon size={16} className="shrink-0 opacity-70" />
                    <span className="truncate text-sm font-medium">
                      {session.title || 'Untitled Chat'}
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(e) => handleDeleteChat(e, session.id)}
                  >
                    <TrashIcon size={14} className="text-destructive" />
                  </Button>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </Card>

      {/* Main Chat Area */}
      <Card className="flex-1 h-full overflow-hidden flex flex-col">
        {isLoadingMessages ? (
          <div className="flex h-full items-center justify-center">Loading messages...</div>
        ) : (
          <ChatInterface
            sessionId={selectedSessionId}
            initialMessages={messages}
            onMessageSent={() => { }}
          />
        )}
      </Card>
    </div>
  )
}
