import { ChatSession, getSessions } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { cn } from '@/lib/utils'
import { format } from 'date-fns'
import { MessageSquareIcon, PlusIcon } from 'lucide-react'
import { useEffect, useState } from 'react'

interface SessionManagerProps {
  currentSessionId: string | null
  onSessionSelect: (sessionId: string) => void
  onNewSession: () => void
}

export default function SessionManager({
  currentSessionId,
  onSessionSelect,
  onNewSession
}: SessionManagerProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const fetchSessions = async () => {
    setIsLoading(true)
    try {
      const data = await getSessions()
      setSessions(data)
    } catch (error) {
      console.error('Failed to fetch sessions:', error)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchSessions()
  }, [currentSessionId]) // Refresh list when session changes (e.g. new one created)

  const handleNewSession = async () => {
    onNewSession()
  }

  return (
    <div className="flex flex-col h-full border-r w-64 bg-muted/10">
      <div className="p-4 border-b">
        <Button onClick={handleNewSession} className="w-full justify-start gap-2" variant="outline">
          <PlusIcon className="w-4 h-4" />
          New Chat
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {sessions.map((session) => (
            <Button
              key={session.id}
              variant={currentSessionId === session.id ? "secondary" : "ghost"}
              className={cn(
                "w-full justify-start text-left font-normal h-auto py-3 px-3",
                currentSessionId === session.id && "bg-muted"
              )}
              onClick={() => onSessionSelect(session.id)}
            >
              <MessageSquareIcon className="w-4 h-4 mr-2 mt-0.5 shrink-0" />
              <div className="flex flex-col gap-1 overflow-hidden">
                <span className="truncate text-sm font-medium">
                  {session.title || "Untitled Chat"}
                </span>
                <span className="text-xs text-muted-foreground">
                  {(() => {
                    try {
                      return session.updated_at 
                        ? format(new Date(session.updated_at), 'MMM d, HH:mm')
                        : ''
                    } catch (e) {
                      return ''
                    }
                  })()}
                </span>
              </div>
            </Button>
          ))}
          {sessions.length === 0 && !isLoading && (
            <div className="text-center text-sm text-muted-foreground p-4">
              No history yet
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
