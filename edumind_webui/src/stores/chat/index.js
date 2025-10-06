import { defineStore } from 'pinia'
import { queryApi } from '@/api'
import { useWorkspaceContextStore } from '../workspaceContext'
import { useUserStore } from '../user'

const CHAT_HISTORY_KEY = 'edumind_ai_chat_history'
const MAX_HISTORY_MESSAGES = 10
const cryptoApi = typeof globalThis !== 'undefined' ? globalThis.crypto : undefined

const normalizeUserId = (userId) =>
  typeof userId === 'string' && userId.trim() ? userId.trim() : 'default'

const toHistoryKey = (userId) => `${CHAT_HISTORY_KEY}:${normalizeUserId(userId)}`

const generateMessageId = () => {
  if (cryptoApi && typeof cryptoApi.randomUUID === 'function') {
    return cryptoApi.randomUUID()
  }
  return `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

const validRoles = new Set(['user', 'assistant', 'system'])

const normalizeRole = (role) => (validRoles.has(role) ? role : 'assistant')

const createFallbackMessage = (index) => ({
  id: `error-${Date.now()}-${index}`,
  role: 'system',
  content: 'Error loading message',
  isError: true,
})

const sanitizeStoredMessage = (message, index) => {
  try {
    if (!message || typeof message !== 'object') {
      throw new Error('Invalid message payload')
    }

    const sanitized = {
      id: typeof message.id === 'string' && message.id ? message.id : generateMessageId(),
      role: normalizeRole(message.role),
      content: typeof message.content === 'string' ? message.content : '',
    }

    if (message.isError === true) {
      sanitized.isError = true
    }

    return sanitized
  } catch (error) {
    console.error('Failed to restore chat message from storage', error)
    return createFallbackMessage(index)
  }
}

const loadStoredMessages = (userId) => {
  if (typeof window === 'undefined' || !window.localStorage) {
    return { messages: [], dirty: false }
  }

  try {
    const raw = window.localStorage.getItem(toHistoryKey(userId))
    if (!raw) {
      return { messages: [], dirty: false }
    }

    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) {
      return { messages: [], dirty: false }
    }

    let dirty = false
    const sanitized = parsed.map((message, index) => {
      const normalized = sanitizeStoredMessage(message, index)
      if (!dirty) {
        const original = message || {}
        dirty =
          normalized.id !== original.id ||
          normalized.role !== original.role ||
          normalized.content !== original.content ||
          !!normalized.isError !== !!original.isError
      }
      return normalized
    })

    return { messages: sanitized, dirty }
  } catch (error) {
    console.error('Failed to parse chat history from storage', error)
    return { messages: [], dirty: false }
  }
}

const persistMessages = (messages, userId) => {
  if (typeof window === 'undefined' || !window.localStorage) {
    return
  }

  try {
    window.localStorage.setItem(toHistoryKey(userId), JSON.stringify(messages))
  } catch (error) {
    console.error('Failed to persist chat history', error)
  }
}

const createMessage = (role, content = '') => ({
  id: generateMessageId(),
  role,
  content,
})

export const useChatStore = defineStore('chat', {
  state: () => {
    const userStore = useUserStore()
    const normalizedUserId = normalizeUserId(userStore.userId)
    const { messages, dirty } = loadStoredMessages(normalizedUserId)
    if (dirty) {
      persistMessages(messages, normalizedUserId)
    }

    return {
      messages,
      historyUserId: normalizedUserId,
      inputValue: '',
      streaming: false,
      error: null,
    }
  },
  getters: {
    hasMessages: (state) => state.messages.length > 0,
    canSend: (state) => state.inputValue.trim().length > 0 && !state.streaming,
  },
  actions: {
    setInput(value) {
      this.inputValue = value
    },
    hydrateHistoryForCurrentUser() {
      const userStore = useUserStore()
      const normalizedUserId = normalizeUserId(userStore.userId)
      const { messages, dirty } = loadStoredMessages(normalizedUserId)
      this.messages = messages
      this.historyUserId = normalizedUserId
      if (dirty) {
        persistMessages(this.messages, normalizedUserId)
      }
    },
    persistHistory() {
      const keyUserId = this.historyUserId || 'default'
      persistMessages(this.messages, keyUserId)
    },
    clearConversation() {
      this.messages = []
      this.error = null
      this.persistHistory()
    },
    async sendMessage() {
      if (!this.canSend) {
        return
      }

      const workspaceStore = useWorkspaceContextStore()
      const userStore = useUserStore()
      const normalizedUserId = normalizeUserId(userStore.userId)
      if (normalizedUserId !== this.historyUserId) {
        this.hydrateHistoryForCurrentUser()
      }
      if (!workspaceStore.hasWorkspace) {
        this.error = new Error('Select a workspace before chatting.')
        return
      }

      const content = this.inputValue.trim()
      this.inputValue = ''
      this.error = null

      const previousTurns = this.messages
        .filter((message) => (message.role === 'user' || message.role === 'assistant') && message.isError !== true)
        .slice(-MAX_HISTORY_MESSAGES)
        .map((message) => ({ role: message.role, content: message.content }))

      const userMessage = createMessage('user', content)
      const assistantMessage = createMessage('assistant', '')

      this.messages = [...this.messages, userMessage, assistantMessage]
      this.streaming = true
      this.persistHistory()

      try {
        await queryApi.streamQuery({
          payload: {
            query: content,
            mode: 'mix',
            response_type: 'Multiple Paragraphs',
            ...(previousTurns.length ? { conversation_history: previousTurns } : {}),
          },
          headers: {
            'X-Workspace': workspaceStore.workspaceId,
          },
          onChunk: (chunk) => {
            assistantMessage.content += chunk
            this.messages = [...this.messages]
            this.persistHistory()
          },
        })
      } catch (error) {
        console.error('Streaming query failed', error)
        this.error = error
        assistantMessage.content = error?.message || 'Something went wrong.'
        assistantMessage.isError = true
        this.messages = [...this.messages]
      } finally {
        this.streaming = false
        this.persistHistory()
      }
    },
  },
})
