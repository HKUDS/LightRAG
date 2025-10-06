import { defineStore } from 'pinia'
import { queryApi } from '@/api'
import { useWorkspaceContextStore } from '../workspaceContext'

let messageCounter = 0

const createMessage = (role, content = '') => ({
  id: `msg-${++messageCounter}`,
  role,
  content,
})

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [],
    inputValue: '',
    streaming: false,
    error: null,
  }),
  getters: {
    hasMessages: (state) => state.messages.length > 0,
    canSend: (state) => state.inputValue.trim().length > 0 && !state.streaming,
  },
  actions: {
    setInput(value) {
      this.inputValue = value
    },
    clearConversation() {
      this.messages = []
      this.error = null
    },
    async sendMessage() {
      if (!this.canSend) {
        return
      }

      const workspaceStore = useWorkspaceContextStore()
      if (!workspaceStore.hasWorkspace) {
        this.error = new Error('Select a workspace before chatting.')
        return
      }

      const content = this.inputValue.trim()
      this.inputValue = ''
      this.error = null

      const userMessage = createMessage('user', content)
      const assistantMessage = createMessage('assistant', '')

      this.messages = [...this.messages, userMessage, assistantMessage]
      this.streaming = true

      try {
        await queryApi.streamQuery({
          payload: {
            query: content,
            mode: 'mix',
            response_type: 'Multiple Paragraphs',
          },
          headers: {
            'X-Workspace': workspaceStore.workspaceId,
          },
          onChunk: (chunk) => {
            assistantMessage.content += chunk
            this.messages = [...this.messages]
          },
        })
      } catch (error) {
        console.error('Streaming query failed', error)
        this.error = error
      } finally {
        this.streaming = false
      }
    },
  },
})
