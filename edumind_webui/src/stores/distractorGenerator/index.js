import { defineStore } from 'pinia'
import { aiApi } from '@/api'
import { useUserStore } from '../user'
import { useWorkspaceContextStore } from '../workspaceContext'
import { useHomeStore } from '../home'

const initialState = () => ({
  questionId: '',
  selectedQuestionText: '',
  selectionActive: false,
  instructions: '',
  loading: false,
  error: null,
  lastMessage: '',
})

export const useDistractorGeneratorStore = defineStore('distractorGenerator', {
  state: initialState,
  actions: {
    setQuestionId(value) {
      const nextValue = typeof value === 'string' ? value : ''
      const previousValue = this.questionId
      this.questionId = nextValue
      if (nextValue !== previousValue) {
        this.selectedQuestionText = ''
      }
    },
    setInstructions(value) {
      this.instructions = value
    },
    beginSelection() {
      this.selectionActive = true
      this.error = null
    },
    completeSelection({ id, text }) {
      if (!id) {
        return
      }

      this.questionId = id
      this.selectedQuestionText = text || ''
      this.selectionActive = false
      this.error = null
    },
    cancelSelection() {
      this.selectionActive = false
    },
    clearSelection() {
      this.questionId = ''
      this.selectedQuestionText = ''
      this.selectionActive = false
    },
    reset() {
      Object.assign(this, initialState())
    },
    async generate() {
      const questionId = this.questionId.trim()
      if (!questionId) {
        this.error = new Error('Enter a question ID first.')
        return
      }

      const userStore = useUserStore()
      const workspaceStore = useWorkspaceContextStore()
      const homeStore = useHomeStore()

      this.loading = true
      this.error = null
      this.lastMessage = ''
      this.selectionActive = false

      try {
        const response = await aiApi.generateQuestionVariants({
          questionId,
          payload: {
            user_id: userStore.userId,
            instructions: this.instructions.trim() || undefined,
            persist: true,
          },
          headers: workspaceStore.workspaceId
            ? { 'X-Workspace': workspaceStore.workspaceId }
            : undefined,
        })

        const variants = Array.isArray(response?.variants) ? response.variants : []
        const applied = homeStore.applyVariantsToQuestion(questionId, variants)

        if (!applied) {
          this.error = new Error('Unable to find that question on the current canvas.')
          return
        }

        this.lastMessage = response?.message || `Generated ${variants.length} variant sets.`
      } catch (error) {
        console.error('Failed to generate distractors', error)
        this.error = error
      } finally {
        this.loading = false
      }
    },
  },
})
