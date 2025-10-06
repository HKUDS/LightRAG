import { defineStore } from 'pinia'
import { aiApi } from '@/api'
import { useWorkspaceContextStore } from '../workspaceContext'
import { useHomeStore } from '../home'
import { useUserStore } from '../user'

const defaultState = () => ({
  topics: '',
  quantity: 5,
  difficulty: 'medium',
  allowMulti: false,
  userInstructions: '',
  userNotes: '',
  sessionId: '',
  projectId: '',
  userId: '',
  maxConcurrency: 5,
  loading: false,
  error: null,
  lastMessage: '',
  advancedOpen: false,
})

const normaliseCorrectAnswers = (options, answers) => {
  if (!Array.isArray(answers)) {
    return []
  }
  const normalised = answers
    .map((value) => {
      const num = Number(value)
      if (Number.isNaN(num)) {
        return null
      }
      if (options && options.length && num >= options.length && num - 1 >= 0) {
        return num - 1
      }
      return num
    })
    .filter((value) => value !== null)
  return [...new Set(normalised)].filter((value) => value >= 0 && value < options.length)
}

const mapQuestionToOutput = (question, fallbackDifficulty) => {
  const options = Array.isArray(question?.options) ? question.options : []
  const correctOptions = normaliseCorrectAnswers(options, question?.correct_answers || question?.correctOptions)
  const tagValue = Array.isArray(question?.tags)
    ? question.tags.join(', ')
    : question?.tag || 'AI Generated'

  return {
    id: `ai-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    type: 'mcq',
    question: question?.question_text || question?.question || 'Generated question',
    options,
    correctOptions,
    difficultyLevel: question?.difficulty_level || fallbackDifficulty,
    aiRational: question?.ai_rational || question?.rationale || '',
    source: question?.source || 'AI Generator',
    tag: tagValue,
  }
}

export const useMcqGeneratorStore = defineStore('mcqGenerator', {
  state: defaultState,
  getters: {
    canGenerate(state) {
      return state.topics.trim().length > 0 && !state.loading
    },
  },
  actions: {
    initialise() {
      const userStore = useUserStore()
      if (!this.userId) {
        this.userId = userStore.userId
      }
    },
    toggleAdvanced(open) {
      this.advancedOpen = typeof open === 'boolean' ? open : !this.advancedOpen
    },
    resetStatus() {
      this.error = null
      this.lastMessage = ''
    },
    async generate() {
      if (!this.canGenerate) {
        return
      }

      const workspaceStore = useWorkspaceContextStore()
      if (!workspaceStore.hasWorkspace) {
        this.error = new Error('Select a workspace before generating questions.')
        return
      }

      const userStore = useUserStore()
      const homeStore = useHomeStore()

      const activeCanvas = await homeStore.ensureActiveCanvas()
      if (!activeCanvas) {
        this.error = new Error('Create a canvas before generating questions.')
        return
      }

      const sessionId = activeCanvas.id || homeStore.activeTabId || 'default'
      this.sessionId = sessionId

      const payload = {
        session_id: sessionId,
        project_id: this.projectId || workspaceStore.workspaceId,
        user_id: this.userId || userStore.userId,
        topics: this.topics,
        n: this.quantity,
        difficulty: this.difficulty,
        user_instructions: this.userInstructions,
        user_notes: this.userNotes,
        max_concurrency: this.maxConcurrency,
        allow_multi: this.allowMulti,
      }

      this.loading = true
      this.error = null
      this.lastMessage = ''

      try {
        const response = await aiApi.generateQuestions({
          payload,
          headers: {
            'X-Workspace': workspaceStore.workspaceId,
          },
        })

        const questions = Array.isArray(response?.questions) ? response.questions : []

        if (!questions.length) {
          this.error = new Error('The generator returned no questions. Try different topics or settings.')
          return
        }

        const outputs = questions.map((question) => mapQuestionToOutput(question, this.difficulty))

        if (!outputs.length) {
          this.error = new Error('Unable to interpret the generated questions.')
          return
        }

        await homeStore.appendOutputs(outputs)

        this.lastMessage = response?.message || `Generated ${outputs.length} questions.`
      } catch (error) {
        console.error('Failed to generate questions', error)
        this.error = error
      } finally {
        this.loading = false
      }
    },
  },
})
