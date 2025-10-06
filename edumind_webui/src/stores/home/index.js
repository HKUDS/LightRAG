import { defineStore } from 'pinia'
import { sessionsApi, questionsApi } from '@/api'
import { useAppStore } from '../AppStore'
import { useWorkspaceContextStore } from '../workspaceContext'
import { useUserStore } from '../user'

const pickRandom = (items) => items[Math.floor(Math.random() * items.length)]

const mcqPool = [
  {
    question: 'Which data structure follows the First-In-First-Out (FIFO) principle?',
    options: ['Stack', 'Queue', 'Tree', 'Graph'],
    correctOptions: [1],
    difficultyLevel: 'medium',
    aiRational: 'Queues process elements in the order they are inserted, making them ideal for task scheduling scenarios.',
    source: 'Computer Science Fundamentals',
    tag: 'Data Structures',
  },
  {
    question: 'What is the capital city of Japan?',
    options: ['Seoul', 'Tokyo', 'Kyoto', 'Osaka'],
    correctOptions: [1],
    difficultyLevel: 'easy',
    aiRational: 'Tokyo has been the political and economic center of Japan since 1868, making it the capital city.',
    source: 'World Geography Reference',
    tag: 'Geography',
  },
  {
    question: 'Which law explains the relationship between voltage, current, and resistance?',
    options: ["Faraday's Law", "Newton's Law", "Ohm's Law", "Maxwell's Equation"],
    correctOptions: [2],
    difficultyLevel: 'medium',
    aiRational: "Ohm's Law states that voltage equals current multiplied by resistance, forming the foundation of circuit analysis.",
    source: 'Physics Core Concepts',
    tag: 'Physics',
  },
]

const assignmentPool = [
  {
    question: 'Design a project brief for implementing a flipped classroom model in a high school setting.',
    difficultyLevel: 'hard',
    aiRational: 'This assignment evaluates strategic planning, curriculum design, and an understanding of pedagogical frameworks.',
    source: 'Instructional Design Toolkit',
    tag: 'Education',
  },
  {
    question: 'Compose a reflection on how renewable energy adoption can transform urban infrastructures over the next decade.',
    difficultyLevel: 'medium',
    aiRational: 'Encourages learners to synthesize environmental science with socio-economic impacts for future-forward thinking.',
    source: 'Sustainable Cities Program',
    tag: 'Environmental Science',
  },
  {
    question: 'Outline a user journey map for a mental wellness mobile application aimed at university students.',
    difficultyLevel: 'medium',
    aiRational: 'Focuses on empathy mapping, UX research, and translating insights into a product strategy.',
    source: 'Digital Product Studio',
    tag: 'Design Thinking',
  },
]

const projectPool = [
  {
    question: 'Create a five-question formative quiz that assesses understanding of the water cycle, including misconceptions.',
    difficultyLevel: 'easy',
    aiRational: 'Highlights the ability to scaffold learning checkpoints aligned with scientific literacy.',
    source: 'Middle School Science Lab',
    tag: 'Science',
  },
  {
    question: 'Draft a collaborative workshop outline for onboarding new remote employees in a global organization.',
    difficultyLevel: 'medium',
    aiRational: 'Targets facilitation skills, asynchronous engagement, and inclusive team culture design.',
    source: 'People Operations Playbook',
    tag: 'People & Culture',
  },
]

const flashcardPool = [
  {
    question: 'Define neuroplasticity and provide an example of its role in learning resilience.',
    difficultyLevel: 'medium',
    aiRational: 'Reinforces conceptual clarity with practical application, suited for rapid revision.',
    source: 'Cognitive Science Primer',
    tag: 'Neuroscience',
  },
  {
    question: "Explain the purpose of Bloom's taxonomy in instructional design.",
    difficultyLevel: 'easy',
    aiRational: 'Supports quick recall of core pedagogical frameworks for lesson planning.',
    source: 'Teaching Methodologies Handbook',
    tag: 'Education',
  },
]

const toolCatalogue = [
  {
    id: 'mcq-generator',
    name: 'MCQ Generator',
    description: 'Create structured multiple-choice questions enriched with rationales.',
    icon: 'mdi-help-box-multiple-outline',
    quickAction: 'Generate MCQ',
  },
  {
    id: 'assignment-generator',
    name: 'Assignment Builder',
    description: 'Draft open-ended tasks that encourage critical thinking and synthesis.',
    icon: 'mdi-clipboard-text-outline',
    quickAction: 'Create Assignment',
  },
  {
    id: 'quiz-builder',
    name: 'Quiz Composer',
    description: 'Assemble thematic quiz prompts for rapid knowledge checks.',
    icon: 'mdi-layers-triple-outline',
    quickAction: 'Build Quiz Prompt',
  },
  {
    id: 'flashcard-creator',
    name: 'Flashcard Studio',
    description: 'Spin up concise prompts ideal for spaced repetition.',
    icon: 'mdi-card-text-outline',
    quickAction: 'Craft Flashcard',
  },
]

const createOutputId = () => `output-${Math.random().toString(36).slice(2, 10)}`

const buildMcqPayload = () => {
  const sample = pickRandom(mcqPool)
  return {
    id: createOutputId(),
    type: 'mcq',
    ...sample,
  }
}

const buildAssignmentPayload = (pool = assignmentPool) => {
  const sample = pickRandom(pool)
  return {
    id: createOutputId(),
    type: 'assignment',
    options: [],
    correctOptions: [],
    ...sample,
  }
}

const extractCanvasIndex = (name = '') => {
  const match = /canvas\s+(\d+)$/i.exec(name.trim())
  if (!match) {
    return null
  }
  const value = Number(match[1])
  return Number.isNaN(value) ? null : value
}

const computeTabCounterFromTabs = (tabs = []) => {
  return tabs.reduce((currentMax, tab) => {
    const candidate = extractCanvasIndex(tab.name)
    if (!candidate) {
      return currentMax
    }
    return Math.max(currentMax, candidate)
  }, 0)
}

const buildProjectHeaders = (projectId) => {
  if (!projectId) {
    return {}
  }
  return { 'X-Workspace': projectId }
}

const mapCanvasToTab = (canvas, index) => ({
  id: canvas?.id || '',
  name: canvas?.name || canvas?.topic || `Canvas ${index + 1}`,
  outputs: [],
  meta: canvas || null,
})

const normaliseStringArray = (value) => {
  if (Array.isArray(value)) {
    return value
  }
  if (value && typeof value === 'object') {
    return Object.values(value)
  }
  if (typeof value === 'string') {
    return [value]
  }
  return []
}

const normaliseOptions = (value) => normaliseStringArray(value).map((option) => String(option))

const normaliseCorrectAnswers = (value, optionCount) => {
  const entries = Array.isArray(value)
    ? value
    : value && typeof value === 'object'
      ? Array.isArray(value.indices)
        ? value.indices
        : Object.values(value)
      : []

  const unique = new Set()
  entries.forEach((entry) => {
    const index = Number(entry)
    if (Number.isInteger(index) && index >= 0 && (optionCount === undefined || index < optionCount)) {
      unique.add(index)
    }
  })
  return Array.from(unique)
}

const mapQuestionRecordToOutput = (question) => {
  if (!question) {
    return null
  }

  const options = normaliseOptions(question.options)
  const correctOptions = normaliseCorrectAnswers(question.correct_answers, options.length)
  const tagList = normaliseStringArray(question.tags)

  return {
    id: question.id || `question-${Math.random().toString(36).slice(2, 10)}`,
    type: question.type || 'mcq',
    question: question.question_text || 'Generated question',
    options,
    correctOptions,
    difficultyLevel: question.difficulty_level || 'medium',
    aiRational: question.ai_rational || question.rationale || 'Generated by AI.',
    source: question.source || 'AI Generator',
    tag: tagList.length ? tagList.join(', ') : 'AI Generated',
    variants: Array.isArray(question.variants) ? question.variants : [],
    meta: question,
  }
}

export const useHomeStore = defineStore('home', {
  state: () => ({
    tools: toolCatalogue,
    expandedToolIds: toolCatalogue.slice(0, 1).map((tool) => tool.id),
    tabs: [],
    activeTabId: '',
    tabCounter: 0,
    loading: false,
    error: null,
    creatingCanvas: false,
    deletingCanvasIds: [],
    currentProjectId: '',
    canvasOutputStatus: {},
  }),
  getters: {
    activeTab(state) {
      if (!state.activeTabId && state.tabs.length) {
        return state.tabs[0]
      }
      return state.tabs.find((tab) => tab.id === state.activeTabId) || null
    },
    activeTabOutputs() {
      return this.activeTab ? this.activeTab.outputs : []
    },
    hasActiveOutputs() {
      return this.activeTabOutputs.length > 0
    },
    canAddTab() {
      const appStore = useAppStore()
      return this.tabs.length < appStore.maxCanvasTabs
    },
    isDeletingCanvas: (state) => (canvasId) => state.deletingCanvasIds.includes(canvasId),
    isCanvasLoading: (state) => (canvasId) => Boolean(state.canvasOutputStatus[canvasId]?.loading),
    hasHydratedCanvas: (state) => (canvasId) => Boolean(state.canvasOutputStatus[canvasId]?.loaded),
  },
  actions: {
    initialise() {
      if (!this.activeTabId && this.tabs.length) {
        this.activeTabId = this.tabs[0].id
      }
    },
    setExpandedToolIds(ids) {
      this.expandedToolIds = Array.from(new Set(ids))
    },
    toggleTool(toolId) {
      if (this.expandedToolIds.includes(toolId)) {
        this.expandedToolIds = this.expandedToolIds.filter((id) => id !== toolId)
      } else {
        this.expandedToolIds = [...this.expandedToolIds, toolId]
      }
    },
    setActiveTab(tabId) {
      this.activeTabId = tabId
      if (tabId) {
        this.loadCanvasOutputs({ tabId }).catch(() => {})
      }
    },
    updateCanvasOutputStatus(canvasId, patch = {}) {
      if (!canvasId) {
        return
      }
      const nextStatus = { ...this.canvasOutputStatus }
      const previous = nextStatus[canvasId] || {}
      nextStatus[canvasId] = {
        ...previous,
        ...patch,
      }
      this.canvasOutputStatus = nextStatus
    },
    applyOutputsToTab(canvasId, outputs = []) {
      if (!canvasId) {
        return
      }
      const target = this.tabs.find((tab) => tab.id === canvasId)
      if (!target) {
        return
      }
      target.outputs = outputs
    },
    setCanvasCollection(canvases = []) {
      const mappedTabs = canvases.map((canvas, index) => mapCanvasToTab(canvas, index))
      this.tabs = mappedTabs
      this.tabCounter = computeTabCounterFromTabs(mappedTabs)

      const nextStatus = {}
      mappedTabs.forEach((tab) => {
        if (this.canvasOutputStatus[tab.id]) {
          nextStatus[tab.id] = this.canvasOutputStatus[tab.id]
        }
      })
      this.canvasOutputStatus = nextStatus

      if (!mappedTabs.length) {
        this.activeTabId = ''
        return
      }
      const currentTab = mappedTabs.find((tab) => tab.id === this.activeTabId)
      const nextActiveId = currentTab ? currentTab.id : mappedTabs[0].id

      if (!nextActiveId) {
        this.activeTabId = ''
        return
      }

      if (this.activeTabId === nextActiveId) {
        this.loadCanvasOutputs({ tabId: nextActiveId }).catch(() => {})
      } else {
        this.setActiveTab(nextActiveId)
      }
    },
    async loadCanvasOutputs({ tabId, force = false } = {}) {
      const workspaceStore = useWorkspaceContextStore()
      const projectId = workspaceStore.workspaceId || this.currentProjectId
      const targetId = tabId || this.activeTabId

      if (!targetId) {
        return
      }

      const status = this.canvasOutputStatus[targetId]
      if (!force) {
        if (status?.loading || status?.loaded) {
          return
        }
      } else if (status?.loading) {
        return
      }

      this.updateCanvasOutputStatus(targetId, { loading: true, loaded: false, error: null })

      try {
        const response = await questionsApi.listQuestions({
          query: {
            session_id: targetId,
            project_id: projectId,
            isArchived: false,
            page: 1,
            pageSize: 50,
            sort: 'updated_at',
            order: 'desc',
          },
          headers: buildProjectHeaders(projectId),
        })

        const questions = Array.isArray(response?.questions)
          ? response.questions
          : Array.isArray(response?.data?.questions)
            ? response.data.questions
            : []

        const outputs = questions
          .map((question) => mapQuestionRecordToOutput(question))
          .filter((output) => output !== null)

        const currentTab = this.tabs.find((tab) => tab.id === targetId)
        const hasExistingOutputs = Array.isArray(currentTab?.outputs) && currentTab.outputs.length > 0

        if (hasExistingOutputs && outputs.length === 0 && !force) {
          this.updateCanvasOutputStatus(targetId, { loading: false, loaded: true, error: null })
          return
        }

        this.applyOutputsToTab(targetId, outputs)
        this.updateCanvasOutputStatus(targetId, { loading: false, loaded: true, error: null })
      } catch (error) {
        console.error('Failed to load canvas outputs', error)
        this.error = error
        this.updateCanvasOutputStatus(targetId, { loading: false, loaded: false, error })
        throw error
      }
    },
    async loadProjectCanvases({ projectId, force = false } = {}) {
      if (this.loading) {
        return
      }

      const workspaceStore = useWorkspaceContextStore()
      const userStore = useUserStore()
      const appStore = useAppStore()

      const targetProjectId = projectId || workspaceStore.workspaceId
      if (!targetProjectId) {
        this.currentProjectId = ''
        this.setCanvasCollection([])
        return
      }

      if (!force && this.currentProjectId === targetProjectId && this.tabs.length) {
        return
      }

      this.loading = true
      this.error = null
      this.currentProjectId = targetProjectId

      try {
        const response = await sessionsApi.listCanvas({
          projectId: targetProjectId,
          userId: userStore.userId,
          limit: appStore.maxCanvasTabs,
          headers: buildProjectHeaders(targetProjectId),
        })

        const canvases = Array.isArray(response?.canvases) ? response.canvases : []
        this.setCanvasCollection(canvases)
      } catch (error) {
        console.error('Failed to load canvases', error)
        this.error = error
        this.setCanvasCollection([])
      } finally {
        this.loading = false
      }
    },
    async ensureActiveCanvas() {
      if (this.activeTab) {
        return this.activeTab
      }

      if (this.currentProjectId) {
        await this.loadProjectCanvases({ projectId: this.currentProjectId, force: true })
        if (this.activeTab) {
          return this.activeTab
        }
      }

      if (!this.canAddTab) {
        return null
      }

      try {
        const tab = await this.addTab()
        return tab
      } catch (error) {
        console.error('Failed to ensure active canvas', error)
        return null
      }
    },
    async addTab({ name } = {}) {
      if (!this.canAddTab) {
        return false
      }

      const workspaceStore = useWorkspaceContextStore()
      const userStore = useUserStore()

      const projectId = workspaceStore.workspaceId || this.currentProjectId
      if (!projectId) {
        this.error = new Error('Select a project before creating a canvas.')
        return false
      }

      const headers = buildProjectHeaders(projectId)
      const trimmedName = typeof name === 'string' ? name.trim() : ''
      const fallbackIndex = this.tabCounter + 1
      const topic = trimmedName || `Canvas ${fallbackIndex}`

      this.creatingCanvas = true
      this.error = null

      try {
        const response = await sessionsApi.createSession({
          payload: {
            user_id: userStore.userId,
            project_id: projectId,
            topic,
          },
          headers,
        })

        const sessionId = response?.session_id
        if (!sessionId) {
          throw new Error('The server did not return a session identifier.')
        }

        let canvasMeta = null
        try {
          const fetched = await sessionsApi.fetchCanvas({ id: sessionId, headers })
          canvasMeta = fetched?.canvas || null
        } catch (fetchError) {
          console.warn('Failed to fetch canvas metadata', fetchError)
        }

        const mappedTab = mapCanvasToTab(
          canvasMeta || {
            id: sessionId,
            name: topic,
            topic,
            userId: userStore.userId,
            projectId,
          },
          this.tabs.length
        )

        mappedTab.id = sessionId

        const derivedIndex = extractCanvasIndex(mappedTab.name) || fallbackIndex
        this.tabCounter = Math.max(this.tabCounter, derivedIndex)

        this.tabs = [...this.tabs, mappedTab]
        this.currentProjectId = projectId
        this.setActiveTab(mappedTab.id)
        return mappedTab
      } catch (error) {
        console.error('Failed to create canvas', error)
        this.error = error
        throw error
      } finally {
        this.creatingCanvas = false
      }
    },
    async closeTab(tabId) {
      if (!tabId) {
        return false
      }

      const existingTab = this.tabs.find((tab) => tab.id === tabId)
      if (!existingTab) {
        return false
      }

      const workspaceStore = useWorkspaceContextStore()
      const projectId = workspaceStore.workspaceId || this.currentProjectId
      const headers = buildProjectHeaders(projectId)

      this.deletingCanvasIds = Array.from(new Set([...this.deletingCanvasIds, tabId]))
      this.error = null

      try {
        await sessionsApi.deleteSession({ id: tabId, headers })

        this.tabs = this.tabs.filter((tab) => tab.id !== tabId)
        const nextStatus = { ...this.canvasOutputStatus }
        delete nextStatus[tabId]
        this.canvasOutputStatus = nextStatus
        if (this.activeTabId === tabId) {
          const fallbackId = this.tabs[0]?.id || ''
          if (fallbackId) {
            this.setActiveTab(fallbackId)
          } else {
            this.activeTabId = ''
          }
        }
        this.tabCounter = computeTabCounterFromTabs(this.tabs)
        return true
      } catch (error) {
        console.error('Failed to delete canvas', error)
        this.error = error
        throw error
      } finally {
        this.deletingCanvasIds = this.deletingCanvasIds.filter((id) => id !== tabId)
      }
    },
    discardActiveTabOutputs() {
      if (!this.activeTab) {
        return
      }
      this.activeTab.outputs = []
    },
    deleteOutputById(outputId) {
      const tab = this.activeTab
      if (!tab) {
        return
      }
      tab.outputs = tab.outputs.filter((output) => output.id !== outputId)
    },
    resetWorkspace() {
      this.setCanvasCollection([])
      this.currentProjectId = ''
      this.tabCounter = 0
    },
    async appendOutputs(outputs) {
      if (!Array.isArray(outputs) || outputs.length === 0) {
        return
      }

      const activeTab = await this.ensureActiveCanvas()
      if (!activeTab) {
        throw new Error('No canvas available to accept generated outputs.')
      }

      activeTab.outputs = [...activeTab.outputs, ...outputs]
      this.updateCanvasOutputStatus(activeTab.id, { loaded: true, loading: false, error: null })
    },
    async executeTool(toolId) {
      try {
        const activeTab = await this.ensureActiveCanvas()
        if (!activeTab) {
          return null
        }

        const generators = {
          'mcq-generator': () => buildMcqPayload(),
          'assignment-generator': () => buildAssignmentPayload(),
          'quiz-builder': () => {
            const choice = Math.random() > 0.5 ? buildMcqPayload() : buildAssignmentPayload(projectPool)
            return choice
          },
          'flashcard-creator': () => buildAssignmentPayload(flashcardPool),
        }

        const generator = generators[toolId] || (() => buildMcqPayload())
        const output = generator()
        activeTab.outputs = [...activeTab.outputs, output]
        return output
      } catch (error) {
        console.error('Failed to execute tool', error)
        this.error = error
        return null
      }
    },
  },
})
