import { defineStore } from 'pinia'
import { useAppStore } from '../AppStore'

const createTabId = (index) => `canvas-${index}-${Math.random().toString(36).slice(2, 8)}`

const buildInitialTab = () => ({
  id: createTabId(1),
  name: 'Canvas 1',
  outputs: [],
})

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
    aiRational: 'Ohm\'s Law states that voltage equals current multiplied by resistance, forming the foundation of circuit analysis.',
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
    question: 'Explain the purpose of Bloom\'s taxonomy in instructional design.',
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

export const useHomeStore = defineStore('home', {
  state: () => {
    const firstTab = buildInitialTab()
    return {
      tools: toolCatalogue,
      expandedToolIds: toolCatalogue.slice(0, 1).map((tool) => tool.id),
      tabs: [firstTab],
      activeTabId: firstTab.id,
      tabCounter: 1,
    }
  },
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
    },
    addTab() {
      if (!this.canAddTab) {
        return false
      }

      this.tabCounter += 1
      const newTab = {
        id: createTabId(this.tabCounter),
        name: `Canvas ${this.tabCounter}`,
        outputs: [],
      }

      this.tabs = [...this.tabs, newTab]
      this.activeTabId = newTab.id
      return true
    },
    closeTab(tabId) {
      if (this.tabs.length === 1) {
        return false
      }

      const tabIndex = this.tabs.findIndex((tab) => tab.id === tabId)
      if (tabIndex === -1) {
        return false
      }

      const updatedTabs = this.tabs.filter((tab) => tab.id !== tabId)
      this.tabs = updatedTabs

      if (this.activeTabId === tabId) {
        const fallbackTab = updatedTabs[Math.max(0, tabIndex - 1)] || updatedTabs[0] || null
        this.activeTabId = fallbackTab ? fallbackTab.id : ''
      }
      return true
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
      this.tabs = [buildInitialTab()]
      this.activeTabId = this.tabs[0].id
      this.tabCounter = 1
    },
    appendOutputs(outputs) {
      if (!Array.isArray(outputs) || outputs.length === 0) {
        return
      }

      if (!this.activeTab) {
        this.resetWorkspace()
      }

      if (this.activeTab) {
        this.activeTab.outputs = [...this.activeTab.outputs, ...outputs]
      }
    },
    appendOutputs(outputs) {
      if (!Array.isArray(outputs) || outputs.length === 0) {
        return
      }

      if (!this.activeTab) {
        this.resetWorkspace()
      }

      if (this.activeTab) {
        this.activeTab.outputs = [...this.activeTab.outputs, ...outputs]
      }
    },
    executeTool(toolId) {
      if (!this.activeTab) {
        this.resetWorkspace()
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

      if (this.activeTab) {
        this.activeTab.outputs = [...this.activeTab.outputs, output]
      }

      return output
    },
  },
})
