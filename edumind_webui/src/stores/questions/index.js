import { defineStore } from 'pinia'
import { questionsApi, sessionsApi } from '@/api'

const defaultFilters = () => ({
  searchQuery: '',
  type: 'all',
  difficulty: 'all',
  tag: 'all',
  workspaceId: 'all',
  sessionId: 'all',
  approved: 'all',
  archived: 'false',
})

const buildProjectHeaders = (projectId) => {
  if (!projectId || projectId === 'all') {
    return {}
  }
  return { 'X-Workspace': projectId }
}

const normaliseQuestion = (question) => {
  if (!question || typeof question !== 'object') {
    return null
  }

  const tags = Array.isArray(question.tags)
    ? question.tags.map((tag) => String(tag))
    : typeof question.tag === 'string'
      ? [question.tag]
      : []

  return {
    id: question.id,
    type: question.type || 'mcq',
    question: question.question_text || '',
    options: Array.isArray(question.options) ? question.options.map((option) => String(option)) : [],
    correct_options: Array.isArray(question.correct_answers)
      ? question.correct_answers.map((answer) => Number(answer))
      : [],
    difficulty_level: question.difficulty_level || 'medium',
    ai_rational: question.ai_rational || question.rationale || 'AI generated question.',
    source: question.source || 'AI Generator',
    tag: tags.length ? tags[0] : 'Untagged',
    tags,
    isApproved: Boolean(question.isApproved),
    isArchived: Boolean(question.isArchived),
    project_id: question.project_id || '',
    session_id: question.session_id || '',
    created_at: question.created_at || '',
    updated_at: question.updated_at || '',
  }
}

export const useQuestionsStore = defineStore('questions', {
  state: () => ({
    questions: [],
    totalQuestions: 0,
    page: 1,
    pageSize: 20,
    loading: false,
    error: null,
    filters: defaultFilters(),
    sessionsByWorkspace: {},
    sessionsLoading: false,
  }),
  getters: {
    uniqueTags(state) {
      const tagSet = new Set()
      state.questions.forEach((question) => {
        if (Array.isArray(question.tags) && question.tags.length) {
          question.tags.forEach((tag) => tagSet.add(tag))
        } else if (question.tag) {
          tagSet.add(question.tag)
        }
      })
      return Array.from(tagSet).sort((a, b) => a.localeCompare(b))
    },
    filteredQuestions(state) {
      const searchTerm = state.filters.searchQuery.trim().toLowerCase()
      return state.questions.filter((question) => {
        const matchesSearch =
          !searchTerm || question.question.toLowerCase().includes(searchTerm)

        const matchesType = state.filters.type === 'all' || question.type === state.filters.type
        const matchesDifficulty =
          state.filters.difficulty === 'all' || question.difficulty_level === state.filters.difficulty
        const matchesTag =
          state.filters.tag === 'all' || question.tag === state.filters.tag || question.tags?.includes(state.filters.tag)

        const matchesWorkspace =
          state.filters.workspaceId === 'all' || question.project_id === state.filters.workspaceId
        const matchesSession =
          state.filters.sessionId === 'all' || question.session_id === state.filters.sessionId

        const matchesApproved =
          state.filters.approved === 'all' || question.isApproved === (state.filters.approved === 'true')
        const matchesArchived =
          state.filters.archived === 'all' || question.isArchived === (state.filters.archived === 'true')

        return (
          matchesSearch &&
          matchesType &&
          matchesDifficulty &&
          matchesTag &&
          matchesWorkspace &&
          matchesSession &&
          matchesApproved &&
          matchesArchived
        )
      })
    },
    pageCount(state) {
      if (!state.totalQuestions) {
        return 1
      }
      return Math.max(1, Math.ceil(state.totalQuestions / state.pageSize))
    },
    resultsSummary() {
      const count = this.filteredQuestions.length
      if (count === 0) {
        return 'No questions match the current filters.'
      }
      const start = (this.page - 1) * this.pageSize + 1
      const end = start + count - 1
      if (this.totalQuestions) {
        return `Showing ${start}-${end} of ${this.totalQuestions} questions (filtered view).`
      }
      if (count === 1) {
        return '1 question matches the current filters.'
      }
      return `${count} questions match the current filters.`
    },
    sessionsForWorkspace(state) {
      return (workspaceId) => state.sessionsByWorkspace[workspaceId] || []
    },
  },
  actions: {
    resetFilters() {
      this.filters = defaultFilters()
      this.page = 1
    },
    setSearchQuery(value) {
      this.filters.searchQuery = value
      this.page = 1
    },
    setFilterType(value) {
      this.filters.type = value
      this.page = 1
    },
    setFilterDifficulty(value) {
      this.filters.difficulty = value
      this.page = 1
    },
    setFilterTag(value) {
      this.filters.tag = value
      this.page = 1
    },
    setFilterWorkspace(value) {
      this.filters.workspaceId = value || 'all'
      if (value === 'all') {
        this.filters.sessionId = 'all'
      }
      this.page = 1
    },
    setFilterSession(value) {
      this.filters.sessionId = value || 'all'
      this.page = 1
    },
    setFilterApproved(value) {
      this.filters.approved = value || 'all'
      this.page = 1
    },
    setFilterArchived(value) {
      this.filters.archived = value || 'all'
      this.page = 1
    },
    normaliseQuestions(data) {
      const list = Array.isArray(data) ? data : []
      return list
        .map((question) => normaliseQuestion(question))
        .filter((entry) => entry !== null)
    },
    setPage(value) {
      const parsed = Number(value)
      if (!Number.isFinite(parsed)) {
        return
      }
      const page = Math.max(1, Math.floor(parsed))
      if (page !== this.page) {
        this.page = page
      }
    },
    setPageSize(value) {
      const parsed = Number(value)
      const allowed = [20, 40, 60, 80, 100]
      if (!allowed.includes(parsed)) {
        return
      }
      if (parsed !== this.pageSize) {
        this.pageSize = parsed
        this.page = 1
      }
    },
    async fetchQuestions() {
      const {
        searchQuery,
        workspaceId,
        sessionId,
        approved,
        archived,
        type,
      } = this.filters

      const query = {
        page: this.page,
        pageSize: this.pageSize,
        sort: 'updated_at',
        order: 'desc',
        q: searchQuery || undefined,
        project_id: workspaceId !== 'all' ? workspaceId : undefined,
        session_id: sessionId !== 'all' ? sessionId : undefined,
        isApproved: approved === 'all' ? undefined : approved === 'true',
        isArchived: archived === 'all' ? undefined : archived === 'true',
        type: type !== 'all' ? type : undefined,
      }

      this.loading = true
      this.error = null

      try {
        const response = await questionsApi.listQuestions({
          query,
          headers: buildProjectHeaders(workspaceId !== 'all' ? workspaceId : undefined),
        })

        const questions = this.normaliseQuestions(response?.questions)
        this.questions = questions
        const total = Number(response?.total) || questions.length
        this.totalQuestions = total
        if (response?.page) {
          this.page = response.page
        }
        if (response?.pageSize) {
          this.pageSize = response.pageSize
        }
      } catch (error) {
        console.error('Failed to load questions', error)
        this.error = error
        this.questions = []
        this.totalQuestions = 0
      } finally {
        this.loading = false
      }
    },
    async fetchSessionsForWorkspace(workspaceId) {
      const targetId = workspaceId || 'all'
      if (targetId === 'all') {
        return []
      }

      if (Array.isArray(this.sessionsByWorkspace[targetId]) && this.sessionsByWorkspace[targetId].length > 0) {
        return this.sessionsByWorkspace[targetId]
      }

      this.sessionsLoading = true
      try {
        const response = await sessionsApi.listSessions({
          query: {
            project_id: targetId,
            limit: 100,
            sort: '-created_at',
          },
          headers: buildProjectHeaders(targetId),
        })

        const sessions = Array.isArray(response?.sessions)
          ? response.sessions.map((session) => ({
              id: session.id,
              name: session.topic || 'Untitled Canvas',
              project_id: session.project_id,
            }))
          : []

        this.sessionsByWorkspace = {
          ...this.sessionsByWorkspace,
          [targetId]: sessions,
        }

        if (this.filters.sessionId !== 'all') {
          const stillExists = sessions.some((session) => session.id === this.filters.sessionId)
          if (!stillExists) {
            this.filters.sessionId = 'all'
          }
        }

        return sessions
      } catch (error) {
        console.error('Failed to load sessions for workspace', error)
        this.error = error
        this.sessionsByWorkspace = {
          ...this.sessionsByWorkspace,
          [targetId]: [],
        }
        return []
      } finally {
        this.sessionsLoading = false
      }
    },
  },
})
