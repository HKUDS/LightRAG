import { defineStore } from 'pinia'

export const useQuestionsStore = defineStore('questions', {
  state: () => ({
    questions: [],
    loading: false,
    searchQuery: '',
    filterType: 'all',
    filterDifficulty: 'all',
    filterTag: 'all',
  }),
  getters: {
    uniqueTags(state) {
      const tagSet = new Set(state.questions.map((question) => question.tag).filter(Boolean))
      return Array.from(tagSet).sort((a, b) => a.localeCompare(b))
    },
    filteredQuestions(state) {
      return state.questions.filter((question) => {
        const matchesSearch = question.question
          .toLowerCase()
          .includes(state.searchQuery.trim().toLowerCase())

        const matchesType = state.filterType === 'all' || question.type === state.filterType
        const matchesDifficulty =
          state.filterDifficulty === 'all' || question.difficulty_level === state.filterDifficulty
        const matchesTag = state.filterTag === 'all' || question.tag === state.filterTag

        return matchesSearch && matchesType && matchesDifficulty && matchesTag
      })
    },
    resultsSummary() {
      const count = this.filteredQuestions.length
      if (count === 0) {
        return 'No questions match the current filters.'
      }
      if (count === 1) {
        return '1 question matches the current filters.'
      }
      return `${count} questions match the current filters.`
    },
  },
  actions: {
    setQuestions(data) {
      this.questions = Array.isArray(data) ? data : []
    },
    setLoading(status) {
      this.loading = status
    },
    setSearchQuery(value) {
      this.searchQuery = value
    },
    setFilterType(value) {
      this.filterType = value
    },
    setFilterDifficulty(value) {
      this.filterDifficulty = value
    },
    setFilterTag(value) {
      this.filterTag = value
    },
    resetFilters() {
      this.searchQuery = ''
      this.filterType = 'all'
      this.filterDifficulty = 'all'
      this.filterTag = 'all'
    },
    hydrateWithSampleData() {
      if (this.questions.length > 0) {
        return
      }

      this.setQuestions([
        {
          id: 'sample-mcq-1',
          type: 'mcq',
          question: 'Which planet is known as the Red Planet?',
          options: ['Venus', 'Mars', 'Jupiter', 'Mercury'],
          correct_options: [1],
          difficulty_level: 'easy',
          ai_rational: 'Mars appears reddish due to iron oxide on its surface, giving it the name Red Planet.',
          source: 'Solar System Essentials',
          tag: 'Astronomy',
          created_at: new Date().toISOString(),
        },
        {
          id: 'sample-assignment-1',
          type: 'assignment',
          question: 'Prepare a presentation on how climate change affects coastal cities and propose two mitigation strategies.',
          options: null,
          correct_options: null,
          difficulty_level: 'medium',
          ai_rational: 'Encourages interdisciplinary reasoning that connects environmental science with civic planning.',
          source: 'Climate Action Curriculum',
          tag: 'Environmental Studies',
          created_at: new Date().toISOString(),
        },
      ])
    },
  },
})
