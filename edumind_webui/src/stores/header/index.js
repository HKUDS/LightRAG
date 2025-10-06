import { defineStore } from 'pinia'

export const useHeaderStore = defineStore('header', {
  state: () => ({
    title: '',
    description: '',
    showBack: false,
    actions: [],
  }),
  actions: {
    setHeader({ title = '', description = '', showBack = false, actions = [] } = {}) {
      this.title = title
      this.description = description
      this.showBack = showBack
      this.actions = Array.isArray(actions) ? actions : []
    },
    resetHeader() {
      this.title = ''
      this.description = ''
      this.showBack = false
      this.actions = []
    },
  },
})
