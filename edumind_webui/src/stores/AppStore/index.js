import { defineStore } from 'pinia'

export const useAppStore = defineStore('app', {
  state: () => ({
    appName: 'EduMind Question Studio',
    tagline: 'Craft rich assessments with adaptive AI support.',
    organization: 'EduMind',
    themePreference: 'light',
    maxCanvasTabs: 5,
  }),
  getters: {
    organizationInitials: (state) =>
      state.organization
        .split(' ')
        .map((chunk) => chunk.charAt(0))
        .join('')
        .toUpperCase(),
  },
  actions: {
    toggleThemePreference() {
      this.themePreference = this.themePreference === 'light' ? 'dark' : 'light'
    },
  },
})
