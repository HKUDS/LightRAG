import { defineStore } from 'pinia'

export const useWorkspaceToolkitStore = defineStore('workspaceToolkit', {
  state: () => ({
    activeTab: 'files',
  }),
  actions: {
    setActiveTab(tab) {
      this.activeTab = tab
    },
  },
})
