import { defineStore } from 'pinia'

export const useWorkspaceContextStore = defineStore('workspaceContext', {
  state: () => ({
    workspaceId: '',
    workspaceName: '',
  }),
  getters: {
    hasWorkspace: (state) => Boolean(state.workspaceId),
  },
  actions: {
    setWorkspace({ id, name }) {
      if (id) {
        this.workspaceId = id
      }
      if (name !== undefined) {
        this.workspaceName = name
      }
    },
    resetWorkspace() {
      this.workspaceId = ''
      this.workspaceName = ''
    },
  },
})
