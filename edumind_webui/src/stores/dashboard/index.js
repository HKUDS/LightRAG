import { defineStore } from 'pinia'
import { workspacesApi } from '@/api'
import { useUserStore } from '../user'

export const useDashboardStore = defineStore('dashboard', {
  state: () => ({
    workspaces: [],
    loading: false,
    error: null,
    hydrated: false,
  }),
  getters: {
    hasWorkspaces: (state) => state.workspaces.length > 0,
    primaryWorkspaceName: (state) => (state.workspaces[0]?.name ? state.workspaces[0].name : 'Workspace'),
  },
  actions: {
    async initialise(options = {}) {
      if (this.loading) {
        return
      }

      if (this.hydrated && !options.force) {
        return
      }

      const userStore = useUserStore()
      const userId = userStore.userId

      this.loading = true
      this.error = null

      try {
        const response = await workspacesApi.listWorkspaces({
          query: {
            user_id: userId,
            limit: 30,
            sort: '-created_at',
          },
        })

        console.info('Fetched workspaces:', response)

        const workspaces = response?.projects || response?.workspaces || []
        this.workspaces = workspaces.map((workspace) => ({
          id: workspace.id,
          name: workspace.name || 'Untitled workspace',
          createdAt: workspace.created_at,
          instructions: workspace.instructions || 'No instructions provided.',
        }))
        this.hydrated = true
      } catch (error) {
        console.error('Failed to load workspaces', error)
        this.error = error
      } finally {
        this.loading = false
      }
    },
  },
})
