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
    async createWorkspace({ name, instructions } = {}) {
      const trimmedName = typeof name === 'string' ? name.trim() : ''
      if (!trimmedName) {
        throw new Error('Workspace name is required.')
      }

      const payload = {
        name: trimmedName,
        user_id: useUserStore().userId,
      }
      if (instructions && instructions.trim()) {
        payload.instructions = instructions.trim()
      }

      try {
        const response = await workspacesApi.createWorkspace({ payload })
        const createdProject = response?.project || response?.workspace || response
        const createdId = createdProject?.id

        await this.initialise({ force: true })

        if (createdId) {
          const fresh = this.workspaces.find((workspace) => workspace.id === createdId)
          if (fresh) {
            return fresh
          }
          return {
            id: createdId,
            name: createdProject?.name || trimmedName,
            createdAt: createdProject?.created_at || new Date().toISOString(),
            instructions: createdProject?.instructions || payload.instructions || 'No instructions provided.',
          }
        }

        return null
      } catch (error) {
        console.error('Failed to create workspace', error)
        throw error
      }
    },
  },
})
