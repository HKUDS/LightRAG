// src/stores/user.ts
import { defineStore } from 'pinia'
import {
  fetchCurrentUser,
  signIn as loginRequest,
  signOut as logoutRequest,
} from '@/api/auth'

export const useUserStore = defineStore('user', {
  state: () => ({
    id: null,
    user: null,
    loading: false,
    error: null,
    hydrated: false,
  }),
  getters: {
    isAuthenticated: (s) => Boolean(s.user),
    userId: (s) => s.user?.id || '',
    displayName: (s) => s.user?.full_name || s.user?.username || '',
    role: (s) => s.user?.role || 'guest',
  },
  actions: {
    async hydrate() {
      if (this.hydrated || this.loading) return
      this.loading = true
      try {
        const { user } = await fetchCurrentUser()
        this.user = user
        this.error = null
      } catch (e) {
        this.user = null
        this.error = e
      } finally {
        this.hydrated = true
        this.loading = false
      }
    },
    async signIn({ username, password }) {
      this.loading = true
      this.error = null
      try {
        const { user } = await loginRequest({ username, password })
        this.user = user
        this.hydrated = true
        return this.user
      } catch (e) {
        this.user = null
        this.error = e
        throw e
      } finally {
        this.loading = false
      }
    },
    async signOut() {
      try {
        await logoutRequest()
      } finally {
        this.user = null
        this.hydrated = true
        this.error = null
      }
    },
    clearAuth() {
      this.isAuthenticated = false
      this.user = null
      this.token = null
      localStorage.removeItem('auth_token')
    },
  },
})
