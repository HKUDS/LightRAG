import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  state: () => ({
    id: '101',
    name: 'EduMind Lead Instructor',
  }),
  getters: {
    userId: (state) => state.id,
  },
})
