import type { NavigateFunction } from 'react-router-dom'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore, useBackendState } from '@/stores/state'

class NavigationService {
  private navigate: NavigateFunction | null = null

  setNavigate(navigate: NavigateFunction) {
    this.navigate = navigate
  }

  /**
   * Reset all application state to ensure a clean environment.
   * This function should be called when:
   * 1. User logs out
   * 2. Authentication token expires
   * 3. Direct access to login page
   *
   * @param preserveHistory If true, chat history will be preserved. Default is false.
   */
  resetAllApplicationState(preserveHistory = false) {
    console.log('Resetting all application state...')

    // Reset graph state
    const graphStore = useGraphStore.getState()
    const sigma = graphStore.sigmaInstance
    graphStore.reset()
    graphStore.setGraphDataFetchAttempted(false)
    graphStore.setLabelsFetchAttempted(false)
    graphStore.setSigmaInstance(null)
    graphStore.setIsFetching(false) // Reset isFetching state to prevent data loading issues

    // Reset backend state
    useBackendState.getState().clear()

    // Reset retrieval history message only if preserveHistory is false
    if (!preserveHistory) {
      useSettingsStore.getState().setRetrievalHistory([])
    }

    // Clear authentication state
    sessionStorage.clear()

    if (sigma) {
      sigma.getGraph().clear()
      sigma.kill()
      useGraphStore.getState().setSigmaInstance(null)
    }
  }

  /**
   * Navigate to login page and reset application state
   */
  navigateToLogin() {
    if (!this.navigate) {
      console.error('Navigation function not set')
      return
    }

    // Store current username before logout for comparison during next login
    const currentUsername = useAuthStore.getState().username
    if (currentUsername) {
      localStorage.setItem('LIGHTRAG-PREVIOUS-USER', currentUsername)
    }

    // Reset application state but preserve history
    // History will be cleared on next login if the user changes
    this.resetAllApplicationState(true)
    useAuthStore.getState().logout()

    this.navigate('/login')
  }

  navigateToHome() {
    if (!this.navigate) {
      console.error('Navigation function not set')
      return
    }

    this.navigate('/')
  }
}

export const navigationService = new NavigationService()
