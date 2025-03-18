import { NavigateFunction } from 'react-router-dom';
import { useAuthStore, useBackendState } from '@/stores/state';
import { useGraphStore } from '@/stores/graph';
import { useSettingsStore } from '@/stores/settings';

class NavigationService {
  private navigate: NavigateFunction | null = null;

  setNavigate(navigate: NavigateFunction) {
    this.navigate = navigate;
  }

  /**
   * Reset all application state to ensure a clean environment.
   * This function should be called when:
   * 1. User logs out
   * 2. Authentication token expires
   * 3. Direct access to login page
   */
  resetAllApplicationState() {
    console.log('Resetting all application state...');
    
    // Clear authentication state
    localStorage.removeItem('LIGHTRAG-API-TOKEN');
    sessionStorage.clear();
    useAuthStore.getState().logout();
    
    // Reset graph state
    const graphStore = useGraphStore.getState();
    graphStore.reset();
    graphStore.setGraphDataFetchAttempted(false);
    graphStore.setLabelsFetchAttempted(false);
    
    // Reset backend state
    useBackendState.getState().clear();
    
    // Reset retrieval history while preserving other user preferences
    useSettingsStore.getState().setRetrievalHistory([]);
  }

  /**
   * Navigate to login page after resetting application state
   * to ensure a clean environment for the next session
   */
  navigateToLogin() {
    // Reset state before navigation
    this.resetAllApplicationState();
    
    if (this.navigate) {
      this.navigate('/login');
    }
  }
}

export const navigationService = new NavigationService();
