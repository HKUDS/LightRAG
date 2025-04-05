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

    // Reset graph state
    const graphStore = useGraphStore.getState();
    const sigma = graphStore.sigmaInstance;
    graphStore.reset();
    graphStore.setGraphDataFetchAttempted(false);
    graphStore.setLabelsFetchAttempted(false);
    graphStore.setSigmaInstance(null);
    graphStore.setIsFetching(false); // Reset isFetching state to prevent data loading issues

    // Reset backend state
    useBackendState.getState().clear();

    // Reset retrieval history message while preserving other user preferences
    useSettingsStore.getState().setRetrievalHistory([]);

    // Clear authentication state
    sessionStorage.clear();

    if (sigma) {
      sigma.getGraph().clear();
      sigma.kill();
      useGraphStore.getState().setSigmaInstance(null);
    }
  }

  /**
   * Handle direct access to login page
   * @returns true if it's a direct access, false if navigated from another page
   */
  handleDirectLoginAccess() {
    const isDirectAccess = !document.referrer;
    if (isDirectAccess) {
      this.resetAllApplicationState();
    }
    return isDirectAccess;
  }

  /**
   * Navigate to login page and reset application state
   * @param skipReset whether to skip state reset (used for direct access scenario where reset is already handled)
   */
  navigateToLogin() {
    if (!this.navigate) {
      console.error('Navigation function not set');
      return;
    }

    this.resetAllApplicationState();
    useAuthStore.getState().logout();

    this.navigate('/login');
  }

  navigateToHome() {
    if (!this.navigate) {
      console.error('Navigation function not set');
      return;
    }

    this.navigate('/');
  }
}

export const navigationService = new NavigationService();
