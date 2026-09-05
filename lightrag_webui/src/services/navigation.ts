import { NavigateFunction } from 'react-router-dom';
import { useAuthStore, useBackendState } from '@/stores/state';

/**
 * Per-entry navigation policy, configured EXPLICITLY by each entry's
 * bootstrap module before the app renders (see src/bootstrap/*). This is the
 * one wiring point where an entry difference would fail silently — a shared
 * layer that "falls back to the admin default" would only misbehave on the
 * workspace entry's 401 path — so an unconfigured service throws instead of
 * guessing.
 *
 * The navigation core must stay import-light: it must NOT import
 * `stores/graph` (that would drag graphology into every entry that can issue
 * a query) nor any history store. Entry-specific cleanup is injected via
 * `resetAdapters` / `clearRetrievalHistory`.
 */
export interface NavigationEntryConfig {
  /** SPA route of this entry's unauthenticated default page
   * ('/welcome' for the workspace entry, '/login' for the admin entry). */
  unauthenticatedRoute: string;
  /** Entry-specific state resets run on logout/401 (e.g. the admin entry's
   * graph-store cleanup). Registered by the entry bootstrap. */
  resetAdapters?: Array<() => void>;
  /** Clears THIS entry's own retrieval history. */
  clearRetrievalHistory: () => void;
}

class NavigationService {
  private navigate: NavigateFunction | null = null;
  private config: NavigationEntryConfig | null = null;

  setNavigate(navigate: NavigateFunction) {
    this.navigate = navigate;
  }

  /** Called once by the entry bootstrap, before the app renders. */
  configureEntry(config: NavigationEntryConfig) {
    this.config = config;
  }

  private requireConfig(): NavigationEntryConfig {
    if (!this.config) {
      // Never silently fall back to the admin defaults — see module docs.
      throw new Error(
        'navigationService is not configured: the entry bootstrap must call configureEntry() before navigation is used'
      );
    }
    return this.config;
  }

  /**
   * Reset all application state to ensure a clean environment.
   * This function should be called when:
   * 1. User logs out
   * 2. Authentication token expires
   * 3. Direct access to the unauthenticated default page
   *
   * @param preserveHistory If true, chat history will be preserved. Default is false.
   */
  resetAllApplicationState(preserveHistory = false) {
    console.log('Resetting all application state...');

    const config = this.requireConfig();

    // Entry-specific resets (e.g. graph store cleanup on the admin entry).
    for (const reset of config.resetAdapters ?? []) {
      try {
        reset();
      } catch (error) {
        console.error('Entry reset adapter failed:', error);
      }
    }

    // Reset backend state
    useBackendState.getState().clear();

    // Reset retrieval history message only if preserveHistory is false
    if (!preserveHistory) {
      config.clearRetrievalHistory();
    }

    // Clear authentication state
    sessionStorage.clear();
  }

  /**
   * Navigate to this entry's unauthenticated default page and reset
   * application state. The admin entry lands on its login page, the
   * workspace entry on its welcome page — entry identity itself is preserved
   * by the mount path (HashRouter navigation never rewrites the path).
   */
  navigateToUnauthenticated() {
    if (!this.navigate) {
      console.error('Navigation function not set');
      return;
    }

    const config = this.requireConfig();

    // Store current username before logout for comparison during next login
    const currentUsername = useAuthStore.getState().username;
    if (currentUsername) {
      localStorage.setItem('LIGHTRAG-PREVIOUS-USER', currentUsername);
    }

    // Reset application state but preserve history
    // History will be cleared on next login if the user changes
    this.resetAllApplicationState(true);
    useAuthStore.getState().logout();

    this.navigate(config.unauthenticatedRoute);
  }

  navigateToHome() {
    if (!this.navigate) {
      console.error('Navigation function not set');
      return;
    }

    this.navigate('/');
  }

  /** Test hook: drop the entry configuration. */
  resetForTests() {
    this.config = null;
    this.navigate = null;
  }
}

export const navigationService = new NavigationService();
