import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { checkHealth, LightragStatus } from '@/api/lightrag'

interface BackendState {
  health: boolean
  message: string | null
  messageTitle: string | null
  status: LightragStatus | null
  lastCheckTime: number
  pipelineBusy: boolean

  check: () => Promise<boolean>
  clear: () => void
  setErrorMessage: (message: string, messageTitle: string) => void
  setPipelineBusy: (busy: boolean) => void
}

interface AuthState {
  isAuthenticated: boolean;
  isGuestMode: boolean;  // Add guest mode flag
  coreVersion: string | null;
  apiVersion: string | null;
  username: string | null; // login username

  login: (token: string, isGuest?: boolean, coreVersion?: string | null, apiVersion?: string | null) => void;
  logout: () => void;
  setVersion: (coreVersion: string | null, apiVersion: string | null) => void;
}

const useBackendStateStoreBase = create<BackendState>()((set) => ({
  health: true,
  message: null,
  messageTitle: null,
  lastCheckTime: Date.now(),
  status: null,
  pipelineBusy: false,

  check: async () => {
    const health = await checkHealth()
    if (health.status === 'healthy') {
      // Update version information if health check returns it
      if (health.core_version || health.api_version) {
        useAuthStore.getState().setVersion(
          health.core_version || null,
          health.api_version || null
        );
      }

      set({
        health: true,
        message: null,
        messageTitle: null,
        lastCheckTime: Date.now(),
        status: health,
        pipelineBusy: health.pipeline_busy
      })
      return true
    }
    set({
      health: false,
      message: health.message,
      messageTitle: 'Backend Health Check Error!',
      lastCheckTime: Date.now(),
      status: null
    })
    return false
  },

  clear: () => {
    set({ health: true, message: null, messageTitle: null })
  },

  setErrorMessage: (message: string, messageTitle: string) => {
    set({ health: false, message, messageTitle })
  },

  setPipelineBusy: (busy: boolean) => {
    set({ pipelineBusy: busy })
  }
}))

const useBackendState = createSelectors(useBackendStateStoreBase)

export { useBackendState }

const parseTokenPayload = (token: string): { sub?: string; role?: string } => {
  try {
    // JWT tokens are in the format: header.payload.signature
    const parts = token.split('.');
    if (parts.length !== 3) return {};
    const payload = JSON.parse(atob(parts[1]));
    return payload;
  } catch (e) {
    console.error('Error parsing token payload:', e);
    return {};
  }
};

const getUsernameFromToken = (token: string): string | null => {
  const payload = parseTokenPayload(token);
  return payload.sub || null;
};

const isGuestToken = (token: string): boolean => {
  const payload = parseTokenPayload(token);
  return payload.role === 'guest';
};

const initAuthState = (): { isAuthenticated: boolean; isGuestMode: boolean; coreVersion: string | null; apiVersion: string | null; username: string | null } => {
  const token = localStorage.getItem('LIGHTRAG-API-TOKEN');
  const coreVersion = localStorage.getItem('LIGHTRAG-CORE-VERSION');
  const apiVersion = localStorage.getItem('LIGHTRAG-API-VERSION');
  const username = token ? getUsernameFromToken(token) : null;

  if (!token) {
    return {
      isAuthenticated: false,
      isGuestMode: false,
      coreVersion: coreVersion,
      apiVersion: apiVersion,
      username: null,
    };
  }

  return {
    isAuthenticated: true,
    isGuestMode: isGuestToken(token),
    coreVersion: coreVersion,
    apiVersion: apiVersion,
    username: username,
  };
};

export const useAuthStore = create<AuthState>(set => {
  // Get initial state from localStorage
  const initialState = initAuthState();

  return {
    isAuthenticated: initialState.isAuthenticated,
    isGuestMode: initialState.isGuestMode,
    coreVersion: initialState.coreVersion,
    apiVersion: initialState.apiVersion,
    username: initialState.username,

    login: (token, isGuest = false, coreVersion = null, apiVersion = null) => {
      localStorage.setItem('LIGHTRAG-API-TOKEN', token);

      if (coreVersion) {
        localStorage.setItem('LIGHTRAG-CORE-VERSION', coreVersion);
      }
      if (apiVersion) {
        localStorage.setItem('LIGHTRAG-API-VERSION', apiVersion);
      }

      const username = getUsernameFromToken(token);
      set({
        isAuthenticated: true,
        isGuestMode: isGuest,
        username: username,
        coreVersion: coreVersion,
        apiVersion: apiVersion,
      });
    },

    logout: () => {
      localStorage.removeItem('LIGHTRAG-API-TOKEN');

      const coreVersion = localStorage.getItem('LIGHTRAG-CORE-VERSION');
      const apiVersion = localStorage.getItem('LIGHTRAG-API-VERSION');

      set({
        isAuthenticated: false,
        isGuestMode: false,
        username: null,
        coreVersion: coreVersion,
        apiVersion: apiVersion,
      });
    },

    setVersion: (coreVersion, apiVersion) => {
      // Update localStorage
      if (coreVersion) {
        localStorage.setItem('LIGHTRAG-CORE-VERSION', coreVersion);
      }
      if (apiVersion) {
        localStorage.setItem('LIGHTRAG-API-VERSION', apiVersion);
      }

      // Update state
      set({
        coreVersion: coreVersion,
        apiVersion: apiVersion
      });
    }
  };
});
