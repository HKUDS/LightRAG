import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { checkHealth, LightragStatus } from '@/api/lightrag'

interface BackendState {
  health: boolean
  message: string | null
  messageTitle: string | null

  status: LightragStatus | null

  lastCheckTime: number

  check: () => Promise<boolean>
  clear: () => void
  setErrorMessage: (message: string, messageTitle: string) => void
}

interface AuthState {
  isAuthenticated: boolean;
  showLoginModal: boolean;
  isGuestMode: boolean;  // Add guest mode flag
  login: (token: string, isGuest?: boolean) => void;
  logout: () => void;
  setShowLoginModal: (show: boolean) => void;
}

const useBackendStateStoreBase = create<BackendState>()((set) => ({
  health: true,
  message: null,
  messageTitle: null,
  lastCheckTime: Date.now(),
  status: null,

  check: async () => {
    const health = await checkHealth()
    if (health.status === 'healthy') {
      set({
        health: true,
        message: null,
        messageTitle: null,
        lastCheckTime: Date.now(),
        status: health
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
  }
}))

const useBackendState = createSelectors(useBackendStateStoreBase)

export { useBackendState }

// Helper function to check if token is a guest token
const isGuestToken = (token: string): boolean => {
  try {
    // JWT tokens are in the format: header.payload.signature
    const parts = token.split('.');
    if (parts.length !== 3) return false;
    
    // Decode the payload (second part)
    const payload = JSON.parse(atob(parts[1]));
    
    // Check if the token has a role field with value "guest"
    return payload.role === 'guest';
  } catch (e) {
    console.error('Error parsing token:', e);
    return false;
  }
};

// Initialize auth state from localStorage
const initAuthState = (): { isAuthenticated: boolean; isGuestMode: boolean } => {
  const token = localStorage.getItem('LIGHTRAG-API-TOKEN');
  if (!token) {
    return { isAuthenticated: false, isGuestMode: false };
  }
  
  return { 
    isAuthenticated: true, 
    isGuestMode: isGuestToken(token)
  };
};

export const useAuthStore = create<AuthState>(set => {
  // Get initial state from localStorage
  const initialState = initAuthState();
  
  return {
    isAuthenticated: initialState.isAuthenticated,
    showLoginModal: false,
    isGuestMode: initialState.isGuestMode,
    
    login: (token, isGuest = false) => {
      localStorage.setItem('LIGHTRAG-API-TOKEN', token);
      set({ 
        isAuthenticated: true, 
        showLoginModal: false,
        isGuestMode: isGuest
      });
    },
    
    logout: () => {
      localStorage.removeItem('LIGHTRAG-API-TOKEN');
      set({ 
        isAuthenticated: false,
        isGuestMode: false
      });
    },
    
    setShowLoginModal: (show) => set({ showLoginModal: show })
  };
});
