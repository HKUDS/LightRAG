import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { checkHealth, LightragStatus } from '@/api/lightrag'
import { useSettingsStore } from './settings'
import { healthCheckInterval } from '@/lib/constants'
import { decodeBase64Url } from '@/lib/base64url'

export type ApiDocsCapability = 'unknown' | 'available' | 'unavailable'

interface BackendState {
  health: boolean
  message: string | null
  messageTitle: string | null
  status: LightragStatus | null
  // Whether the backend serves /docs. Tri-state on purpose: `status` is
  // transient (null before the first response and reset on failure), so a
  // predicate over it would show the docs entry point against servers that
  // explicitly disabled docs. Only a successful /health response updates
  // this; failures retain the last known value (RFC #3671).
  apiDocsCapability: ApiDocsCapability
  lastCheckTime: number
  pipelineBusy: boolean
  pipelineActive: boolean
  healthCheckIntervalId: ReturnType<typeof setInterval> | null
  healthCheckFunction: (() => void) | null
  healthCheckIntervalValue: number

  check: () => Promise<boolean>
  // Resolve `apiDocsCapability` alone, without touching the shared backend
  // health state. Needed when periodic health checks are disabled: reusing
  // `check()` there would let a single failed probe latch `health: false`,
  // which stops the document list polling for good and re-opens the API key
  // alert on every dismissal (RFC #3671).
  probeApiDocsCapability: () => Promise<void>
  clear: () => void
  setErrorMessage: (message: string, messageTitle: string) => void
  setPipelineBusy: (busy: boolean) => void
  setHealthCheckFunction: (fn: () => void) => void
  resetHealthCheckTimer: () => void
  resetHealthCheckTimerDelayed: (delayMs: number) => void
  clearHealthCheckTimer: () => void
}

interface AuthState {
  isAuthenticated: boolean;
  isGuestMode: boolean;  // Add guest mode flag
  coreVersion: string | null;
  apiVersion: string | null;
  username: string | null; // login username
  webuiTitle: string | null; // Custom title
  webuiDescription: string | null; // Title description
  lastTokenRenewal: string | null; // Human-readable local time of last token renewal (for debugging and monitoring)
  tokenExpiresAt: number | null; // Token expiration timestamp (extracted from JWT)

  login: (token: string, isGuest?: boolean, coreVersion?: string | null, apiVersion?: string | null, webuiTitle?: string | null, webuiDescription?: string | null) => void;
  logout: () => void;
  setVersion: (coreVersion: string | null, apiVersion: string | null) => void;
  setCustomTitle: (webuiTitle: string | null, webuiDescription: string | null) => void;
  setTokenRenewal: (renewalTime: number, expiresAt: number) => void; // Track token renewal
}

// Single-flight guard for the docs capability probe: React 19 strict mode
// mounts effects twice, and the probe is idempotent but not free.
let apiDocsProbeInFlight: Promise<void> | null = null

// Last-caller-wins guard for health checks: concurrent check() calls (a
// credential probe racing a save-triggered re-probe, or overlapping periodic
// checks) must not let COMPLETION ORDER decide the shared state — a stale
// request finishing last would overwrite health/message with an outdated
// result (e.g. the previous API key's failure after the replacement already
// validated). Only the newest in-flight check may write.
let healthCheckGeneration = 0

const useBackendStateStoreBase = create<BackendState>()((set, get) => ({
  health: true,
  message: null,
  messageTitle: null,
  lastCheckTime: Date.now(),
  status: null,
  apiDocsCapability: 'unknown',
  pipelineBusy: false,
  pipelineActive: false,
  healthCheckIntervalId: null,
  healthCheckFunction: null,
  healthCheckIntervalValue: healthCheckInterval * 1000, // Use constant from lib/constants

  check: async () => {
    const generation = ++healthCheckGeneration
    const health = await checkHealth()
    if (generation !== healthCheckGeneration) {
      // Superseded mid-flight by a newer check: discard this result entirely
      // (no state, version, title, or graph-limit writes) and report the
      // state the newest check established.
      return get().health
    }
    if (health.status === 'healthy') {
      // Update version information if health check returns it
      if (health.core_version || health.api_version) {
        useAuthStore.getState().setVersion(
          health.core_version || null,
          health.api_version || null
        );
      }

      // Update custom title information if health check returns it
      if ('webui_title' in health || 'webui_description' in health) {
        useAuthStore.getState().setCustomTitle(
          'webui_title' in health ? (health.webui_title ?? null) : null,
          'webui_description' in health ? (health.webui_description ?? null) : null
        );
      }

      // Extract and store backend max graph nodes limit
      if (health.configuration?.max_graph_nodes) {
        const maxNodes = parseInt(health.configuration.max_graph_nodes, 10)
        if (!isNaN(maxNodes) && maxNodes > 0) {
          const currentBackendMaxNodes = useSettingsStore.getState().backendMaxGraphNodes

          // Only update if the backend limit has actually changed
          if (currentBackendMaxNodes !== maxNodes) {
            useSettingsStore.getState().setBackendMaxGraphNodes(maxNodes)

            // Auto-adjust current graphMaxNodes if it exceeds the new backend limit
            const currentMaxNodes = useSettingsStore.getState().graphMaxNodes
            if (currentMaxNodes > maxNodes) {
              useSettingsStore.getState().setGraphMaxNodes(maxNodes, true)
            }
          }
        }
      }

      set({
        health: true,
        message: null,
        messageTitle: null,
        lastCheckTime: Date.now(),
        status: health,
        // A missing field means an older backend, which always exposes docs.
        // This interpretation is only safe here, on a successful response.
        apiDocsCapability: health.api_docs_available === false ? 'unavailable' : 'available',
        pipelineBusy: health.pipeline_busy,
        pipelineActive: health.pipeline_active ?? health.pipeline_busy
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

  probeApiDocsCapability: async () => {
    if (get().apiDocsCapability !== 'unknown') return
    if (apiDocsProbeInFlight) return apiDocsProbeInFlight

    apiDocsProbeInFlight = (async () => {
      const health = await checkHealth()
      // A failed probe mutates nothing: the capability stays 'unknown' (the
      // docs entry point stays hidden) and no health/message write can leak
      // into the periodic-check state machine.
      if (health.status !== 'healthy') return
      set({
        apiDocsCapability: health.api_docs_available === false ? 'unavailable' : 'available'
      })
    })().finally(() => {
      apiDocsProbeInFlight = null
    })

    return apiDocsProbeInFlight
  },

  clear: () => {
    set({ health: true, message: null, messageTitle: null })
  },

  setErrorMessage: (message: string, messageTitle: string) => {
    set({ health: false, message, messageTitle })
  },

  setPipelineBusy: (busy: boolean) => {
    set({ pipelineBusy: busy })
  },

  setHealthCheckFunction: (fn: () => void) => {
    set({ healthCheckFunction: fn })
  },

  resetHealthCheckTimer: () => {
    const { healthCheckIntervalId, healthCheckFunction, healthCheckIntervalValue } = get()
    if (healthCheckIntervalId) {
      clearInterval(healthCheckIntervalId)
    }
    if (healthCheckFunction) {
      healthCheckFunction() // run health check immediately
      const newIntervalId = setInterval(healthCheckFunction, healthCheckIntervalValue)
      set({ healthCheckIntervalId: newIntervalId })
    }
  },

  resetHealthCheckTimerDelayed: (delayMs: number) => {
    setTimeout(() => {
      get().resetHealthCheckTimer()
    }, delayMs)
  },

  clearHealthCheckTimer: () => {
    const { healthCheckIntervalId } = get()
    if (healthCheckIntervalId) {
      clearInterval(healthCheckIntervalId)
      set({ healthCheckIntervalId: null })
    }
  }
}))

const useBackendState = createSelectors(useBackendStateStoreBase)

export { useBackendState }

// Format timestamp to human-readable local time with timezone
const formatTimestampToLocalString = (timestamp: number): string => {
  const date = new Date(timestamp);
  // Use Swedish locale 'sv-SE' to get YYYY-MM-DD HH:mm:ss format
  const localTime = date.toLocaleString('sv-SE', { hour12: false });
  // Get timezone offset
  const offsetMinutes = -date.getTimezoneOffset();
  const offsetHours = Math.floor(Math.abs(offsetMinutes) / 60);
  const offsetSign = offsetMinutes >= 0 ? '+' : '-';
  return `${localTime} (UTC${offsetSign}${offsetHours})`;
};

/**
 * Read a JWT's claims.
 *
 * MUST use the same decoder as `isTokenLocallyValid` below. Bare `atob`
 * rejects the `-`/`_` alphabet, so the two would disagree about the very
 * same token: validation admits it, this returns `{}`, and the session runs
 * with `username === null`, `isGuestMode === false` and no expiry. The
 * username is the worst of those — `navigationService` only records
 * LIGHTRAG-PREVIOUS-USER `if (currentUsername)`, so that user's logout
 * writes no identity marker at all and the retrieval-history cleanup never
 * fires for them, leaving their conversations to whoever logs in next.
 *
 * Whether a payload needs the url alphabet depends on its exact bytes,
 * `exp` included — so it varies per token issuance for the SAME user, which
 * is what made this intermittent rather than reproducible.
 */
const parseTokenPayload = (token: string): { sub?: string; role?: string; exp?: number } => {
  try {
    // JWT tokens are in the format: header.payload.signature
    const parts = token.split('.');
    if (parts.length !== 3) return {};
    const decoded = decodeBase64Url(parts[1]);
    if (decoded === null) return {};
    return JSON.parse(decoded);
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

const getTokenExpiresAt = (token: string): number | null => {
  const payload = parseTokenPayload(token);
  return payload.exp ? payload.exp * 1000 : null; // Convert to milliseconds
};

const TOKEN_STORAGE_KEY = 'LIGHTRAG-API-TOKEN';
// localStorage keys that only make sense alongside a valid token; cleared
// together with it when local validation rejects the token.
const TOKEN_COMPANION_STORAGE_KEYS = ['LIGHTRAG-LAST-TOKEN-RENEWAL'];

/**
 * LOCAL token validity check: JWT structure parses and `exp` has not passed.
 *
 * This deliberately replaces the old "token exists ⇒ authenticated" rule: an
 * expired or structurally broken token used to render the whole app before a
 * 401 bounced the user back. It does NOT take over authentication — a token
 * that only the server can reject (invalid signature, revoked) still passes
 * here and is corrected by the usual 401 path.
 */
export const isTokenLocallyValid = (token: string): boolean => {
  const parts = token.split('.');
  if (parts.length !== 3) return false;
  const decoded = decodeBase64Url(parts[1]);
  if (decoded === null) return false;
  let payload: unknown;
  try {
    payload = JSON.parse(decoded);
  } catch {
    return false;
  }
  if (!payload || typeof payload !== 'object') return false;
  const exp = (payload as { exp?: unknown }).exp;
  if (typeof exp !== 'number') return false;
  return exp * 1000 > Date.now();
};

/** Remove a locally-invalid token together with its companion keys. */
export const clearLocalToken = (): void => {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
  for (const key of TOKEN_COMPANION_STORAGE_KEYS) {
    localStorage.removeItem(key);
  }
};

export const initAuthState = (): { isAuthenticated: boolean; isGuestMode: boolean; coreVersion: string | null; apiVersion: string | null; username: string | null; webuiTitle: string | null; webuiDescription: string | null; lastTokenRenewal: string | null; tokenExpiresAt: number | null } => {
  let token = localStorage.getItem(TOKEN_STORAGE_KEY);
  if (token && !isTokenLocallyValid(token)) {
    // Expired or structurally broken: clear it (and its companions) and
    // proceed as "no valid token" — the entry's unauthenticated default page
    // (login / welcome) takes over instead of the app rendering first.
    clearLocalToken();
    token = null;
  }
  const coreVersion = localStorage.getItem('LIGHTRAG-CORE-VERSION');
  const apiVersion = localStorage.getItem('LIGHTRAG-API-VERSION');
  const webuiTitle = localStorage.getItem('LIGHTRAG-WEBUI-TITLE');
  const webuiDescription = localStorage.getItem('LIGHTRAG-WEBUI-DESCRIPTION');
  // Read AFTER the validity check above so a just-cleared companion key is
  // not resurrected into the store.
  const lastTokenRenewal = localStorage.getItem('LIGHTRAG-LAST-TOKEN-RENEWAL');
  const username = token ? getUsernameFromToken(token) : null;
  const tokenExpiresAt = token ? getTokenExpiresAt(token) : null;

  if (!token) {
    return {
      isAuthenticated: false,
      isGuestMode: false,
      coreVersion: coreVersion,
      apiVersion: apiVersion,
      username: null,
      webuiTitle: webuiTitle,
      webuiDescription: webuiDescription,
      lastTokenRenewal: null,
      tokenExpiresAt: null,
    };
  }

  return {
    isAuthenticated: true,
    isGuestMode: isGuestToken(token),
    coreVersion: coreVersion,
    apiVersion: apiVersion,
    username: username,
    webuiTitle: webuiTitle,
    webuiDescription: webuiDescription,
    lastTokenRenewal: lastTokenRenewal,
    tokenExpiresAt: tokenExpiresAt,
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
    webuiTitle: initialState.webuiTitle,
    webuiDescription: initialState.webuiDescription,
    lastTokenRenewal: initialState.lastTokenRenewal,
    tokenExpiresAt: initialState.tokenExpiresAt,

    login: (token, isGuest = false, coreVersion = null, apiVersion = null, webuiTitle = null, webuiDescription = null) => {
      localStorage.setItem('LIGHTRAG-API-TOKEN', token);

      if (coreVersion) {
        localStorage.setItem('LIGHTRAG-CORE-VERSION', coreVersion);
      }
      if (apiVersion) {
        localStorage.setItem('LIGHTRAG-API-VERSION', apiVersion);
      }

      if (webuiTitle) {
        localStorage.setItem('LIGHTRAG-WEBUI-TITLE', webuiTitle);
      } else {
        localStorage.removeItem('LIGHTRAG-WEBUI-TITLE');
      }

      if (webuiDescription) {
        localStorage.setItem('LIGHTRAG-WEBUI-DESCRIPTION', webuiDescription);
      } else {
        localStorage.removeItem('LIGHTRAG-WEBUI-DESCRIPTION');
      }

      const username = getUsernameFromToken(token);
      const tokenExpiresAt = getTokenExpiresAt(token);
      const now = Date.now();
      const formattedTime = formatTimestampToLocalString(now);

      // Initialize token issuance time with human-readable format
      localStorage.setItem('LIGHTRAG-LAST-TOKEN-RENEWAL', formattedTime);

      set({
        isAuthenticated: true,
        isGuestMode: isGuest,
        username: username,
        coreVersion: coreVersion,
        apiVersion: apiVersion,
        webuiTitle: webuiTitle,
        webuiDescription: webuiDescription,
        tokenExpiresAt: tokenExpiresAt,
        lastTokenRenewal: formattedTime,
      });
    },

    logout: () => {
      localStorage.removeItem('LIGHTRAG-API-TOKEN');
      localStorage.removeItem('LIGHTRAG-LAST-TOKEN-RENEWAL');

      const coreVersion = localStorage.getItem('LIGHTRAG-CORE-VERSION');
      const apiVersion = localStorage.getItem('LIGHTRAG-API-VERSION');
      const webuiTitle = localStorage.getItem('LIGHTRAG-WEBUI-TITLE');
      const webuiDescription = localStorage.getItem('LIGHTRAG-WEBUI-DESCRIPTION');

      set({
        isAuthenticated: false,
        isGuestMode: false,
        username: null,
        coreVersion: coreVersion,
        apiVersion: apiVersion,
        webuiTitle: webuiTitle,
        webuiDescription: webuiDescription,
        lastTokenRenewal: null,
        tokenExpiresAt: null,
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
    },

    setCustomTitle: (webuiTitle, webuiDescription) => {
      // Update localStorage
      if (webuiTitle) {
        localStorage.setItem('LIGHTRAG-WEBUI-TITLE', webuiTitle);
      } else {
        localStorage.removeItem('LIGHTRAG-WEBUI-TITLE');
      }

      if (webuiDescription) {
        localStorage.setItem('LIGHTRAG-WEBUI-DESCRIPTION', webuiDescription);
      } else {
        localStorage.removeItem('LIGHTRAG-WEBUI-DESCRIPTION');
      }

      // Update state
      set({
        webuiTitle: webuiTitle,
        webuiDescription: webuiDescription
      });
    },

    setTokenRenewal: (renewalTime, expiresAt) => {
      const formattedTime = formatTimestampToLocalString(renewalTime);

      // Update localStorage with human-readable format
      localStorage.setItem('LIGHTRAG-LAST-TOKEN-RENEWAL', formattedTime);

      // Update state
      set({
        lastTokenRenewal: formattedTime,
        tokenExpiresAt: expiresAt
      });
    }
  };
});
