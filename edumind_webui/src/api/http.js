import { getToken, decodeJwt } from '@/api/auth'

const DEFAULT_TIMEOUT = 100000

export const resolveBaseUrl = () => {
  const envUrl = import.meta?.env?.VITE_EDUMIND_API_BASE_URL || "http://localhost:9621"
  if (envUrl) {
    return envUrl.endsWith('/') ? envUrl.slice(0, -1) : envUrl
  }
  return ''
}

const appendQueryParams = (url, query) => {
  if (!query) {
    return url
  }
  const urlInstance = typeof url === 'string' ? new URL(url, window.location.origin) : new URL(url.toString())
  Object.entries(query).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') {
      return
    }
    if (Array.isArray(value)) {
      value.forEach((entry) => urlInstance.searchParams.append(key, entry))
    } else if (typeof value === 'object') {
      urlInstance.searchParams.append(key, JSON.stringify(value))
    } else {
      urlInstance.searchParams.append(key, value)
    }
  })

  return urlInstance
}

const parseResponse = async (response) => {
  const rawContentType = response.headers.get('content-type') || ''

  if (rawContentType.includes('application/json')) {
    const json = await response.json()
    if (!response.ok) {
      const error = new Error(json?.message || response.statusText)
      error.status = response.status
      error.payload = json
      throw error
    }
    return json
  }

  const text = await response.text()
  if (!response.ok) {
    const error = new Error(text || response.statusText)
    error.status = response.status
    error.payload = text
    throw error
  }
  return text
}

const TOKEN_KEY = 'edumind_access_token'
export const setAuthToken = (t) => sessionStorage.setItem(TOKEN_KEY, t)
export const getAuthToken = () => sessionStorage.getItem(TOKEN_KEY) || ''
export const clearAuthToken = () => sessionStorage.removeItem(TOKEN_KEY)

function getUserIdFromToken() {
  const t = getToken();
  if (!t) return null;
  const p = decodeJwt(t);
  if (!p) return null;
  return (p.metadata && p.metadata.user_id) ? p.metadata.user_id : "not found";
}

export const apiRequest = async (path, options = {}) => {
  const {
    method = 'GET',
    baseUrl = resolveBaseUrl(),
    query,
    headers,
    body,
    timeout = DEFAULT_TIMEOUT,
    signal,
    auth = {},
  } = options

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason)
    else signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true })
  }

  const resolvedUserId = (auth.userId ?? getUserIdFromToken()) || null
  const mergedQuery = {
    ...(query || {}),
    ...(query?.user_id === undefined && resolvedUserId ? { user_id: resolvedUserId } : {})
  }

  const target = appendQueryParams(`${baseUrl}${path}`, mergedQuery)

  const isFormData = typeof FormData !== 'undefined' && body instanceof FormData
  const isUrlEncoded = typeof URLSearchParams !== 'undefined' && body instanceof URLSearchParams

  const h = new Headers({
    Accept: 'application/json',
    ...headers,
  })

  // Content-Type rules
  if (isFormData) {
    // let browser set boundary
  } else if (isUrlEncoded) {
    if (!h.has('Content-Type')) h.set('Content-Type', 'application/x-www-form-urlencoded')
  } else if (body !== undefined && !h.has('Content-Type')) {
    h.set('Content-Type', 'application/json')
  }

  // --- Inject auth headers if present ---
  const token = auth.token ?? getAuthToken()
  if (token) h.set('Authorization', `Bearer ${token}`)

  if (auth.apiKey ?? import.meta.env.VITE_LIGHTRAG_API_KEY) {
    h.set('X-API-Key', auth.apiKey ?? import.meta.env.VITE_LIGHTRAG_API_KEY)
  }

  // Your backend prefers X-Workspace header (or ?workspace=)
  const ws = auth.workspace ?? import.meta.env.VITE_DEFAULT_WORKSPACE
  if (ws) h.set('X-Workspace', ws)

  const userId = getUserIdFromToken();
  if (auth.userId) h.set('X-User-ID', auth.userId)

  const requestOptions = {
    method,
    headers: h,
    body: isFormData ? body : isUrlEncoded ? body : body !== undefined ? JSON.stringify(body) : undefined,
    signal: controller.signal,
  }

  try {
    const response = await fetch(target.toString(), requestOptions)
    clearTimeout(timeoutId)
    return await parseResponse(response)
  } catch (error) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      const abortError = new Error('Request timed out or was aborted')
      abortError.aborted = true
      throw abortError
    }
    throw error
  }
}

export const buildHeaders = ({ apiKey, workspace, workspaceHeader } = {}) => {
  const result = {}
  if (apiKey) {
    result['X-API-Key'] = apiKey
  }
  if (workspace) {
    result['workspace'] = workspace
  }
  if (workspaceHeader) {
    result['X-Workspace'] = workspaceHeader
  }
  return result
}

export const withAuthHeaders = (headers = {}, authOptions = {}) => ({
  ...headers,
  ...buildHeaders(authOptions),
})

export const buildQuery = (params = {}) => {
  return Object.fromEntries(
    Object.entries(params).filter(([, value]) => value !== undefined && value !== null)
  )
}
