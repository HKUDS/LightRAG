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

export const apiRequest = async (path, options = {}) => {
  const {
    method = 'GET',
    baseUrl = resolveBaseUrl(),
    query,
    headers,
    body,
    timeout = DEFAULT_TIMEOUT,
    signal,
  } = options

  console.log('API Request:', { baseUrl, method, path, query, headers, body })

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  if (signal) {
    if (signal.aborted) {
      controller.abort(signal.reason)
    } else {
      signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true })
    }
  }

  const target = appendQueryParams(`${baseUrl}${path}`, query)
  const isFormData = typeof FormData !== 'undefined' && body instanceof FormData

  const requestOptions = {
    method,
    headers: {
      ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
      ...headers,
    },
    body: isFormData ? body : body !== undefined ? JSON.stringify(body) : undefined,
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
