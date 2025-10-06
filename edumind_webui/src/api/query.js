import { resolveBaseUrl } from './http'

const decoder = new TextDecoder()

export const streamQuery = async ({ payload, headers = {}, query, signal, onChunk }) => {
  const baseUrl = resolveBaseUrl()
  const url = new URL(`${baseUrl}/query/stream`, window.location.origin)

  if (query) {
    Object.entries(query).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        url.searchParams.append(key, value)
      }
    })
  }

  const controller = new AbortController()
  const signals = [signal, controller.signal].filter(Boolean)
  const abortHandler = (reason) => controller.abort(reason)
  if (signal) {
    if (signal.aborted) {
      controller.abort(signal.reason)
    } else {
      signal.addEventListener('abort', () => abortHandler(signal.reason), { once: true })
    }
  }

  const response = await fetch(url.toString(), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify(payload),
    signal: controller.signal,
  })

  if (!response.ok) {
    const text = await response.text()
    const error = new Error(text || response.statusText)
    error.status = response.status
    throw error
  }

  if (!response.body || typeof response.body.getReader !== 'function') {
    const text = await response.text()
    if (typeof onChunk === 'function') {
      onChunk(text)
    }
    return
  }

  const reader = response.body.getReader()
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) {
        break
      }
      if (value && typeof onChunk === 'function') {
        const chunk = decoder.decode(value, { stream: true })
        if (chunk) {
          onChunk(chunk)
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
