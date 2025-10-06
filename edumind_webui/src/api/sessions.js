import { apiRequest, buildQuery } from './http'

const basePath = '/sessions'

const normaliseQuery = (query) => buildQuery(query)

const mapSessionToCanvas = (session = {}) => ({
  id: session.id,
  name: session.topic || 'Untitled Canvas',
  topic: session.topic || 'Untitled Canvas',
  userId: session.user_id,
  projectId: session.project_id,
  memoryState: Array.isArray(session.memory_state) ? session.memory_state : [],
  createdAt: session.created_at,
  lastActiveAt: session.last_active_at,
})

export const createSession = ({ payload, query, headers } = {}) => {
  if (!payload) {
    throw new Error('createSession requires a payload parameter')
  }

  return apiRequest(basePath, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const listSessions = ({ query, headers } = {}) =>
  apiRequest(basePath, {
    method: 'GET',
    query: normaliseQuery(query),
    headers,
  })

export const getSession = ({ id, query, headers } = {}) => {
  if (!id) {
    throw new Error('getSession requires an id parameter')
  }

  return apiRequest(`${basePath}/${encodeURIComponent(id)}`, {
    method: 'GET',
    query: normaliseQuery(query),
    headers,
  })
}

export const updateSession = ({ id, payload, query, headers } = {}) => {
  if (!id) {
    throw new Error('updateSession requires an id parameter')
  }
  if (!payload) {
    throw new Error('updateSession requires a payload parameter')
  }

  return apiRequest(`${basePath}/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const deleteSession = ({ id, query, headers } = {}) => {
  if (!id) {
    throw new Error('deleteSession requires an id parameter')
  }

  return apiRequest(`${basePath}/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    query: normaliseQuery(query),
    headers,
  })
}

export const listCanvas = async ({ projectId, userId, limit = 5, query = {}, headers } = {}) => {
  const response = await listSessions({
    query: {
      ...query,
      project_id: projectId ?? query.project_id,
      user_id: userId ?? query.user_id,
      limit: query.limit ?? limit,
      sort: query.sort ?? '-created_at',
    },
    headers,
  })

  const sessions = Array.isArray(response?.sessions) ? response.sessions : []
  return {
    ...response,
    canvases: sessions.map((session) => mapSessionToCanvas(session)),
  }
}

export const fetchCanvas = async ({ id, query, headers } = {}) => {
  if (!id) {
    throw new Error('fetchCanvas requires an id parameter')
  }

  const response = await getSession({ id, query, headers })
  const canvas = response?.session ? mapSessionToCanvas(response.session) : null
  return {
    ...response,
    canvas,
  }
}
