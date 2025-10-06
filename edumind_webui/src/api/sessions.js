import { apiRequest, buildQuery } from './http'

const basePath = '/sessions'

const normaliseQuery = (query) => buildQuery(query)

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
