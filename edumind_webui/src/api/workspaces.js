import { apiRequest, buildQuery } from './http'

const basePath = '/workspaces'

const normaliseQuery = (query) => buildQuery(query)

export const createWorkspace = ({ payload, query, headers } = {}) => {
  if (!payload) {
    throw new Error('createWorkspace requires a payload parameter')
  }

  return apiRequest(basePath, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const listWorkspaces = ({ query, headers } = {}) =>
  apiRequest(basePath, {
    method: 'GET',
    query: normaliseQuery(query),
    headers,
  })

export const getWorkspace = ({ id, query, headers } = {}) => {
  if (!id) {
    throw new Error('getWorkspace requires an id parameter')
  }

  return apiRequest(`${basePath}/${encodeURIComponent(id)}`, {
    method: 'GET',
    query: normaliseQuery(query),
    headers,
  })
}

export const updateWorkspace = ({ id, payload, query, headers } = {}) => {
  if (!id) {
    throw new Error('updateWorkspace requires an id parameter')
  }
  if (!payload) {
    throw new Error('updateWorkspace requires a payload parameter')
  }

  return apiRequest(`${basePath}/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const deleteWorkspace = ({ id, query, headers } = {}) => {
  if (!id) {
    throw new Error('deleteWorkspace requires an id parameter')
  }

  return apiRequest(`${basePath}/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    query: normaliseQuery(query),
    headers,
  })
}

export const initializeWorkspace = ({ id, query, headers } = {}) => {
  if (!id) {
    throw new Error('initializeWorkspace requires an id parameter')
  }

  return apiRequest(`${basePath}/${encodeURIComponent(id)}/initializations`, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
  })
}
