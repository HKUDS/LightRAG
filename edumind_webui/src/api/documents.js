import { apiRequest, buildQuery } from './http'

const basePath = '/documents'

const normaliseQuery = (query) => buildQuery(query)

export const scanForNewDocuments = ({ query, headers } = {}) =>
  apiRequest(`${basePath}/scan`, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
  })

export const uploadDocument = ({ file, metadata = {}, query, headers } = {}) => {
  if (!file) {
    throw new Error('uploadDocument requires a file parameter')
  }

  const formData = new FormData()
  formData.append('file', file)

  Object.entries(metadata).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      formData.append(key, value)
    }
  })

  return apiRequest(`${basePath}/upload`, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
    body: formData,
  })
}

export const insertText = ({ payload, query, headers } = {}) => {
  if (!payload) {
    throw new Error('insertText requires a payload parameter')
  }

  return apiRequest(`${basePath}/text`, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const insertTexts = ({ payload, query, headers } = {}) => {
  if (!payload) {
    throw new Error('insertTexts requires a payload parameter')
  }

  return apiRequest(`${basePath}/texts`, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const insertLinks = ({ payload, query, headers } = {}) => {
  if (!payload) {
    throw new Error('insertLinks requires a payload parameter')
  }

  return apiRequest(`${basePath}/links`, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const clearAllDocuments = ({ query, headers } = {}) =>
  apiRequest(basePath, {
    method: 'DELETE',
    query: normaliseQuery(query),
    headers,
  })

export const getDocumentsStatus = ({ query, headers } = {}) =>
  apiRequest(basePath, {
    method: 'GET',
    query: normaliseQuery(query),
    headers,
  })

export const getDocumentPipelineStatus = ({ query, headers } = {}) =>
  apiRequest(`${basePath}/pipeline_status`, {
    method: 'GET',
    query: normaliseQuery(query),
    headers,
  })

export const deleteDocumentsById = ({ payload, query, headers } = {}) => {
  if (!payload) {
    throw new Error('deleteDocumentsById requires a payload parameter')
  }

  return apiRequest(`${basePath}/delete_document`, {
    method: 'DELETE',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const clearCache = ({ payload = {}, query, headers } = {}) =>
  apiRequest(`${basePath}/clear_cache`, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })

export const deleteEntity = ({ payload, query, headers } = {}) => {
  if (!payload) {
    throw new Error('deleteEntity requires a payload parameter')
  }

  return apiRequest(`${basePath}/delete_entity`, {
    method: 'DELETE',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const deleteRelation = ({ payload, query, headers } = {}) => {
  if (!payload) {
    throw new Error('deleteRelation requires a payload parameter')
  }

  return apiRequest(`${basePath}/delete_relation`, {
    method: 'DELETE',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const getTrackStatus = ({ trackId, query, headers } = {}) => {
  if (!trackId) {
    throw new Error('getTrackStatus requires a trackId parameter')
  }

  return apiRequest(`${basePath}/track_status/${encodeURIComponent(trackId)}`, {
    method: 'GET',
    query: normaliseQuery(query),
    headers,
  })
}

export const getPaginatedDocuments = ({ payload, query, headers } = {}) => {
  if (!payload) {
    throw new Error('getPaginatedDocuments requires a payload parameter')
  }

  return apiRequest(`${basePath}/paginated`, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const getDocumentStatusCounts = ({ query, headers } = {}) =>
  apiRequest(`${basePath}/status_counts`, {
    method: 'GET',
    query: normaliseQuery(query),
    headers,
  })
