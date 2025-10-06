import { apiRequest, buildQuery } from './http'

const basePath = '/ai'

const normaliseQuery = (query) => buildQuery(query)

export const generateQuestions = ({ payload, query, headers } = {}) => {
  if (!payload) {
    throw new Error('generateQuestions requires a payload parameter')
  }

  return apiRequest(`${basePath}/questions/generate`, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}

export const generateQuestionVariants = ({ questionId, payload, query, headers } = {}) => {
  if (!questionId) {
    throw new Error('generateQuestionVariants requires a questionId parameter')
  }
  if (!payload) {
    throw new Error('generateQuestionVariants requires a payload parameter')
  }

  return apiRequest(`${basePath}/questions/${encodeURIComponent(questionId)}/variants`, {
    method: 'POST',
    query: normaliseQuery(query),
    headers,
    body: payload,
  })
}
