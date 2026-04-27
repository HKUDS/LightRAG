import { describe, expect, test } from 'bun:test'
import {
  areasFixture,
  approvalQueueFixture,
  assistantsFixture,
  auditTrailFixture,
  chatThreadsFixture,
  criticFindingsFixture,
  documentsFixture,
  getAuditTrailByWorkspace,
  getActivitiesByWorkspace,
  getAssistantsByWorkspace,
  getChatsByWorkspace,
  getDocumentsByWorkspace,
  getPendingApprovalsByWorkspace,
  getSkillActionById,
  homeSnapshotFixture,
  internalAgentsFixture,
  internalSubagentsFixture,
  modelCatalogSyncFixture,
  skillActionsFixture,
  toWorkspaceHeader
} from './littleBullKnowledge'

const areaIds = new Set(areasFixture.map((area) => area.id))
const documentIds = new Set(documentsFixture.map((document) => document.id))

describe('littleBullKnowledge fixtures', () => {
  test('uses workspace ids that can be sent through LIGHTRAG-WORKSPACE', () => {
    for (const area of areasFixture) {
      expect(toWorkspaceHeader(area.id)).toBe(area.id)
      expect(area.id).toMatch(/^[a-zA-Z0-9_]+$/)
    }
  })

  test('keeps documents attached to existing workspaces', () => {
    for (const document of documentsFixture) {
      expect(areaIds.has(document.workspaceId)).toBe(true)
      expect(document.fileName.length).toBeGreaterThan(3)
      expect(document.summary.length).toBeGreaterThan(20)
    }
  })

  test('keeps chat citations attached to existing documents', () => {
    for (const thread of chatThreadsFixture) {
      expect(areaIds.has(thread.workspaceId)).toBe(true)

      for (const message of thread.messages) {
        for (const citation of message.citations ?? []) {
          expect(documentIds.has(citation.documentId)).toBe(true)
          expect(citation.excerpt.length).toBeGreaterThan(20)
        }
      }
    }
  })

  test('offers workspace-scoped slices for future UI pages', () => {
    expect(getDocumentsByWorkspace('casa').length).toBeGreaterThan(0)
    expect(getChatsByWorkspace('casa').length).toBeGreaterThan(0)
    expect(getActivitiesByWorkspace('casa').length).toBeGreaterThan(0)
    expect(getAssistantsByWorkspace('casa').length).toBeGreaterThan(0)
  })

  test('keeps assistants attached to existing workspaces and profiles', () => {
    for (const assistant of assistantsFixture) {
      expect(assistant.workspaceIds.length).toBeGreaterThan(0)
      expect(['rapido', 'equilibrado', 'inteligente', 'privado']).toContain(
        assistant.defaultProfile
      )

      for (const workspaceId of assistant.workspaceIds) {
        expect(areaIds.has(workspaceId)).toBe(true)
      }
    }
  })

  test('keeps home snapshot aligned with fixture totals', () => {
    const areaCard = homeSnapshotFixture.summaryCards.find((card) => card.label === 'Áreas')
    const documentCard = homeSnapshotFixture.summaryCards.find(
      (card) => card.label === 'Documentos'
    )

    expect(areaCard?.value).toBe(String(areasFixture.length))
    expect(documentCard?.value).toBe(String(documentsFixture.length))
    expect(areaIds.has(homeSnapshotFixture.activeWorkspaceId)).toBe(true)
  })

  test('represents the agentic layer with known agents, subagents, and skills', () => {
    expect(internalAgentsFixture.map((agent) => agent.id)).toContain('orchestrator_agent')
    expect(internalAgentsFixture.map((agent) => agent.id)).toContain('retrieval_agent')
    expect(internalSubagentsFixture.map((subagent) => subagent.id)).toContain(
      'prompt_injection_guard_subagent'
    )
    expect(skillActionsFixture.map((skill) => skill.id)).toContain('query_lightrag')
    expect(skillActionsFixture.map((skill) => skill.id)).toContain('sync_model_catalog')
  })

  test('requires approval for destructive or sensitive actions', () => {
    const destructiveActions = skillActionsFixture.filter((action) => action.isDestructive)
    expect(destructiveActions.length).toBeGreaterThan(0)

    for (const action of destructiveActions) {
      expect(action.requiresHumanApproval).toBe(true)
    }

    for (const approval of approvalQueueFixture) {
      expect(areaIds.has(approval.workspaceId)).toBe(true)
      expect(getSkillActionById(approval.actionId)).toBeDefined()
    }
  })

  test('keeps critic findings and audit trail attached to real workspaces and actions', () => {
    for (const finding of criticFindingsFixture) {
      expect(areaIds.has(finding.workspaceId)).toBe(true)
      expect(getSkillActionById(finding.relatedActionId)).toBeDefined()
    }

    for (const event of auditTrailFixture) {
      expect(areaIds.has(event.workspaceId)).toBe(true)
      expect(event.tenantId).toBe('little_bull_home_demo')
    }

    expect(getPendingApprovalsByWorkspace('casa').length).toBeGreaterThan(0)
    expect(getAuditTrailByWorkspace('casa').length).toBeGreaterThan(0)
  })

  test('captures model catalog sync state without hardcoding visible model ids', () => {
    expect(modelCatalogSyncFixture.provider).toBe('OpenRouter')
    expect(modelCatalogSyncFixture.modelCount).toBeGreaterThan(0)
    expect(modelCatalogSyncFixture.outputPath).toContain('openrouter_catalog.json')
  })
})
