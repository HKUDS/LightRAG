import type {
  LittleBullKnowledgeGroup,
  LittleBullKnowledgeSubgroup,
  LittleBullPrincipal
} from '@/api/lightrag'

export type LittleBullPage =
  | 'inicio'
  | 'workspaces'
  | 'grupos'
  | 'subgrupos'
  | 'conhecimento'
  | 'notas'
  | 'inbox'
  | 'daily'
  | 'canvas'
  | 'mocs'
  | 'trilhas'
  | 'grafo'
  | 'perguntar'
  | 'agent-builder'
  | 'assistentes'
  | 'modelos'
  | 'custos'
  | 'jobs'
  | 'juridico'
  | 'relatorios'
  | 'atividade'
  | 'auditoria'
  | 'aprovacoes'
  | 'admin'

export const littleBullPermissionMap = {
  readAreas: 'little_bull.areas.read',
  manageWorkspaces: 'little_bull.workspaces.manage',
  readDocuments: 'little_bull.documents.read',
  uploadDocuments: 'little_bull.documents.upload',
  deleteDocuments: 'little_bull.documents.delete',
  query: 'little_bull.query',
  readAssistants: 'little_bull.assistants.read',
  readActivity: 'little_bull.activity.read',
  readApprovals: 'little_bull.approvals.read',
  decideApprovals: 'little_bull.approvals.decide',
  readAudit: 'little_bull.audit.read',
  manageModels: 'little_bull.models.manage',
  manageAgents: 'little_bull.agents.manage',
  readConversations: 'little_bull.conversations.read',
  saveConversations: 'little_bull.conversations.save',
  exportConversations: 'little_bull.conversations.export',
  suggestCorrelations: 'little_bull.correlations.suggest',
  decideCorrelations: 'little_bull.correlations.decide'
} as const

export const littleBullPagePermissionRules: Partial<Record<LittleBullPage, string[]>> = {
  perguntar: [littleBullPermissionMap.query],
  workspaces: [littleBullPermissionMap.readAreas, littleBullPermissionMap.manageWorkspaces],
  grupos: [littleBullPermissionMap.readDocuments],
  subgrupos: [littleBullPermissionMap.readDocuments],
  conhecimento: [littleBullPermissionMap.readDocuments],
  notas: [littleBullPermissionMap.readDocuments],
  inbox: [littleBullPermissionMap.readDocuments],
  daily: [littleBullPermissionMap.readDocuments],
  canvas: [littleBullPermissionMap.readDocuments],
  mocs: [littleBullPermissionMap.readDocuments],
  trilhas: [littleBullPermissionMap.readDocuments],
  grafo: [littleBullPermissionMap.readDocuments],
  'agent-builder': [littleBullPermissionMap.manageAgents],
  modelos: [littleBullPermissionMap.manageModels],
  custos: [littleBullPermissionMap.readAudit],
  jobs: [littleBullPermissionMap.readActivity],
  juridico: [littleBullPermissionMap.readDocuments],
  relatorios: [littleBullPermissionMap.exportConversations],
  auditoria: [littleBullPermissionMap.readAudit],
  aprovacoes: [littleBullPermissionMap.readApprovals, littleBullPermissionMap.decideApprovals],
  assistentes: [littleBullPermissionMap.readAssistants],
  atividade: [littleBullPermissionMap.readActivity],
  admin: [
    littleBullPermissionMap.readApprovals,
    littleBullPermissionMap.decideApprovals,
    littleBullPermissionMap.readAudit,
    littleBullPermissionMap.manageModels,
    littleBullPermissionMap.manageAgents,
    littleBullPermissionMap.readConversations,
    littleBullPermissionMap.suggestCorrelations,
    littleBullPermissionMap.decideCorrelations
  ]
}

export const hasLittleBullPermission = (
  principal: LittleBullPrincipal | null,
  permission: string
): boolean => {
  if (!principal) return false
  return principal.is_master_global || principal.permissions.includes('*') || principal.permissions.includes(permission)
}

export const hasAnyLittleBullPermission = (
  principal: LittleBullPrincipal | null,
  permissions: string[]
): boolean => permissions.some((permission) => hasLittleBullPermission(principal, permission))

export const canLoadLittleBullKnowledgeTaxonomy = (
  principal: LittleBullPrincipal | null
): boolean => hasLittleBullPermission(principal, littleBullPermissionMap.readAreas)

export const canUseLittleBullClassifiedUpload = (
  principal: LittleBullPrincipal | null
): boolean => (
  hasLittleBullPermission(principal, littleBullPermissionMap.uploadDocuments)
  && canLoadLittleBullKnowledgeTaxonomy(principal)
)

export const canAccessLittleBullPage = (
  principal: LittleBullPrincipal | null,
  page: LittleBullPage
): boolean => {
  const permissions = littleBullPagePermissionRules[page]
  if (!permissions) return true
  return hasAnyLittleBullPermission(principal, permissions)
}

export const visibleLittleBullPageIdsFor = (
  principal: LittleBullPrincipal | null,
  pageIds: LittleBullPage[]
): LittleBullPage[] => pageIds.filter((page) => canAccessLittleBullPage(principal, page))

export const fallbackLittleBullPageFor = (
  principal: LittleBullPrincipal | null,
  pageIds: LittleBullPage[],
  defaultPage: LittleBullPage = 'inicio'
): LittleBullPage => visibleLittleBullPageIdsFor(principal, pageIds)[0] ?? defaultPage

export const filterLittleBullSubgroupsForGroup = (
  subgroups: LittleBullKnowledgeSubgroup[],
  groupId: string
): LittleBullKnowledgeSubgroup[] => {
  if (!groupId) return []
  return subgroups.filter((subgroup) => subgroup.group_id === groupId)
}

export const isLittleBullUploadReady = ({
  canUpload,
  groupId,
  subgroupId
}: {
  canUpload: boolean
  groupId: string
  subgroupId: string
}): boolean => canUpload && groupId.length > 0 && subgroupId.length > 0

export const sanitizeLittleBullUploadSelection = ({
  groupId,
  subgroupId,
  groups,
  subgroups
}: {
  groupId: string
  subgroupId: string
  groups: LittleBullKnowledgeGroup[]
  subgroups: LittleBullKnowledgeSubgroup[]
}): { groupId: string; subgroupId: string } => {
  if (!groupId || !groups.some((group) => group.group_id === groupId)) {
    return { groupId: '', subgroupId: '' }
  }

  const filteredSubgroups = filterLittleBullSubgroupsForGroup(subgroups, groupId)
  if (!subgroupId || !filteredSubgroups.some((subgroup) => subgroup.subgroup_id === subgroupId)) {
    return { groupId, subgroupId: '' }
  }

  return { groupId, subgroupId }
}
