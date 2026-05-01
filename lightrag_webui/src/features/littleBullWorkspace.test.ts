import { describe, expect, test } from 'bun:test'
import type {
  LittleBullKnowledgeGroup,
  LittleBullKnowledgeSubgroup,
  LittleBullPrincipal
} from '@/api/lightrag'
import {
  canLoadLittleBullKnowledgeTaxonomy,
  canAccessLittleBullPage,
  canUseLittleBullClassifiedUpload,
  fallbackLittleBullPageFor,
  fallbackLittleBullAreasForPrincipal,
  filterLittleBullSubgroupsForGroup,
  hasLittleBullPermission,
  isLittleBullUploadReady,
  littleBullPermissionMap,
  sanitizeLittleBullUploadSelection
} from './littleBullWorkspace'

const groups: LittleBullKnowledgeGroup[] = [
  {
    group_id: 'group-a',
    workspace_id: 'workspace-1',
    slug: 'group-a',
    name: 'Group A',
    description: '',
    privacy: 'workspace',
    color: '#2563eb',
    metadata: {}
  },
  {
    group_id: 'group-b',
    workspace_id: 'workspace-1',
    slug: 'group-b',
    name: 'Group B',
    description: '',
    privacy: 'workspace',
    color: '#16a34a',
    metadata: {}
  }
]

const subgroups: LittleBullKnowledgeSubgroup[] = [
  {
    subgroup_id: 'subgroup-a1',
    workspace_id: 'workspace-1',
    group_id: 'group-a',
    slug: 'subgroup-a1',
    name: 'Subgroup A1',
    description: '',
    privacy: 'workspace',
    metadata: {}
  },
  {
    subgroup_id: 'subgroup-b1',
    workspace_id: 'workspace-1',
    group_id: 'group-b',
    slug: 'subgroup-b1',
    name: 'Subgroup B1',
    description: '',
    privacy: 'workspace',
    metadata: {}
  }
]

const principal = (permissions: string[], isMaster = false): LittleBullPrincipal => ({
  user_id: 'user-1',
  sub: 'user-1',
  tenant_id: 'tenant-1',
  is_master_global: isMaster,
  roles: [],
  workspace_ids: ['workspace-1'],
  permission_version: 1,
  permissions
})

describe('littleBullWorkspace classified upload helpers', () => {
  test('filters subgroups by selected group', () => {
    expect(filterLittleBullSubgroupsForGroup(subgroups, 'group-a').map((item) => item.subgroup_id)).toEqual([
      'subgroup-a1'
    ])
    expect(filterLittleBullSubgroupsForGroup(subgroups, '')).toEqual([])
  })

  test('requires permission, group, and subgroup before upload is ready', () => {
    expect(isLittleBullUploadReady({ canUpload: true, groupId: 'group-a', subgroupId: 'subgroup-a1' })).toBe(true)
    expect(isLittleBullUploadReady({ canUpload: false, groupId: 'group-a', subgroupId: 'subgroup-a1' })).toBe(false)
    expect(isLittleBullUploadReady({ canUpload: true, groupId: '', subgroupId: 'subgroup-a1' })).toBe(false)
    expect(isLittleBullUploadReady({ canUpload: true, groupId: 'group-a', subgroupId: '' })).toBe(false)
  })

  test('keeps document access separate from taxonomy needed by classified upload', () => {
    const docsOnly = principal([littleBullPermissionMap.readDocuments])
    const uploaderWithoutTaxonomy = principal([littleBullPermissionMap.readDocuments, littleBullPermissionMap.uploadDocuments])
    const classifiedUploader = principal([
      littleBullPermissionMap.readDocuments,
      littleBullPermissionMap.uploadDocuments,
      littleBullPermissionMap.readAreas
    ])

    expect(canAccessLittleBullPage(docsOnly, 'conhecimento')).toBe(true)
    expect(canLoadLittleBullKnowledgeTaxonomy(docsOnly)).toBe(false)
    expect(canUseLittleBullClassifiedUpload(uploaderWithoutTaxonomy)).toBe(false)
    expect(canUseLittleBullClassifiedUpload(classifiedUploader)).toBe(true)
  })

  test('derives scoped workspace choices when area listing is not allowed', () => {
    const docsOnly = principal([littleBullPermissionMap.readDocuments])
    const areaReader = principal([littleBullPermissionMap.readDocuments, littleBullPermissionMap.readAreas])

    expect(fallbackLittleBullAreasForPrincipal(areaReader)).toEqual([])
    expect(fallbackLittleBullAreasForPrincipal(docsOnly)).toMatchObject([
      {
        id: 'workspace-1',
        label: 'workspace-1',
        privacy: 'scoped'
      }
    ])
  })

  test('clears stale group or subgroup selections', () => {
    expect(sanitizeLittleBullUploadSelection({
      groupId: 'missing-group',
      subgroupId: 'subgroup-a1',
      groups,
      subgroups
    })).toEqual({ groupId: '', subgroupId: '' })

    expect(sanitizeLittleBullUploadSelection({
      groupId: 'group-a',
      subgroupId: 'subgroup-b1',
      groups,
      subgroups
    })).toEqual({ groupId: 'group-a', subgroupId: '' })

    expect(sanitizeLittleBullUploadSelection({
      groupId: 'group-a',
      subgroupId: 'subgroup-a1',
      groups,
      subgroups
    })).toEqual({ groupId: 'group-a', subgroupId: 'subgroup-a1' })
  })
})

describe('littleBullWorkspace premium navigation permissions', () => {
  test('allows master and wildcard principals across premium pages', () => {
    expect(hasLittleBullPermission(principal([], true), littleBullPermissionMap.manageAgents)).toBe(true)
    expect(canAccessLittleBullPage(principal(['*']), 'agent-builder')).toBe(true)
  })

  test('blocks premium pages when required permissions are missing', () => {
    const readOnly = principal([littleBullPermissionMap.readDocuments])

    expect(canAccessLittleBullPage(readOnly, 'conhecimento')).toBe(true)
    expect(canAccessLittleBullPage(readOnly, 'juridico')).toBe(true)
    expect(canAccessLittleBullPage(readOnly, 'custos')).toBe(false)
    expect(canAccessLittleBullPage(readOnly, 'agent-builder')).toBe(false)
  })

  test('selects the first accessible page as fallback', () => {
    const auditOnly = principal([littleBullPermissionMap.readAudit])

    expect(fallbackLittleBullPageFor(auditOnly, ['agent-builder', 'custos', 'conhecimento'])).toBe('custos')
    expect(fallbackLittleBullPageFor(null, ['agent-builder', 'custos'], 'inicio')).toBe('inicio')
  })
})
