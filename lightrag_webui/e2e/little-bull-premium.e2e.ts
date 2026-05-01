import { expect, test, type Page } from '@playwright/test'

const principal = {
  user_id: 'user-visual',
  sub: 'user-visual',
  tenant_id: 'tenant-visual',
  is_master_global: true,
  roles: ['MASTER'],
  workspace_ids: ['workspace-visual'],
  permission_version: 1,
  permissions: ['*']
}

const workspace = {
  id: 'workspace-visual',
  label: 'Visual Workspace',
  slug: 'visual-workspace',
  description: 'Workspace mockado para smoke visual',
  privacy: 'workspace',
  document_count: 1,
  ready_count: 1,
  processing_count: 0,
  accent: '#2563eb',
  emoji: 'LB'
}

const jsonResponse = (body: unknown) => ({
  status: 200,
  contentType: 'application/json',
  body: JSON.stringify(body)
})

const fakeJwt = () => {
  const header = Buffer.from(JSON.stringify({ alg: 'none', typ: 'JWT' })).toString('base64')
  const payload = Buffer.from(JSON.stringify({ sub: 'user-visual', exp: 4_102_444_800 })).toString('base64')
  return `${header}.${payload}.signature`
}

const mockLittleBullApi = async (page: Page) => {
  await page.route('**/auth/me', async (route) => {
    await route.fulfill(jsonResponse(principal))
  })

  await page.route('**/little-bull/**', async (route) => {
    const url = new URL(route.request().url())
    const path = url.pathname

    if (path === '/little-bull/me') {
      await route.fulfill(jsonResponse(principal))
      return
    }
    if (path === '/little-bull/areas') {
      await route.fulfill(jsonResponse({ areas: [workspace] }))
      return
    }
    if (path === '/little-bull/documents') {
      await route.fulfill(jsonResponse({
        documents: [{
          id: 'document-visual',
          file_path: 'visual.md',
          title: 'Visual Contract',
          status: 'processed',
          content_summary: 'Resumo mockado para smoke visual.',
          content_length: 1200,
          group_id: 'group-visual',
          subgroup_id: 'subgroup-visual',
          registry_document_id: 'registry-document-visual',
          metadata: {}
        }],
        total_count: 1,
        status_counts: { processed: 1 }
      }))
      return
    }
    if (path === '/little-bull/knowledge-groups') {
      await route.fulfill(jsonResponse({
        groups: [{
          group_id: 'group-visual',
          workspace_id: 'workspace-visual',
          slug: 'visual',
          name: 'Visual',
          description: 'Grupo mockado',
          privacy: 'workspace',
          color: '#2563eb',
          metadata: {}
        }]
      }))
      return
    }
    if (path === '/little-bull/knowledge-subgroups') {
      await route.fulfill(jsonResponse({
        subgroups: [{
          subgroup_id: 'subgroup-visual',
          workspace_id: 'workspace-visual',
          group_id: 'group-visual',
          slug: 'contracts',
          name: 'Contracts',
          description: 'Subgrupo mockado',
          privacy: 'workspace',
          metadata: {}
        }]
      }))
      return
    }
    if (path === '/little-bull/activity') {
      await route.fulfill(jsonResponse({ activity: [] }))
      return
    }
    if (path === '/little-bull/assistants') {
      await route.fulfill(jsonResponse({ assistants: [] }))
      return
    }
    if (path === '/little-bull/dossiers') {
      await route.fulfill(jsonResponse({ dossiers: [] }))
      return
    }
    if (path === '/little-bull/legal/extractions') {
      await route.fulfill(jsonResponse({ runs: [] }))
      return
    }
    if (path === '/little-bull/costs/summary') {
      await route.fulfill(jsonResponse({
        workspace_id: 'workspace-visual',
        currency: 'USD',
        periods: {},
        by_user: [],
        by_agent: [],
        by_model: [],
        by_group_subgroup: [],
        by_operation: []
      }))
      return
    }
    if (path === '/little-bull/admin/models') {
      await route.fulfill(jsonResponse({ models: [] }))
      return
    }
    if (path === '/little-bull/admin/embedding-models') {
      await route.fulfill(jsonResponse({ models: [] }))
      return
    }
    if (path === '/little-bull/admin/knowledge-bases') {
      await route.fulfill(jsonResponse({ knowledge_bases: [] }))
      return
    }
    if (path === '/little-bull/admin/agents') {
      await route.fulfill(jsonResponse({ agents: [] }))
      return
    }
    if (path === '/little-bull/conversations') {
      await route.fulfill(jsonResponse({ conversations: [] }))
      return
    }
    if (path === '/little-bull/correlation-suggestions') {
      await route.fulfill(jsonResponse({ suggestions: [] }))
      return
    }

    await route.fulfill(jsonResponse({}))
  })

  await page.route('**/approvals', async (route) => {
    await route.fulfill(jsonResponse({ approvals: [] }))
  })
  await page.route('**/audit/events?**', async (route) => {
    await route.fulfill(jsonResponse({ events: [] }))
  })
}

test.describe('Little Bull Premium visual smoke', () => {
  test.beforeEach(async ({ page }) => {
    await mockLittleBullApi(page)
    await page.addInitScript((token) => {
      localStorage.setItem('LIGHTRAG-API-TOKEN', token)
    }, fakeJwt())
  })

  for (const viewport of [
    { name: 'mobile', width: 390, height: 844 },
    { name: 'tablet', width: 768, height: 1024 },
    { name: 'desktop', width: 1440, height: 1000 }
  ]) {
    test(`renders premium shell without blank viewport on ${viewport.name}`, async ({ page }) => {
      await page.setViewportSize(viewport)
      await page.goto('/#/little-bull')

      await expect(page.getByRole('heading', { name: 'Little Bull operacional' })).toBeVisible()
      await expect(page.getByRole('button', { name: /Visual Workspace/ })).toBeVisible()

      const screenshot = await page.screenshot({
        path: `/tmp/trag-little-bull-premium-${viewport.name}.png`
      })
      expect(screenshot.byteLength).toBeGreaterThan(20_000)
    })
  }
})
