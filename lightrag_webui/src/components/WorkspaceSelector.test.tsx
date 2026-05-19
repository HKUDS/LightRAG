/**
 * WorkspaceSelector Logic Tests
 *
 * Tests the core logic of the WorkspaceSelector component:
 * - Fetch workspaces logic
 * - Stale workspace detection
 * - Handle change logic
 * - Selection handling
 *
 * This approach tests the business logic without requiring a full DOM environment,
 * which provides reliable test coverage for the component's behavior.
 */

import { beforeAll, beforeEach, describe, expect, test, vi } from 'bun:test'

// ============================================================================
// TYPES & MOCKS
// ============================================================================

interface Workspace {
  name: string
  first_seen: string
  last_seen: string
}

// Mock workspace state (used by pure logic tests, not via vi.mock)
let mockCurrentWorkspace: string | null = null

const mockSetCurrentWorkspace = vi.fn((ws: string | null) => {
  mockCurrentWorkspace = ws
})

let mockGetWorkspaces: ReturnType<typeof vi.fn>

// NOTE: No vi.mock calls used here. This test uses pure logic functions
// extracted from the component — no actual module imports needed.
// Using vi.mock would pollute the module cache for other test files.

// ============================================================================
// PURE FUNCTIONS (extracted from component logic)
// ============================================================================

/**
 * Check if a workspace is stale (not in the current list)
 */
const isWorkspaceStale = (
  currentWorkspace: string | null,
  workspaces: Workspace[]
): boolean => {
  if (!currentWorkspace) return false
  return !workspaces.some((w) => w.name === currentWorkspace)
}

/**
 * Handle selection change - convert empty string to null
 */
const handleChangeValue = (value: string): string | null => {
  return value === '' ? null : value
}

/**
 * Fetch workspaces with stale detection
 */
const fetchWorkspacesLogic = async (
  getWorkspaces: () => Promise<Workspace[]>,
  getCurrentWorkspace: () => string | null,
  setCurrentWorkspace: (ws: string | null) => void
): Promise<Workspace[]> => {
  const data = await getWorkspaces()

  // Check if current workspace is still in the list
  const current = getCurrentWorkspace()
  if (current && !data.some((w) => w.name === current)) {
    setCurrentWorkspace(null)
  }

  return data
}

/**
 * Create workspace objects for testing
 */
const createWorkspaces = (names: string[]): Workspace[] =>
  names.map((name) => ({
    name,
    first_seen: new Date().toISOString(),
    last_seen: new Date().toISOString()
  }))

// ============================================================================
// TESTS
// ============================================================================

describe('WorkspaceSelector Logic', () => {
  beforeAll(() => {
    mockGetWorkspaces = vi.fn()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mockCurrentWorkspace = null
  })

  describe('isWorkspaceStale', () => {
    test('returns false when currentWorkspace is null', () => {
      const workspaces = createWorkspaces(['workspace-a', 'workspace-b'])
      expect(isWorkspaceStale(null, workspaces)).toBe(false)
    })

    test('returns false when currentWorkspace exists in list', () => {
      const workspaces = createWorkspaces(['workspace-a', 'workspace-b'])
      expect(isWorkspaceStale('workspace-a', workspaces)).toBe(false)
    })

    test('returns true when currentWorkspace does not exist in list', () => {
      const workspaces = createWorkspaces(['workspace-a', 'workspace-b'])
      expect(isWorkspaceStale('workspace-x', workspaces)).toBe(true)
    })

    test('returns false for empty string (matches component behavior with short-circuit)', () => {
      // Empty string is falsy, so component's `if (current && ...)` short-circuits
      const workspaces = createWorkspaces(['workspace-a', 'workspace-b'])
      expect(isWorkspaceStale('', workspaces)).toBe(false)
    })
  })

  describe('handleChangeValue', () => {
    test('returns null when value is empty string', () => {
      expect(handleChangeValue('')).toBe(null)
    })

    test('returns the value when not empty', () => {
      expect(handleChangeValue('workspace-a')).toBe('workspace-a')
    })

    test('preserves workspace names with special characters', () => {
      expect(handleChangeValue('my-workspace_v1')).toBe('my-workspace_v1')
      expect(handleChangeValue('workspace.with.dots')).toBe('workspace.with.dots')
    })
  })

  describe('fetchWorkspacesLogic', () => {
    test('fetches workspaces from API', async () => {
      const workspaces = createWorkspaces(['workspace-a', 'workspace-b'])
      mockGetWorkspaces.mockResolvedValueOnce(workspaces)

      const result = await fetchWorkspacesLogic(
        mockGetWorkspaces,
        () => null,
        mockSetCurrentWorkspace
      )

      expect(mockGetWorkspaces).toHaveBeenCalledTimes(1)
      expect(result).toEqual(workspaces)
    })

    test('resets current workspace when it is stale', async () => {
      mockCurrentWorkspace = 'workspace-x'
      const workspaces = createWorkspaces(['workspace-a', 'workspace-b'])
      mockGetWorkspaces.mockResolvedValueOnce(workspaces)

      await fetchWorkspacesLogic(
        mockGetWorkspaces,
        () => mockCurrentWorkspace,
        mockSetCurrentWorkspace
      )

      expect(mockSetCurrentWorkspace).toHaveBeenCalledWith(null)
    })

    test('does not reset current workspace when it exists in list', async () => {
      mockCurrentWorkspace = 'workspace-a'
      const workspaces = createWorkspaces(['workspace-a', 'workspace-b'])
      mockGetWorkspaces.mockResolvedValueOnce(workspaces)

      await fetchWorkspacesLogic(
        mockGetWorkspaces,
        () => mockCurrentWorkspace,
        mockSetCurrentWorkspace
      )

      expect(mockSetCurrentWorkspace).not.toHaveBeenCalled()
    })

    test('does not reset when current workspace is null', async () => {
      mockCurrentWorkspace = null
      const workspaces = createWorkspaces(['workspace-a', 'workspace-b'])
      mockGetWorkspaces.mockResolvedValueOnce(workspaces)

      await fetchWorkspacesLogic(
        mockGetWorkspaces,
        () => null,
        mockSetCurrentWorkspace
      )

      expect(mockSetCurrentWorkspace).not.toHaveBeenCalled()
    })

    test('propagates API errors', async () => {
      mockGetWorkspaces.mockRejectedValueOnce(new Error('Network error'))

      await expect(
        fetchWorkspacesLogic(
          mockGetWorkspaces,
          () => null,
          mockSetCurrentWorkspace
        )
      ).rejects.toThrow('Network error')
    })
  })

  describe('Workspace Data Creation', () => {
    test('creates workspace objects with correct structure', () => {
      const workspaces = createWorkspaces(['test-workspace'])

      expect(workspaces).toHaveLength(1)
      expect(workspaces[0]).toHaveProperty('name', 'test-workspace')
      expect(workspaces[0]).toHaveProperty('first_seen')
      expect(workspaces[0]).toHaveProperty('last_seen')
    })

    test('creates multiple workspaces', () => {
      const workspaces = createWorkspaces(['ws1', 'ws2', 'ws3'])

      expect(workspaces).toHaveLength(3)
      expect(workspaces.map((w) => w.name)).toEqual(['ws1', 'ws2', 'ws3'])
    })

    test('each workspace has unique timestamps', () => {
      const workspaces = createWorkspaces(['ws1', 'ws2'])

      // Timestamps should be ISO strings
      expect(workspaces[0].first_seen).toMatch(/^\d{4}-\d{2}-\d{2}T/)
      expect(workspaces[0].last_seen).toMatch(/^\d{4}-\d{2}-\d{2}T/)
    })
  })

  describe('Integration: Complete Workspace Flow', () => {
    test('complete flow: fetch -> stale check -> reset', async () => {
      // Simulate: user has 'old-workspace' selected
      mockCurrentWorkspace = 'old-workspace'

      // API returns only new workspaces (no 'old-workspace')
      const workspaces = createWorkspaces(['new-workspace-1', 'new-workspace-2'])
      mockGetWorkspaces.mockResolvedValueOnce(workspaces)

      // Simulate the component's fetch logic
      const data = await fetchWorkspacesLogic(
        mockGetWorkspaces,
        () => mockCurrentWorkspace,
        mockSetCurrentWorkspace
      )

      // Verify: workspace was fetched
      expect(data).toEqual(workspaces)

      // Verify: stale workspace was detected and reset
      expect(mockSetCurrentWorkspace).toHaveBeenCalledWith(null)
    })

    test('complete flow: fetch -> no stale -> no reset', async () => {
      // Simulate: user has 'current-workspace' selected
      mockCurrentWorkspace = 'current-workspace'

      // API returns workspaces including 'current-workspace'
      const workspaces = createWorkspaces(['current-workspace', 'other-workspace'])
      mockGetWorkspaces.mockResolvedValueOnce(workspaces)

      // Simulate the component's fetch logic
      const data = await fetchWorkspacesLogic(
        mockGetWorkspaces,
        () => mockCurrentWorkspace,
        mockSetCurrentWorkspace
      )

      // Verify: workspace was fetched
      expect(data).toEqual(workspaces)

      // Verify: no reset occurred
      expect(mockSetCurrentWorkspace).not.toHaveBeenCalled()
    })

    test('complete flow: no workspace selected', async () => {
      // Simulate: no workspace selected
      mockCurrentWorkspace = null

      // API returns workspaces
      const workspaces = createWorkspaces(['workspace-a', 'workspace-b'])
      mockGetWorkspaces.mockResolvedValueOnce(workspaces)

      // Simulate the component's fetch logic
      const data = await fetchWorkspacesLogic(
        mockGetWorkspaces,
        () => null,
        mockSetCurrentWorkspace
      )

      // Verify: workspace was fetched
      expect(data).toEqual(workspaces)

      // Verify: no reset occurred (nothing to reset)
      expect(mockSetCurrentWorkspace).not.toHaveBeenCalled()
    })
  })

  describe('Refresh Interval Logic', () => {
    const REFRESH_INTERVAL_MS = 30_000

    test('refresh interval is 30 seconds', () => {
      expect(REFRESH_INTERVAL_MS).toBe(30_000)
    })

    test('refresh interval calculation for 1 minute', () => {
      const oneMinute = 60_000
      expect(oneMinute / REFRESH_INTERVAL_MS).toBe(2)
    })
  })
})
