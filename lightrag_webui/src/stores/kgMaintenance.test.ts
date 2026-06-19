import { afterEach, describe, expect, test } from 'bun:test'
import { useKGMaintenanceStore } from './kgMaintenance'

afterEach(() => {
  useKGMaintenanceStore.setState({
    activeSection: 'check',
    selectedItem: null,
    selectedWorkspace: null,
    latestRunId: 'latest'
  })
})

describe('KG maintenance store', () => {
  test('tracks active workflow section, workspace, and run id', () => {
    useKGMaintenanceStore.getState().setActiveSection('validate')
    useKGMaintenanceStore.getState().setSelectedWorkspace('influenza_medical_v1')
    useKGMaintenanceStore.getState().setLatestRunId('review-run-2026-06-18')

    expect(useKGMaintenanceStore.getState().activeSection).toBe('validate')
    expect(useKGMaintenanceStore.getState().selectedWorkspace).toBe('influenza_medical_v1')
    expect(useKGMaintenanceStore.getState().latestRunId).toBe('review-run-2026-06-18')
  })
})
