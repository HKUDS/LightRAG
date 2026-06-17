import { describe, expect, test } from 'bun:test'
import { useKGMaintenanceStore } from './kgMaintenance'

describe('KG maintenance store', () => {
  test('tracks active section, selected item, workspace, and run id', () => {
    useKGMaintenanceStore.getState().setActiveSection('graph')
    useKGMaintenanceStore.getState().setSelectedItem({ kind: 'node', id: 'persistent-fever' })
    useKGMaintenanceStore.getState().setSelectedWorkspace('influenza_medical_v1')
    useKGMaintenanceStore.getState().setLatestRunId('latest')

    expect(useKGMaintenanceStore.getState().activeSection).toBe('graph')
    expect(useKGMaintenanceStore.getState().selectedItem).toEqual({
      kind: 'node',
      id: 'persistent-fever'
    })
    expect(useKGMaintenanceStore.getState().selectedWorkspace).toBe('influenza_medical_v1')
    expect(useKGMaintenanceStore.getState().latestRunId).toBe('latest')
  })
})
