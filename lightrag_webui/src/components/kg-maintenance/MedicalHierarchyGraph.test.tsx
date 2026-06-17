import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import MedicalHierarchyGraph from './MedicalHierarchyGraph'

describe('MedicalHierarchyGraph accessibility and hit targets', () => {
  test('renders relation edges as accessible buttons with thick hit targets', () => {
    const markup = renderToStaticMarkup(
      <MedicalHierarchyGraph
        graph={{
          workspace: 'influenza_medical_v1',
          runId: 'latest',
          nodes: [
            { id: 'virus', label: '流感病毒', entity_type: 'Pathogen', properties: {} },
            { id: 'flu', label: '流行性感冒', entity_type: 'Disease', properties: {} }
          ],
          edges: [
            {
              id: 'edge-1',
              source: 'virus',
              target: 'flu',
              label: '病原导致',
              keywords: '病原导致',
              sourceLabel: '流感病毒',
              targetLabel: '流行性感冒'
            }
          ]
        }}
        onSelectItem={() => undefined}
      />
    )

    expect(markup).toContain('role="button"')
    expect(markup).toContain('aria-label="流感病毒 病原导致 流行性感冒"')
    expect(markup).toContain('data-testid="kg-maintenance-edge-hit-target"')
    expect(markup).toContain('stroke-width="12"')
  })
})
