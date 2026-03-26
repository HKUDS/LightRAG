import type { ReactNode } from 'react'
import type { GraphWorkbenchQueryRequest } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import Checkbox from '@/components/ui/Checkbox'
import Input from '@/components/ui/Input'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { getDefaultGraphWorkbenchFilterDraft, useGraphWorkbenchStore } from '@/stores/graphWorkbench'
import { useGraphStore } from '@/stores/graph'
import GraphWorkbenchSummary from './GraphWorkbenchSummary'

type DraftSection = keyof GraphWorkbenchQueryRequest

const listFields = new Set([
  'entity_types',
  'relation_types',
  'source_entity_types',
  'target_entity_types',
  'file_paths'
])

const nullableNumberFields = new Set(['degree_min', 'degree_max', 'weight_min', 'weight_max'])

const parseListInput = (value: string): string[] =>
  value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

const parseNumberInput = (value: string): number | null => {
  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : null
}

const cloneDraft = (query: GraphWorkbenchQueryRequest): GraphWorkbenchQueryRequest => ({
  scope: { ...query.scope },
  node_filters: {
    ...query.node_filters,
    entity_types: [...query.node_filters.entity_types]
  },
  edge_filters: {
    ...query.edge_filters,
    relation_types: [...query.edge_filters.relation_types],
    source_entity_types: [...query.edge_filters.source_entity_types],
    target_entity_types: [...query.edge_filters.target_entity_types]
  },
  source_filters: {
    ...query.source_filters,
    file_paths: [...query.source_filters.file_paths]
  },
  view_options: { ...query.view_options }
})

const normalizeScopeNumber = (
  rawValue: string,
  currentValue: number,
  minValue: number
): number => {
  const trimmed = rawValue.trim()
  if (!trimmed) {
    return currentValue
  }
  const parsed = Number.parseInt(rawValue, 10)
  if (!Number.isFinite(parsed)) {
    return currentValue
  }
  return Math.max(minValue, parsed)
}

export const updateDraftFromInput = <
  TSection extends DraftSection,
  TField extends keyof GraphWorkbenchQueryRequest[TSection]
>(
  draft: GraphWorkbenchQueryRequest,
  section: TSection,
  field: TField,
  rawValue: string | boolean
): GraphWorkbenchQueryRequest => {
  const nextDraft = cloneDraft(draft)
  const key = String(field)
  const sectionDraft = nextDraft[section] as Record<string, unknown>
  if (typeof rawValue === 'boolean') {
    sectionDraft[key] = rawValue
    return nextDraft
  }

  if (listFields.has(key)) {
    sectionDraft[key] = parseListInput(rawValue)
    return nextDraft
  }

  if (nullableNumberFields.has(key)) {
    sectionDraft[key] = parseNumberInput(rawValue)
    return nextDraft
  }

  if (section === 'scope' && key === 'max_depth') {
    sectionDraft[key] = normalizeScopeNumber(
      rawValue,
      Number(nextDraft.scope.max_depth),
      1
    )
    return nextDraft
  }

  if (section === 'scope' && key === 'max_nodes') {
    sectionDraft[key] = normalizeScopeNumber(
      rawValue,
      Number(nextDraft.scope.max_nodes),
      1
    )
    return nextDraft
  }

  if (section === 'source_filters' && (key === 'time_from' || key === 'time_to')) {
    const trimmed = rawValue.trim()
    sectionDraft[key] = trimmed || null
    return nextDraft
  }

  sectionDraft[key] = rawValue
  return nextDraft
}

export const applyWorkbenchFilters = () => {
  useGraphWorkbenchStore.getState().applyFilterDraft()
}

export const resetWorkbenchFilters = () => {
  const defaults = getDefaultGraphWorkbenchFilterDraft()
  const store = useGraphWorkbenchStore.getState()
  store.setFilterDraft(defaults)
  store.applyFilterDraft()
}

const Section = ({ title, children }: { title: string; children: ReactNode }) => (
  <section className="bg-background/70 rounded-lg border p-3">
    <h3 className="mb-2 text-sm font-semibold">{title}</h3>
    <div className="space-y-2">{children}</div>
  </section>
)

const FieldLabel = ({ children }: { children: ReactNode }) => (
  <label className="text-muted-foreground block text-[11px] font-medium tracking-wide uppercase">
    {children}
  </label>
)

const TextField = ({
  label,
  value,
  onChange,
  type = 'text',
  placeholder
}: {
  label: string
  value: string | number
  onChange: (value: string) => void
  type?: 'text' | 'number' | 'datetime-local'
  placeholder?: string
}) => (
  <div className="space-y-1">
    <FieldLabel>{label}</FieldLabel>
    <Input type={type} value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} />
  </div>
)

const ToggleField = ({
  label,
  checked,
  onCheckedChange
}: {
  label: string
  checked: boolean
  onCheckedChange: (checked: boolean) => void
}) => (
  <label className="flex items-center gap-2 text-sm">
    <Checkbox checked={checked} onCheckedChange={(next) => onCheckedChange(next === true)} />
    <span>{label}</span>
  </label>
)

export const FilterWorkbench = () => {
  const filterDraft = useGraphWorkbenchStore.use.filterDraft()
  const appliedQuery = useGraphWorkbenchStore.use.appliedQuery()
  const queryVersion = useGraphWorkbenchStore.use.queryVersion()
  const setFilterDraft = useGraphWorkbenchStore.use.setFilterDraft()
  const rawGraph = useGraphStore.use.rawGraph()

  const nodeCount = rawGraph?.nodes.length ?? 0
  const edgeCount = rawGraph?.edges.length ?? 0

  const updateField = <
    TSection extends DraftSection,
    TField extends keyof GraphWorkbenchQueryRequest[TSection]
  >(
    section: TSection,
    field: TField,
    value: string | boolean
  ) => {
    const nextDraft = updateDraftFromInput(filterDraft, section, field, value)
    setFilterDraft(nextDraft)
  }

  return (
    <div className="bg-background/80 h-full rounded-xl border backdrop-blur-sm">
      <div className="flex h-full flex-col gap-3 p-3">
        <GraphWorkbenchSummary
          draft={filterDraft}
          appliedQuery={appliedQuery}
          queryVersion={queryVersion}
          nodeCount={nodeCount}
          edgeCount={edgeCount}
        />

        <ScrollArea className="min-h-0 flex-1 pr-2">
          <div className="space-y-3 pr-1">
            <Section title="Node Filters">
              <TextField
                label="Entity Types"
                value={filterDraft.node_filters.entity_types.join(', ')}
                placeholder="PERSON, ORGANIZATION"
                onChange={(value) => updateField('node_filters', 'entity_types', value)}
              />
              <TextField
                label="Name Query"
                value={filterDraft.node_filters.name_query}
                onChange={(value) => updateField('node_filters', 'name_query', value)}
              />
              <TextField
                label="Description Query"
                value={filterDraft.node_filters.description_query}
                onChange={(value) => updateField('node_filters', 'description_query', value)}
              />
              <div className="grid grid-cols-2 gap-2">
                <TextField
                  label="Degree Min"
                  type="number"
                  value={filterDraft.node_filters.degree_min ?? ''}
                  onChange={(value) => updateField('node_filters', 'degree_min', value)}
                />
                <TextField
                  label="Degree Max"
                  type="number"
                  value={filterDraft.node_filters.degree_max ?? ''}
                  onChange={(value) => updateField('node_filters', 'degree_max', value)}
                />
              </div>
              <ToggleField
                label="Isolated Only"
                checked={filterDraft.node_filters.isolated_only}
                onCheckedChange={(checked) => updateField('node_filters', 'isolated_only', checked)}
              />
            </Section>

            <Section title="Edge Filters">
              <TextField
                label="Relation Types"
                value={filterDraft.edge_filters.relation_types.join(', ')}
                placeholder="owns, partner_of"
                onChange={(value) => updateField('edge_filters', 'relation_types', value)}
              />
              <TextField
                label="Keyword Query"
                value={filterDraft.edge_filters.keyword_query}
                onChange={(value) => updateField('edge_filters', 'keyword_query', value)}
              />
              <div className="grid grid-cols-2 gap-2">
                <TextField
                  label="Weight Min"
                  type="number"
                  value={filterDraft.edge_filters.weight_min ?? ''}
                  onChange={(value) => updateField('edge_filters', 'weight_min', value)}
                />
                <TextField
                  label="Weight Max"
                  type="number"
                  value={filterDraft.edge_filters.weight_max ?? ''}
                  onChange={(value) => updateField('edge_filters', 'weight_max', value)}
                />
              </div>
              <TextField
                label="Source Entity Types"
                value={filterDraft.edge_filters.source_entity_types.join(', ')}
                onChange={(value) => updateField('edge_filters', 'source_entity_types', value)}
              />
              <TextField
                label="Target Entity Types"
                value={filterDraft.edge_filters.target_entity_types.join(', ')}
                onChange={(value) => updateField('edge_filters', 'target_entity_types', value)}
              />
            </Section>

            <Section title="Scope Filters">
              <TextField
                label="Start Label"
                value={filterDraft.scope.label}
                onChange={(value) => updateField('scope', 'label', value)}
              />
              <div className="grid grid-cols-2 gap-2">
                <TextField
                  label="Max Depth"
                  type="number"
                  value={filterDraft.scope.max_depth}
                  onChange={(value) => updateField('scope', 'max_depth', value)}
                />
                <TextField
                  label="Max Nodes"
                  type="number"
                  value={filterDraft.scope.max_nodes}
                  onChange={(value) => updateField('scope', 'max_nodes', value)}
                />
              </div>
              <ToggleField
                label="Only Matched Neighborhood"
                checked={filterDraft.scope.only_matched_neighborhood}
                onCheckedChange={(checked) =>
                  updateField('scope', 'only_matched_neighborhood', checked)
                }
              />
            </Section>

            <Section title="Source Filters">
              <TextField
                label="Source ID Query"
                value={filterDraft.source_filters.source_id_query}
                onChange={(value) => updateField('source_filters', 'source_id_query', value)}
              />
              <TextField
                label="File Paths"
                value={filterDraft.source_filters.file_paths.join(', ')}
                placeholder="/inputs/a.md, /inputs/b.md"
                onChange={(value) => updateField('source_filters', 'file_paths', value)}
              />
              <div className="grid grid-cols-2 gap-2">
                <TextField
                  label="Time From"
                  type="datetime-local"
                  value={filterDraft.source_filters.time_from ?? ''}
                  onChange={(value) => updateField('source_filters', 'time_from', value)}
                />
                <TextField
                  label="Time To"
                  type="datetime-local"
                  value={filterDraft.source_filters.time_to ?? ''}
                  onChange={(value) => updateField('source_filters', 'time_to', value)}
                />
              </div>
            </Section>

            <Section title="View Controls">
              <ToggleField
                label="Show Nodes Only"
                checked={filterDraft.view_options.show_nodes_only}
                onCheckedChange={(checked) => updateField('view_options', 'show_nodes_only', checked)}
              />
              <ToggleField
                label="Show Edges Only"
                checked={filterDraft.view_options.show_edges_only}
                onCheckedChange={(checked) => updateField('view_options', 'show_edges_only', checked)}
              />
              <ToggleField
                label="Hide Low Weight Edges"
                checked={filterDraft.view_options.hide_low_weight_edges}
                onCheckedChange={(checked) =>
                  updateField('view_options', 'hide_low_weight_edges', checked)
                }
              />
              <ToggleField
                label="Hide Empty Description"
                checked={filterDraft.view_options.hide_empty_description}
                onCheckedChange={(checked) =>
                  updateField('view_options', 'hide_empty_description', checked)
                }
              />
              <ToggleField
                label="Highlight Matches"
                checked={filterDraft.view_options.highlight_matches}
                onCheckedChange={(checked) => updateField('view_options', 'highlight_matches', checked)}
              />
            </Section>
          </div>
        </ScrollArea>

        <div className="flex items-center gap-2">
          <Button size="sm" className="flex-1" onClick={applyWorkbenchFilters}>
            Apply
          </Button>
          <Button size="sm" variant="outline" className="flex-1" onClick={resetWorkbenchFilters}>
            Reset
          </Button>
        </div>
      </div>
    </div>
  )
}

export default FilterWorkbench
