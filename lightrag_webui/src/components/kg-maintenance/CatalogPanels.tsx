import type {
  KBIterationEntityCatalogResponse,
  KBIterationGraphEdge,
  KBIterationGraphNode,
  KBIterationRelationCatalogResponse
} from '@/api/lightrag'
import type { KGMaintenanceSelectedItem } from '@/stores/kgMaintenance'
import { useMemo, useState } from 'react'

interface EntityCatalogPanelProps {
  catalog: KBIterationEntityCatalogResponse | null
  onSelect: (item: KGMaintenanceSelectedItem) => void
}

interface RelationCatalogPanelProps {
  catalog: KBIterationRelationCatalogResponse | null
  onSelect: (item: KGMaintenanceSelectedItem) => void
}

const RELATION_FALLBACK = '未标注关系'
const EMPTY_ENTITIES: KBIterationGraphNode[] = []
const EMPTY_RELATIONS: KBIterationGraphEdge[] = []

const relationLabel = (edge: KBIterationGraphEdge) => {
  const label = (edge.label || edge.keywords || '').trim()
  return !label || label === '邻接' ? RELATION_FALLBACK : label
}

export function EntityCatalogPanel({ catalog, onSelect }: EntityCatalogPanelProps) {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<'all' | 'missing-evidence' | 'suspicious'>('all')
  const entities = catalog?.entities ?? EMPTY_ENTITIES
  const filteredEntities = useMemo(
    () =>
      entities.filter((node) => {
        const haystack = `${node.label || ''} ${node.id} ${node.entity_type || ''} ${node.description || ''}`.toLowerCase()
        const matchesQuery = haystack.includes(query.trim().toLowerCase())
        const missingEvidence = !node.file_path || !node.source_id
        const suspicious = isSuspiciousEntity(node)
        if (filter === 'missing-evidence') return matchesQuery && missingEvidence
        if (filter === 'suspicious') return matchesQuery && suspicious
        return matchesQuery
      }),
    [entities, filter, query]
  )

  if (!catalog) {
    return <EmptyPanel title="Entity Catalog" />
  }

  return (
    <section className="space-y-3">
      <PanelHeader
        title="Entity Catalog"
        subtitle={`${entities.length} entities grouped by extracted type`}
      />
      <CatalogToolbar
        query={query}
        onQueryChange={setQuery}
        filter={filter}
        onFilterChange={(value) => setFilter(value as typeof filter)}
        options={[
          ['all', 'All entities'],
          ['missing-evidence', 'Missing evidence'],
          ['suspicious', 'Suspicious']
        ]}
      />
      <StatsStrip
        items={[
          ['Types', catalog.stats?.length || 0],
          ['Missing evidence', entities.filter((node) => !node.file_path || !node.source_id).length],
          ['Suspicious', entities.filter(isSuspiciousEntity).length],
          ['Showing', filteredEntities.length]
        ]}
      />
      <div className="border-border/70 overflow-auto rounded-lg border">
        <div className="bg-muted/40 grid min-w-[760px] grid-cols-[minmax(180px,1.3fr)_120px_minmax(220px,2fr)_160px] gap-3 px-3 py-2 text-xs font-medium">
          <span>Name</span>
          <span>Type</span>
          <span>Description</span>
          <span>Source</span>
        </div>
        <div className="divide-border/70 divide-y">
          {filteredEntities.slice(0, 250).map((node) => (
            <button
              key={node.id}
              type="button"
              onClick={() => onSelect({ kind: 'node', id: node.id })}
              className="hover:bg-accent/40 grid min-w-[760px] w-full grid-cols-[minmax(180px,1.3fr)_120px_minmax(220px,2fr)_160px] gap-3 px-3 py-2 text-left text-sm"
            >
              <span className="truncate font-medium">{node.label || node.id}</span>
              <span className="text-muted-foreground truncate">{node.entity_type || 'Unknown'}</span>
              <span className="text-muted-foreground truncate">
                {node.description || 'No description'}
              </span>
              <span className="text-muted-foreground truncate">
                {node.file_path || 'Missing evidence'}
              </span>
            </button>
          ))}
        </div>
      </div>
      <MarkdownReference content={catalog.catalog} />
    </section>
  )
}

export function RelationCatalogPanel({ catalog, onSelect }: RelationCatalogPanelProps) {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<'all' | 'missing-evidence' | 'generic' | 'high-risk'>(
    'all'
  )
  const relations = catalog?.relations ?? EMPTY_RELATIONS
  const filteredRelations = useMemo(
    () =>
      relations.filter((edge) => {
        const label = relationLabel(edge)
        const haystack = `${edge.sourceLabel || edge.source} ${label} ${edge.targetLabel || edge.target} ${edge.description || ''}`.toLowerCase()
        const matchesQuery = haystack.includes(query.trim().toLowerCase())
        const missingEvidence = !edge.file_path || !edge.source_id
        const generic = label === RELATION_FALLBACK
        const highRisk = Boolean(edge.qualityFlags?.length)
        if (filter === 'missing-evidence') return matchesQuery && missingEvidence
        if (filter === 'generic') return matchesQuery && generic
        if (filter === 'high-risk') return matchesQuery && highRisk
        return matchesQuery
      }),
    [filter, query, relations]
  )

  if (!catalog) {
    return <EmptyPanel title="Relation Catalog" />
  }

  return (
    <section className="space-y-3">
      <PanelHeader
        title="Relation Catalog"
        subtitle={`${relations.length} directional triples`}
      />
      <CatalogToolbar
        query={query}
        onQueryChange={setQuery}
        filter={filter}
        onFilterChange={(value) => setFilter(value as typeof filter)}
        options={[
          ['all', 'All relations'],
          ['missing-evidence', 'Missing evidence'],
          ['generic', 'Generic relation'],
          ['high-risk', 'Quality risk']
        ]}
      />
      <StatsStrip
        items={[
          ['Relation types', catalog.stats?.length || 0],
          ['Missing evidence', relations.filter((edge) => !edge.file_path || !edge.source_id).length],
          ['Generic', relations.filter((edge) => relationLabel(edge) === RELATION_FALLBACK).length],
          ['Showing', filteredRelations.length]
        ]}
      />
      <div className="border-border/70 overflow-auto rounded-lg border">
        <div className="bg-muted/40 grid min-w-[760px] grid-cols-[minmax(180px,1fr)_160px_minmax(180px,1fr)_160px] gap-3 px-3 py-2 text-xs font-medium">
          <span>Source</span>
          <span>Relation</span>
          <span>Target</span>
          <span>Evidence</span>
        </div>
        <div className="divide-border/70 divide-y">
          {filteredRelations.slice(0, 250).map((edge) => (
            <button
              key={edge.id}
              type="button"
              onClick={() => onSelect({ kind: 'edge', id: edge.id })}
              className="hover:bg-accent/40 grid min-w-[760px] w-full grid-cols-[minmax(180px,1fr)_160px_minmax(180px,1fr)_160px] gap-3 px-3 py-2 text-left text-sm"
            >
              <span className="truncate font-medium">{edge.sourceLabel || edge.source}</span>
              <span className="truncate text-emerald-700 dark:text-emerald-300">
                {relationLabel(edge)}
              </span>
              <span className="truncate font-medium">{edge.targetLabel || edge.target}</span>
              <span className="text-muted-foreground truncate">
                {edge.file_path || 'Missing evidence'}
              </span>
            </button>
          ))}
        </div>
      </div>
      <MarkdownReference content={catalog.catalog} />
    </section>
  )
}

function CatalogToolbar({
  query,
  onQueryChange,
  filter,
  onFilterChange,
  options
}: {
  query: string
  onQueryChange: (query: string) => void
  filter: string
  onFilterChange: (filter: string) => void
  options: Array<[string, string]>
}) {
  return (
    <div className="flex flex-wrap gap-2">
      <input
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        className="border-input bg-background h-9 min-w-56 rounded-md border px-3 text-sm"
        placeholder="Search catalog"
      />
      <select
        value={filter}
        onChange={(event) => onFilterChange(event.target.value)}
        className="border-input bg-background h-9 rounded-md border px-3 text-sm"
      >
        {options.map(([value, label]) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </select>
    </div>
  )
}

function StatsStrip({ items }: { items: Array<[string, number]> }) {
  return (
    <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
      {items.map(([label, value]) => (
        <div key={label} className="border-border/70 rounded-md border px-3 py-2">
          <div className="text-muted-foreground text-xs">{label}</div>
          <div className="text-lg font-semibold">{value}</div>
        </div>
      ))}
    </div>
  )
}

function isSuspiciousEntity(node: KBIterationGraphNode) {
  const label = String(node.label || '')
  return (
    !node.file_path ||
    !node.source_id ||
    /^\d+(\.\d+)?\s*(mg|g|ml|h|小时|天|%|次)$/i.test(label) ||
    /^第?\d+页$/.test(label)
  )
}

function PanelHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div>
      <h2 className="text-sm font-semibold">{title}</h2>
      <p className="text-muted-foreground mt-1 text-sm">{subtitle}</p>
    </div>
  )
}

function EmptyPanel({ title }: { title: string }) {
  return (
    <section className="border-border/70 bg-muted/20 rounded-lg border p-6">
      <h2 className="text-sm font-semibold">{title}</h2>
      <p className="text-muted-foreground mt-2 text-sm">Run KB iteration review first.</p>
    </section>
  )
}

function MarkdownReference({ content }: { content: string }) {
  if (!content) return null
  return (
    <details className="border-border/70 rounded-lg border p-3">
      <summary className="cursor-pointer text-sm font-medium">Markdown artifact</summary>
      <pre className="text-muted-foreground mt-3 max-h-72 overflow-auto whitespace-pre-wrap text-xs">
        {content}
      </pre>
    </details>
  )
}
