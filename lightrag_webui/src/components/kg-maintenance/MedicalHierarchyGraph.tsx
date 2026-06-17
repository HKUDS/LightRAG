import Button from '@/components/ui/Button'
import type { KBIterationGraphResponse } from '@/api/lightrag'
import type { KGMaintenanceSelectedItem } from '@/stores/kgMaintenance'
import { cn } from '@/lib/utils'
import { buildKGMaintenanceGraphView } from './kgMaintenanceGraph'
import { useMemo, useState } from 'react'

type ViewMode = 'hierarchy' | 'raw' | 'evidence' | 'quality'

interface MedicalHierarchyGraphProps {
  graph: KBIterationGraphResponse | null
  onSelectItem: (item: KGMaintenanceSelectedItem) => void
}

const modes: Array<{ id: ViewMode; label: string }> = [
  { id: 'hierarchy', label: '医学层级' },
  { id: 'raw', label: '原始抽取' },
  { id: 'evidence', label: '证据' },
  { id: 'quality', label: '质量' }
]

export default function MedicalHierarchyGraph({
  graph,
  onSelectItem
}: MedicalHierarchyGraphProps) {
  const [mode, setMode] = useState<ViewMode>('hierarchy')
  const view = useMemo(
    () => (graph ? buildKGMaintenanceGraphView(graph) : { nodes: [], edges: [] }),
    [graph]
  )

  if (!graph) {
    return (
      <section className="border-border/70 bg-muted/20 rounded-lg border p-6">
        <h2 className="text-sm font-semibold">Medical Graph</h2>
        <p className="text-muted-foreground mt-2 text-sm">Run KB iteration review first.</p>
      </section>
    )
  }

  return (
    <section className="flex h-full min-h-[640px] flex-col gap-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">Medical Graph</h2>
          <p className="text-muted-foreground mt-1 text-sm">
            {graph.nodes.length} nodes · {graph.edges.length} directional relations
          </p>
        </div>
        <div className="bg-muted flex rounded-md p-1">
          {modes.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setMode(item.id)}
              className={cn(
                'rounded-sm px-3 py-1.5 text-sm transition-colors',
                mode === item.id ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground'
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="border-border/70 bg-background relative min-h-[560px] flex-1 overflow-hidden rounded-lg border">
        <svg className="h-full min-h-[560px] w-full" viewBox="0 0 720 520" role="img">
          <title>Medical knowledge graph hierarchy</title>
          <g>
            {view.edges.slice(0, 160).map((edge) => {
              const risk = edge.qualityFlags?.includes('generic_relation')
              const missing = edge.evidenceStatus === 'missing'
              const midX = (edge.sourceX + edge.targetX) / 2
              const midY = (edge.sourceY + edge.targetY) / 2
              const edgeLabel = `${edge.sourceLabel || edge.source} ${edge.label} ${edge.targetLabel || edge.target}`
              const selectEdge = () => onSelectItem({ kind: 'edge', id: edge.id })
              return (
                <g
                  key={edge.id}
                  role="button"
                  tabIndex={0}
                  aria-label={edgeLabel}
                  onClick={selectEdge}
                  onKeyDown={(event) => {
                    if (event.key !== 'Enter' && event.key !== ' ') return
                    event.preventDefault()
                    selectEdge()
                  }}
                  className="cursor-pointer outline-none"
                >
                  <line
                    data-testid="kg-maintenance-edge-hit-target"
                    x1={edge.sourceX}
                    y1={edge.sourceY}
                    x2={edge.targetX}
                    y2={edge.targetY}
                    stroke="transparent"
                    strokeWidth={12}
                    pointerEvents="stroke"
                  />
                  <line
                    x1={edge.sourceX}
                    y1={edge.sourceY}
                    x2={edge.targetX}
                    y2={edge.targetY}
                    className={cn(
                      'cursor-pointer stroke-slate-300 dark:stroke-slate-700',
                      mode === 'quality' && risk && 'stroke-amber-500',
                      mode === 'evidence' && missing && 'stroke-rose-500'
                    )}
                    strokeWidth={risk || missing ? 2.5 : 1.4}
                    strokeDasharray={missing ? '5 5' : undefined}
                  />
                  <text
                    x={midX}
                    y={midY}
                    textAnchor="middle"
                    className="pointer-events-none fill-slate-600 text-[10px] dark:fill-slate-300"
                  >
                    {edge.label}
                  </text>
                </g>
              )
            })}
          </g>
          <g>
            {view.nodes.slice(0, 220).map((node) => {
              const missing = node.evidenceStatus === 'missing'
              const risk = node.qualityFlags?.length
              return (
                <g
                  key={node.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelectItem({ kind: 'node', id: node.id })}
                  className="cursor-pointer outline-none"
                >
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={node.size / 2}
                    className={cn(
                      'fill-background stroke-slate-300 transition-colors dark:stroke-slate-700',
                      node.role === 'disease' && 'fill-emerald-500 stroke-emerald-200',
                      node.role === 'category' && 'fill-cyan-500 stroke-cyan-200',
                      node.role === 'subgroup' && 'fill-indigo-500 stroke-indigo-200',
                      mode === 'evidence' && missing && 'stroke-rose-500',
                      mode === 'quality' && risk && 'stroke-amber-500'
                    )}
                    strokeWidth={missing || risk ? 3 : 1.5}
                  />
                  <text
                    x={node.x}
                    y={node.y + node.size / 2 + 13}
                    textAnchor="middle"
                    className="pointer-events-none fill-foreground text-[11px]"
                  >
                    {node.label || node.id}
                  </text>
                </g>
              )
            })}
          </g>
        </svg>

        <div className="absolute right-3 bottom-3 w-64 rounded-lg border border-border/70 bg-background/95 p-3 text-xs shadow-sm">
          <div className="font-medium">Legend</div>
          <div className="mt-2 grid gap-1">
            <LegendItem color="bg-emerald-500" label="Disease center, largest size" />
            <LegendItem color="bg-cyan-500" label="Medical category, medium size" />
            <LegendItem color="bg-indigo-500" label="Subgroup or bridge layer" />
            <LegendItem color="bg-background border border-border" label="Source-grounded fact" />
            <LegendItem color="bg-rose-500" label="Missing evidence highlight" />
            <LegendItem color="bg-amber-500" label="Quality risk highlight" />
          </div>
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" size="sm" onClick={() => onSelectItem(null)}>
          Clear Selection
        </Button>
      </div>
    </section>
  )
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className={cn('size-3 rounded-full', color)} />
      <span className="text-muted-foreground">{label}</span>
    </div>
  )
}
