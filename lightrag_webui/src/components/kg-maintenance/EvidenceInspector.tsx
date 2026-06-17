import type { KBIterationGraphEdge, KBIterationGraphNode } from '@/api/lightrag'

interface EvidenceInspectorProps {
  node: KBIterationGraphNode | null
  edge: KBIterationGraphEdge | null
}

export default function EvidenceInspector({ node, edge }: EvidenceInspectorProps) {
  if (!node && !edge) {
    return (
      <div>
        <h2 className="text-sm font-semibold">Evidence Inspector</h2>
        <p className="text-muted-foreground mt-2 text-sm">
          Select a node or relation to inspect source grounding.
        </p>
        <p className="text-muted-foreground mt-4 text-xs">
          LLM analysis is review material, not medical evidence.
        </p>
      </div>
    )
  }

  const item = node || edge
  const title = node ? node.label || node.id : edge?.label || edge?.id
  const itemKind = node
    ? node.entity_type === 'MedicalGroup'
      ? 'Organization node'
      : 'Fact/entity node'
    : edge?.properties?.edge_kind === 'navigation'
      ? 'Organization edge'
      : 'Factual relation edge'
  const chunkPreview = item?.source_id
    ? `${item.source_id}${item.description ? ` - ${item.description}` : ''}`
    : 'Missing source/chunk reference'

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-sm font-semibold">{title}</h2>
        <p className="text-muted-foreground mt-1 text-xs">{node ? node.entity_type : 'Relation'}</p>
      </div>
      {edge && (
        <InspectorRow
          label="Direction"
          value={`${edge.direction || 'outgoing'} ${edge.sourceLabel || edge.source} -> ${edge.targetLabel || edge.target}`}
        />
      )}
      <InspectorRow label="Review status" value={itemKind} />
      <InspectorRow label="Description" value={item?.description || 'No description'} />
      <InspectorRow label="Source document" value={item?.file_path || 'Missing'} />
      <InspectorRow label="source_id" value={item?.source_id || 'Missing'} />
      <InspectorRow label="file_path" value={item?.file_path || 'Missing'} />
      <InspectorRow label="Chunk / paragraph preview" value={chunkPreview} />
      <InspectorRow label="Evidence" value={item?.evidenceStatus || 'unknown'} />
      <InspectorRow label="Quality Flags" value={item?.qualityFlags?.join(', ') || 'None'} />
      <div className="border-border/70 rounded-lg border p-3 text-xs text-muted-foreground">
        Medical facts should be checked against source files or chunks before approval.
        Descriptions and LLM analysis are review material, not medical evidence.
      </div>
    </div>
  )
}

function InspectorRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 break-words text-sm">{value}</div>
    </div>
  )
}
