import { useMemo } from 'react'
import { useGraphStore, RawNodeType, RawEdgeType } from '@/stores/graph'
import { useBackendState } from '@/stores/state'
import Text from '@/components/ui/Text'
import Button from '@/components/ui/Button'
import useLightragGraph from '@/hooks/useLightragGraph'
import { useTranslation } from 'react-i18next'
import { GitBranchPlus, Scissors, Lock } from 'lucide-react'
import EditablePropertyRow from './EditablePropertyRow'
import {
  groupMedicalRelations,
  type GroupedMedicalRelations,
  type MedicalRelation
} from './medicalRelationGroups'

/**
 * Component that view properties of elements in graph.
 */
const PropertiesView = () => {
  const { getNode, getEdge } = useLightragGraph()
  const selectedNode = useGraphStore.use.selectedNode()
  const focusedNode = useGraphStore.use.focusedNode()
  const selectedEdge = useGraphStore.use.selectedEdge()
  const focusedEdge = useGraphStore.use.focusedEdge()
  const graphDataVersion = useGraphStore.use.graphDataVersion()
  const pipelineBusy = useBackendState.use.pipelineBusy()

  const { currentElement, currentType } = useMemo(() => {
    let type: 'node' | 'edge' | null = null
    let element: RawNodeType | RawEdgeType | null = null
    if (focusedNode) {
      type = 'node'
      element = getNode(focusedNode)
    } else if (selectedNode) {
      type = 'node'
      element = getNode(selectedNode)
    } else if (focusedEdge) {
      type = 'edge'
      element = getEdge(focusedEdge, true)
    } else if (selectedEdge) {
      type = 'edge'
      element = getEdge(selectedEdge, true)
    }

    if (element) {
      return {
        currentElement: type === 'node'
          ? refineNodeProperties(element as any)
          : refineEdgeProperties(element as any),
        currentType: type
      }
    }
    return { currentElement: null, currentType: null }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusedNode, selectedNode, focusedEdge, selectedEdge, graphDataVersion, getNode, getEdge])

  if (!currentElement) {
    return <></>
  }
  return (
    <div className="bg-background/80 max-w-sm rounded-lg border-2 p-2 text-xs backdrop-blur-lg">
      {currentType == 'node' ? (
        <NodePropertiesView node={currentElement as any} pipelineBusy={pipelineBusy} />
      ) : (
        <EdgePropertiesView edge={currentElement as any} pipelineBusy={pipelineBusy} />
      )}
    </div>
  )
}

type NodeType = RawNodeType & {
  relationshipGroups: GroupedMedicalRelations[]
}

type EdgeType = RawEdgeType & {
  sourceNode?: RawNodeType
  targetNode?: RawNodeType
}

const refineNodeProperties = (node: RawNodeType): NodeType => {
  const state = useGraphStore.getState()
  const relationships: MedicalRelation[] = []

  if (state.sigmaGraph && state.rawGraph) {
    try {
      if (!state.sigmaGraph.hasNode(node.id)) {
        console.warn('Node not found in sigmaGraph:', node.id)
        return {
          ...node,
          relationshipGroups: []
        }
      }

      const edges = state.sigmaGraph.edges(node.id)

      for (const edgeId of edges) {
        if (!state.sigmaGraph.hasEdge(edgeId)) continue;

        const edge = state.rawGraph.getEdge(edgeId, true)
        if (edge) {
          const isTarget = node.id === edge.source
          const neighbourId = isTarget ? edge.target : edge.source

          if (!state.sigmaGraph.hasNode(neighbourId)) continue;

          const neighbour = state.rawGraph.getNode(neighbourId)
          if (neighbour) {
            relationships.push({
              id: neighbourId,
              label: neighbour.properties['entity_id'] ? neighbour.properties['entity_id'] : neighbour.labels.join(', '),
              edgeId: edge.id,
              selectedNodeId: node.id,
              sourceId: edge.source,
              targetId: edge.target,
              edgeKeywords: edge.properties?.keywords,
              neighborEntityType: neighbour.properties?.entity_type,
              neighborLabels: neighbour.labels
            })
          }
        }
      }
    } catch (error) {
      console.error('Error refining node properties:', error)
    }
  }

  return {
    ...node,
    relationshipGroups: groupMedicalRelations(relationships, state.rawGraph?.metadata)
  }
}

const refineEdgeProperties = (edge: RawEdgeType): EdgeType => {
  const state = useGraphStore.getState()
  let sourceNode: RawNodeType | undefined = undefined
  let targetNode: RawNodeType | undefined = undefined

  if (state.sigmaGraph && state.rawGraph) {
    try {
      if (!state.sigmaGraph.hasEdge(edge.dynamicId)) {
        console.warn('Edge not found in sigmaGraph:', edge.id, 'dynamicId:', edge.dynamicId)
        return {
          ...edge,
          sourceNode: undefined,
          targetNode: undefined
        }
      }

      if (state.sigmaGraph.hasNode(edge.source)) {
        sourceNode = state.rawGraph.getNode(edge.source)
      }

      if (state.sigmaGraph.hasNode(edge.target)) {
        targetNode = state.rawGraph.getNode(edge.target)
      }
    } catch (error) {
      console.error('Error refining edge properties:', error)
    }
  }

  return {
    ...edge,
    sourceNode,
    targetNode
  }
}

const PropertyRow = ({
  name,
  value,
  onClick,
  tooltip,
  nodeId,
  edgeId,
  dynamicId,
  entityId,
  entityType,
  sourceId,
  targetId,
  isEditable = false,
  truncate,
  pipelineBusy = false
}: {
  name: string
  value: any
  onClick?: () => void
  tooltip?: string
  nodeId?: string
  entityId?: string
  edgeId?: string
  dynamicId?: string
  entityType?: 'node' | 'edge'
  sourceId?: string
  targetId?: string
  isEditable?: boolean
  truncate?: string
  pipelineBusy?: boolean
}) => {
  const { t } = useTranslation()

  const getPropertyNameTranslation = (name: string) => {
    const translationKey = `graphPanel.propertiesView.node.propertyNames.${name}`
    const translation = t(translationKey)
    return translation === translationKey ? name : translation
  }

  // Utility function to convert <SEP> to newlines
  const formatValueWithSeparators = (value: any): string => {
    if (typeof value === 'string') {
      return value.replace(/<SEP>/g, ';\n')
    }
    return typeof value === 'string' ? value : JSON.stringify(value, null, 2)
  }

  // Format the value to convert <SEP> to newlines
  const formattedValue = formatValueWithSeparators(value)
  let formattedTooltip = tooltip || formatValueWithSeparators(value)

  // If this is source_id field and truncate info exists, append it to the tooltip
  if (name === 'source_id' && truncate) {
    formattedTooltip += `\n(Truncated: ${truncate})`
  }

  // Use EditablePropertyRow for editable fields (description, entity_id and entity_type)
  if (isEditable && (name === 'description' || name === 'entity_id' || name === 'entity_type'  || name === 'keywords')) {
    return (
      <EditablePropertyRow
        name={name}
        value={value}
        onClick={onClick}
        nodeId={nodeId}
        entityId={entityId}
        edgeId={edgeId}
        dynamicId={dynamicId}
        entityType={entityType}
        sourceId={sourceId}
        targetId={targetId}
        isEditable={true}
        pipelineBusy={pipelineBusy}
        tooltip={tooltip || (typeof value === 'string' ? value : JSON.stringify(value, null, 2))}
      />
    )
  }

  // For non-editable fields, use the regular Text component
  return (
    <div className="flex items-center gap-2">
      <span className="text-primary/60 tracking-wide whitespace-nowrap">
        {getPropertyNameTranslation(name)}
        {name === 'source_id' && truncate && <sup className="text-red-500">†</sup>}
      </span>:
      <Text
        className="hover:bg-primary/20 rounded p-1 overflow-hidden text-ellipsis"
        tooltipClassName="max-w-96 -translate-x-13"
        text={formattedValue}
        tooltip={formattedTooltip}
        side="left"
        onClick={onClick}
      />
    </div>
  )
}

const NodePropertiesView = ({ node, pipelineBusy }: { node: NodeType; pipelineBusy: boolean }) => {
  const { t } = useTranslation()

  const handleExpandNode = () => {
    useGraphStore.getState().triggerNodeExpand(node.id)
  }

  const handlePruneNode = () => {
    useGraphStore.getState().triggerNodePrune(node.id)
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-between items-center">
        <h3 className="text-md pl-1 font-bold tracking-wide text-blue-700">{t('graphPanel.propertiesView.node.title')}</h3>
        <div className="flex gap-3">
          {pipelineBusy && (
            <Button
              type="button"
              size="icon"
              variant="ghost"
              aria-label={t('graphPanel.propertiesView.editLockedByPipeline')}
              aria-disabled="true"
              className="h-7 w-7 border border-amber-400 hover:bg-amber-50 dark:border-amber-600 dark:hover:bg-amber-900/40 !cursor-default"
              tooltip={t('graphPanel.propertiesView.editLockedByPipeline')}
              onClick={(e) => e.preventDefault()}
            >
              <Lock className="h-4 w-4 text-amber-600 dark:text-amber-400" />
            </Button>
          )}
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 border border-gray-400 hover:bg-gray-200 dark:border-gray-600 dark:hover:bg-gray-700"
            onClick={handleExpandNode}
            tooltip={t('graphPanel.propertiesView.node.expandNode')}
          >
            <GitBranchPlus className="h-4 w-4 text-gray-700 dark:text-gray-300" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 border border-gray-400 hover:bg-gray-200 dark:border-gray-600 dark:hover:bg-gray-700"
            onClick={handlePruneNode}
            tooltip={t('graphPanel.propertiesView.node.pruneNode')}
          >
            <Scissors className="h-4 w-4 text-gray-900 dark:text-gray-300" />
          </Button>
        </div>
      </div>
      <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
        <PropertyRow name={t('graphPanel.propertiesView.node.id')} value={String(node.id)} />
        <PropertyRow
          name={t('graphPanel.propertiesView.node.labels')}
          value={node.labels.join(', ')}
          onClick={() => {
            useGraphStore.getState().setSelectedNode(node.id, true)
          }}
        />
        <PropertyRow name={t('graphPanel.propertiesView.node.degree')} value={node.degree} />
      </div>
      <h3 className="text-md pl-1 font-bold tracking-wide text-amber-700">{t('graphPanel.propertiesView.node.properties')}</h3>
      <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
        {Object.keys(node.properties)
          .sort()
          .map((name) => {
            if (name === 'created_at' || name === 'truncate') return null; // Hide created_at and truncate properties
            return (
              <PropertyRow
                key={name}
                name={name}
                value={node.properties[name]}
                nodeId={String(node.id)}
                entityId={node.properties['entity_id']}
                entityType="node"
                isEditable={name === 'description' || name === 'entity_id' || name === 'entity_type'}
                truncate={node.properties['truncate']}
                pipelineBusy={pipelineBusy}
              />
            )
          })}
      </div>
      {node.relationshipGroups.length > 0 && (
        <>
          <h3 className="text-md pl-1 font-bold tracking-wide text-emerald-700">
            {t('graphPanel.propertiesView.node.medicalRelationships', '医学关系')}
          </h3>
          <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
            {node.relationshipGroups.map((group) => (
              <div key={group.key} className="mb-2 last:mb-0">
                <div className="px-1 pb-1 font-semibold text-primary/70">
                  {t(
                    `graphPanel.propertiesView.node.medicalGroups.${group.key}`,
                    group.metadataLabel || group.key
                  )}
                  <span className="ml-1 text-primary/40">({group.relations.length})</span>
                </div>
                <div className="flex flex-col gap-1">
                  {group.relations.map((relation) => (
                    <PropertyRow
                      key={relation.edgeId || relation.id}
                      name={relation.displayName || t('graphPanel.propertiesView.node.related', '相关')}
                      value={relation.displayValue || relation.label}
                      tooltip={relation.triple || relation.label}
                      onClick={() => {
                        useGraphStore.getState().setSelectedNode(relation.id, true)
                      }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

const EdgePropertiesView = ({ edge, pipelineBusy }: { edge: EdgeType; pipelineBusy: boolean }) => {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-between items-center">
        <h3 className="text-md pl-1 font-bold tracking-wide text-violet-700">{t('graphPanel.propertiesView.edge.title')}</h3>
        {pipelineBusy && (
          <Button
            type="button"
            size="icon"
            variant="ghost"
            aria-label={t('graphPanel.propertiesView.editLockedByPipeline')}
            aria-disabled="true"
            className="h-7 w-7 border border-amber-400 hover:bg-amber-50 dark:border-amber-600 dark:hover:bg-amber-900/40 !cursor-default"
            tooltip={t('graphPanel.propertiesView.editLockedByPipeline')}
            onClick={(e) => e.preventDefault()}
          >
            <Lock className="h-4 w-4 text-amber-600 dark:text-amber-400" />
          </Button>
        )}
      </div>
      <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
        <PropertyRow name={t('graphPanel.propertiesView.edge.id')} value={edge.id} />
        {edge.type && <PropertyRow name={t('graphPanel.propertiesView.edge.type')} value={edge.type} />}
        <PropertyRow
          name={t('graphPanel.propertiesView.edge.source')}
          value={edge.sourceNode ? edge.sourceNode.labels.join(', ') : edge.source}
          onClick={() => {
            useGraphStore.getState().setSelectedNode(edge.source, true)
          }}
        />
        <PropertyRow
          name={t('graphPanel.propertiesView.edge.target')}
          value={edge.targetNode ? edge.targetNode.labels.join(', ') : edge.target}
          onClick={() => {
            useGraphStore.getState().setSelectedNode(edge.target, true)
          }}
        />
      </div>
      <h3 className="text-md pl-1 font-bold tracking-wide text-amber-700">{t('graphPanel.propertiesView.edge.properties')}</h3>
      <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
        {Object.keys(edge.properties)
          .sort()
          .map((name) => {
            if (name === 'created_at' || name === 'truncate') return null; // Hide created_at and truncate properties
            return (
              <PropertyRow
                key={name}
                name={name}
                value={edge.properties[name]}
                edgeId={String(edge.id)}
                dynamicId={String(edge.dynamicId)}
                entityType="edge"
                sourceId={edge.sourceNode?.properties['entity_id'] || edge.source}
                targetId={edge.targetNode?.properties['entity_id'] || edge.target}
                isEditable={name === 'description' || name === 'keywords'}
                truncate={edge.properties['truncate']}
                pipelineBusy={pipelineBusy}
              />
            )
          })}
      </div>
    </div>
  )
}

export default PropertiesView
