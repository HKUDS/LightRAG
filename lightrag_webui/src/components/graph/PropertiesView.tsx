import { useEffect, useState } from 'react'
import { useGraphStore, RawNodeType, RawEdgeType } from '@/stores/graph'
import Text from '@/components/ui/Text'
import Button from '@/components/ui/Button'
import useLightragGraph from '@/hooks/useLightragGraph'
import { useTranslation } from 'react-i18next'
import { GitBranchPlus, Scissors } from 'lucide-react'

/**
 * Component that view properties of elements in graph.
 */
const PropertiesView = () => {
  const { getNode, getEdge } = useLightragGraph()
  const selectedNode = useGraphStore.use.selectedNode()
  const focusedNode = useGraphStore.use.focusedNode()
  const selectedEdge = useGraphStore.use.selectedEdge()
  const focusedEdge = useGraphStore.use.focusedEdge()

  const [currentElement, setCurrentElement] = useState<NodeType | EdgeType | null>(null)
  const [currentType, setCurrentType] = useState<'node' | 'edge' | null>(null)

  useEffect(() => {
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
      if (type == 'node') {
        setCurrentElement(refineNodeProperties(element as any))
      } else {
        setCurrentElement(refineEdgeProperties(element as any))
      }
      setCurrentType(type)
    } else {
      setCurrentElement(null)
      setCurrentType(null)
    }
  }, [
    focusedNode,
    selectedNode,
    focusedEdge,
    selectedEdge,
    setCurrentElement,
    setCurrentType,
    getNode,
    getEdge
  ])

  if (!currentElement) {
    return <></>
  }
  return (
    <div className="bg-background/80 max-w-xs rounded-lg border-2 p-2 text-xs backdrop-blur-lg">
      {currentType == 'node' ? (
        <NodePropertiesView node={currentElement as any} />
      ) : (
        <EdgePropertiesView edge={currentElement as any} />
      )}
    </div>
  )
}

type NodeType = RawNodeType & {
  relationships: {
    type: string
    id: string
    label: string
  }[]
}

type EdgeType = RawEdgeType & {
  sourceNode?: RawNodeType
  targetNode?: RawNodeType
}

const refineNodeProperties = (node: RawNodeType): NodeType => {
  const state = useGraphStore.getState()
  const relationships = []

  if (state.sigmaGraph && state.rawGraph) {
    try {
      if (!state.sigmaGraph.hasNode(node.id)) {
        return {
          ...node,
          relationships: []
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
              type: 'Neighbour',
              id: neighbourId,
              label: neighbour.properties['entity_id'] ? neighbour.properties['entity_id'] : neighbour.labels.join(', ')
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
    relationships
  }
}

const refineEdgeProperties = (edge: RawEdgeType): EdgeType => {
  const state = useGraphStore.getState()
  let sourceNode: RawNodeType | undefined = undefined
  let targetNode: RawNodeType | undefined = undefined

  if (state.sigmaGraph && state.rawGraph) {
    try {
      if (!state.sigmaGraph.hasEdge(edge.id)) {
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
  tooltip
}: {
  name: string
  value: any
  onClick?: () => void
  tooltip?: string
}) => {
  const { t } = useTranslation()

  const getPropertyNameTranslation = (name: string) => {
    const translationKey = `graphPanel.propertiesView.node.propertyNames.${name}`
    const translation = t(translationKey)
    return translation === translationKey ? name : translation
  }

  return (
    <div className="flex items-center gap-2">
      <label className="text-primary/60 tracking-wide whitespace-nowrap">{getPropertyNameTranslation(name)}</label>:
      <Text
        className="hover:bg-primary/20 rounded p-1 overflow-hidden text-ellipsis"
        tooltipClassName="max-w-80"
        text={value}
        tooltip={tooltip || (typeof value === 'string' ? value : JSON.stringify(value, null, 2))}
        side="left"
        onClick={onClick}
      />
    </div>
  )
}

const NodePropertiesView = ({ node }: { node: NodeType }) => {
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
        <label className="text-md pl-1 font-bold tracking-wide text-blue-700">{t('graphPanel.propertiesView.node.title')}</label>
        <div className="flex gap-3">
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 border border-gray-400 hover:bg-gray-200"
            onClick={handleExpandNode}
            tooltip={t('graphPanel.propertiesView.node.expandNode')}
          >
            <GitBranchPlus className="h-4 w-4 text-gray-700" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 border border-gray-400 hover:bg-gray-200"
            onClick={handlePruneNode}
            tooltip={t('graphPanel.propertiesView.node.pruneNode')}
          >
            <Scissors className="h-4 w-4 text-gray-900" />
          </Button>
        </div>
      </div>
      <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
        <PropertyRow name={t('graphPanel.propertiesView.node.id')} value={node.id} />
        <PropertyRow
          name={t('graphPanel.propertiesView.node.labels')}
          value={node.labels.join(', ')}
          onClick={() => {
            useGraphStore.getState().setSelectedNode(node.id, true)
          }}
        />
        <PropertyRow name={t('graphPanel.propertiesView.node.degree')} value={node.degree} />
      </div>
      <label className="text-md pl-1 font-bold tracking-wide text-amber-700">{t('graphPanel.propertiesView.node.properties')}</label>
      <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
        {Object.keys(node.properties)
          .sort()
          .map((name) => {
            return <PropertyRow key={name} name={name} value={node.properties[name]} />
          })}
      </div>
      {node.relationships.length > 0 && (
        <>
          <label className="text-md pl-1 font-bold tracking-wide text-emerald-700">
            {t('graphPanel.propertiesView.node.relationships')}
          </label>
          <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
            {node.relationships.map(({ type, id, label }) => {
              return (
                <PropertyRow
                  key={id}
                  name={type}
                  value={label}
                  onClick={() => {
                    useGraphStore.getState().setSelectedNode(id, true)
                  }}
                />
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}

const EdgePropertiesView = ({ edge }: { edge: EdgeType }) => {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col gap-2">
      <label className="text-md pl-1 font-bold tracking-wide text-violet-700">{t('graphPanel.propertiesView.edge.title')}</label>
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
      <label className="text-md pl-1 font-bold tracking-wide text-amber-700">{t('graphPanel.propertiesView.edge.properties')}</label>
      <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
        {Object.keys(edge.properties)
          .sort()
          .map((name) => {
            return <PropertyRow key={name} name={name} value={edge.properties[name]} />
          })}
      </div>
    </div>
  )
}

export default PropertiesView
