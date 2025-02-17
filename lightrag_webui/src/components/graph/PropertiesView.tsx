import { useEffect, useState } from 'react'
import { useGraphStore, RawNodeType, RawEdgeType } from '@/stores/graph'
import Text from '@/components/ui/Text'
import useLightragGraph from '@/hooks/useLightragGraph'

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
    for (const edgeId of state.sigmaGraph.edges(node.id)) {
      const edge = state.rawGraph.getEdge(edgeId, true)
      if (edge) {
        const isTarget = node.id === edge.source
        const neighbourId = isTarget ? edge.target : edge.source
        const neighbour = state.rawGraph.getNode(neighbourId)
        if (neighbour) {
          relationships.push({
            type: isTarget ? 'Target' : 'Source',
            id: neighbourId,
            label: neighbour.labels.join(', ')
          })
        }
      }
    }
  }
  return {
    ...node,
    relationships
  }
}

const refineEdgeProperties = (edge: RawEdgeType): EdgeType => {
  const state = useGraphStore.getState()
  const sourceNode = state.rawGraph?.getNode(edge.source)
  const targetNode = state.rawGraph?.getNode(edge.target)
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
  return (
    <div className="flex items-center gap-2">
      <label className="text-primary/60 tracking-wide">{name}</label>:
      <Text
        className="hover:bg-primary/20 rounded p-1 text-ellipsis"
        tooltipClassName="max-w-80"
        text={value}
        tooltip={tooltip || value}
        side="left"
        onClick={onClick}
      />
    </div>
  )
}

const NodePropertiesView = ({ node }: { node: NodeType }) => {
  return (
    <div className="flex flex-col gap-2">
      <label className="text-md pl-1 font-bold tracking-wide text-sky-300">Node</label>
      <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
        <PropertyRow name={'Id'} value={node.id} />
        <PropertyRow
          name={'Labels'}
          value={node.labels.join(', ')}
          onClick={() => {
            useGraphStore.getState().setSelectedNode(node.id, true)
          }}
        />
        <PropertyRow name={'Degree'} value={node.degree} />
      </div>
      <label className="text-md pl-1 font-bold tracking-wide text-yellow-400/90">Properties</label>
      <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
        {Object.keys(node.properties)
          .sort()
          .map((name) => {
            return <PropertyRow key={name} name={name} value={node.properties[name]} />
          })}
      </div>
      {node.relationships.length > 0 && (
        <>
          <label className="text-md pl-1 font-bold tracking-wide text-teal-600/90">
            Relationships
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
  return (
    <div className="flex flex-col gap-2">
      <label className="text-md pl-1 font-bold tracking-wide text-teal-600">Relationship</label>
      <div className="bg-primary/5 max-h-96 overflow-auto rounded p-1">
        <PropertyRow name={'Id'} value={edge.id} />
        {edge.type && <PropertyRow name={'Type'} value={edge.type} />}
        <PropertyRow
          name={'Source'}
          value={edge.sourceNode ? edge.sourceNode.labels.join(', ') : edge.source}
          onClick={() => {
            useGraphStore.getState().setSelectedNode(edge.source, true)
          }}
        />
        <PropertyRow
          name={'Target'}
          value={edge.targetNode ? edge.targetNode.labels.join(', ') : edge.target}
          onClick={() => {
            useGraphStore.getState().setSelectedNode(edge.target, true)
          }}
        />
      </div>
      <label className="text-md pl-1 font-bold tracking-wide text-yellow-400/90">Properties</label>
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
