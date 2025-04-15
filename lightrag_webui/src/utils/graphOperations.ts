import { useGraphStore } from '@/stores/graph'

/**
 * Interface for tracking edges that need updating when a node ID changes
 */
interface EdgeToUpdate {
  originalDynamicId: string
  newEdgeId: string
  edgeIndex: number
}

/**
 * Update node in the graph visualization
 * Handles both property updates and entity ID changes
 *
 * @param nodeId - ID of the node to update
 * @param propertyName - Name of the property being updated
 * @param newValue - New value for the property
 */
export const updateGraphNode = async (nodeId: string, entityId:string, propertyName: string, newValue: string) => {
  // Get graph state from store
  const sigmaGraph = useGraphStore.getState().sigmaGraph
  const rawGraph = useGraphStore.getState().rawGraph

  // Validate graph state
  if (!sigmaGraph || !rawGraph || !sigmaGraph.hasNode(nodeId)) {
    return
  }

  try {
    const nodeAttributes = sigmaGraph.getNodeAttributes(nodeId)

    console.log('updateGraphNode', nodeId, entityId, propertyName, newValue)

    // For entity_id changes (node renaming) with NetworkX graph storage
    if ((nodeId === entityId) && (propertyName === 'entity_id')) {
      // Create new node with updated ID but same attributes
      sigmaGraph.addNode(newValue, { ...nodeAttributes, label: newValue })

      const edgesToUpdate: EdgeToUpdate[] = []

      // Process all edges connected to this node
      sigmaGraph.forEachEdge(nodeId, (edge, attributes, source, target) => {
        const otherNode = source === nodeId ? target : source
        const isOutgoing = source === nodeId

        // Get original edge dynamic ID for later reference
        const originalEdgeDynamicId = edge
        const edgeIndexInRawGraph = rawGraph.edgeDynamicIdMap[originalEdgeDynamicId]

        // Create new edge with updated node reference
        const newEdgeId = sigmaGraph.addEdge(
          isOutgoing ? newValue : otherNode,
          isOutgoing ? otherNode : newValue,
          attributes
        )

        // Track edges that need updating in the raw graph
        if (edgeIndexInRawGraph !== undefined) {
          edgesToUpdate.push({
            originalDynamicId: originalEdgeDynamicId,
            newEdgeId: newEdgeId,
            edgeIndex: edgeIndexInRawGraph
          })
        }

        // Remove the old edge
        sigmaGraph.dropEdge(edge)
      })

      // Remove the old node after all edges are processed
      sigmaGraph.dropNode(nodeId)

      // Update node reference in raw graph data
      const nodeIndex = rawGraph.nodeIdMap[nodeId]
      if (nodeIndex !== undefined) {
        rawGraph.nodes[nodeIndex].id = newValue
        rawGraph.nodes[nodeIndex].labels = [newValue]
        rawGraph.nodes[nodeIndex].properties.entity_id = newValue
        delete rawGraph.nodeIdMap[nodeId]
        rawGraph.nodeIdMap[newValue] = nodeIndex
      }

      // Update all edge references in raw graph data
      edgesToUpdate.forEach(({ originalDynamicId, newEdgeId, edgeIndex }) => {
        if (rawGraph.edges[edgeIndex]) {
          // Update source/target references
          if (rawGraph.edges[edgeIndex].source === nodeId) {
            rawGraph.edges[edgeIndex].source = newValue
          }
          if (rawGraph.edges[edgeIndex].target === nodeId) {
            rawGraph.edges[edgeIndex].target = newValue
          }

          // Update dynamic ID mappings
          rawGraph.edges[edgeIndex].dynamicId = newEdgeId
          delete rawGraph.edgeDynamicIdMap[originalDynamicId]
          rawGraph.edgeDynamicIdMap[newEdgeId] = edgeIndex
        }
      })

      // Update selected node in store
      useGraphStore.getState().setSelectedNode(newValue)
    } else {
      // for none NetworkX nodes or none entity_id changes
      const nodeIndex = rawGraph.nodeIdMap[String(nodeId)]
      if (nodeIndex !== undefined) {
        rawGraph.nodes[nodeIndex].properties[propertyName] = newValue
        if (propertyName === 'entity_id') {
          rawGraph.nodes[nodeIndex].labels = [newValue]
          sigmaGraph.setNodeAttribute(String(nodeId), 'label', newValue)
        }
      }
    }
  } catch (error) {
    console.error('Error updating node in graph:', error)
    throw new Error('Failed to update node in graph')
  }
}

/**
 * Update edge in the graph visualization
 *
 * @param sourceId - ID of the source node
 * @param targetId - ID of the target node
 * @param propertyName - Name of the property being updated
 * @param newValue - New value for the property
 */
export const updateGraphEdge = async (edgeId: string, dynamicId: string, sourceId: string, targetId: string, propertyName: string, newValue: string) => {
  // Get graph state from store
  const sigmaGraph = useGraphStore.getState().sigmaGraph
  const rawGraph = useGraphStore.getState().rawGraph

  // Validate graph state
  if (!sigmaGraph || !rawGraph) {
    return
  }

  try {

    const edgeIndex = rawGraph.edgeIdMap[String(edgeId)]
    if (edgeIndex !== undefined && rawGraph.edges[edgeIndex]) {
      rawGraph.edges[edgeIndex].properties[propertyName] = newValue
      if(dynamicId !== undefined && propertyName === 'keywords') {
        sigmaGraph.setEdgeAttribute(dynamicId, 'label', newValue)
      }
    }

  } catch (error) {
    console.error(`Error updating edge ${sourceId}->${targetId} in graph:`, error)
    throw new Error('Failed to update edge in graph')
  }
}
