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
export const updateGraphNode = async (nodeId: string, propertyName: string, newValue: string) => {
  // Get graph state from store
  const sigmaGraph = useGraphStore.getState().sigmaGraph
  const rawGraph = useGraphStore.getState().rawGraph

  // Validate graph state
  if (!sigmaGraph || !rawGraph || !sigmaGraph.hasNode(String(nodeId))) {
    return
  }

  try {
    const nodeAttributes = sigmaGraph.getNodeAttributes(String(nodeId))

    // Special handling for entity_id changes (node renaming)
    if (propertyName === 'entity_id') {
      // Create new node with updated ID but same attributes
      sigmaGraph.addNode(newValue, { ...nodeAttributes, label: newValue })

      const edgesToUpdate: EdgeToUpdate[] = []

      // Process all edges connected to this node
      sigmaGraph.forEachEdge(String(nodeId), (edge, attributes, source, target) => {
        const otherNode = source === String(nodeId) ? target : source
        const isOutgoing = source === String(nodeId)

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
      sigmaGraph.dropNode(String(nodeId))

      // Update node reference in raw graph data
      const nodeIndex = rawGraph.nodeIdMap[String(nodeId)]
      if (nodeIndex !== undefined) {
        rawGraph.nodes[nodeIndex].id = newValue
        rawGraph.nodes[nodeIndex].properties.entity_id = newValue
        delete rawGraph.nodeIdMap[String(nodeId)]
        rawGraph.nodeIdMap[newValue] = nodeIndex
      }

      // Update all edge references in raw graph data
      edgesToUpdate.forEach(({ originalDynamicId, newEdgeId, edgeIndex }) => {
        if (rawGraph.edges[edgeIndex]) {
          // Update source/target references
          if (rawGraph.edges[edgeIndex].source === String(nodeId)) {
            rawGraph.edges[edgeIndex].source = newValue
          }
          if (rawGraph.edges[edgeIndex].target === String(nodeId)) {
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
      // For other properties, just update the property in raw graph
      const nodeIndex = rawGraph.nodeIdMap[String(nodeId)]
      if (nodeIndex !== undefined) {
        rawGraph.nodes[nodeIndex].properties[propertyName] = newValue
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
export const updateGraphEdge = async (sourceId: string, targetId: string, propertyName: string, newValue: string) => {
  // Get graph state from store
  const sigmaGraph = useGraphStore.getState().sigmaGraph
  const rawGraph = useGraphStore.getState().rawGraph

  // Validate graph state
  if (!sigmaGraph || !rawGraph) {
    return
  }

  try {
    // Find the edge between source and target nodes
    const allEdges = sigmaGraph.edges()
    let keyToUse = null

    for (const edge of allEdges) {
      const edgeSource = sigmaGraph.source(edge)
      const edgeTarget = sigmaGraph.target(edge)

      // Match edge in either direction (undirected graph support)
      if ((edgeSource === sourceId && edgeTarget === targetId) ||
          (edgeSource === targetId && edgeTarget === sourceId)) {
        keyToUse = edge
        break
      }
    }

    if (keyToUse !== null) {
      // Special handling for keywords property (updates edge label)
      if(propertyName === 'keywords') {
        sigmaGraph.setEdgeAttribute(keyToUse, 'label', newValue)
      } else {
        sigmaGraph.setEdgeAttribute(keyToUse, propertyName, newValue)
      }

      // Update edge in raw graph data using dynamic ID mapping
      if (keyToUse && rawGraph.edgeDynamicIdMap[keyToUse] !== undefined) {
        const edgeIndex = rawGraph.edgeDynamicIdMap[keyToUse]
        if (rawGraph.edges[edgeIndex]) {
          rawGraph.edges[edgeIndex].properties[propertyName] = newValue
        }
      } else if (keyToUse !== null) {
        // Fallback: try to find edge by key in edge ID map
        const edgeIndexByKey = rawGraph.edgeIdMap[keyToUse]
        if (edgeIndexByKey !== undefined && rawGraph.edges[edgeIndexByKey]) {
          rawGraph.edges[edgeIndexByKey].properties[propertyName] = newValue
        }
      }
    }
  } catch (error) {
    console.error(`Error updating edge ${sourceId}->${targetId} in graph:`, error)
    throw new Error('Failed to update edge in graph')
  }
}
