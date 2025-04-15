import { useGraphStore } from '@/stores/graph'

/**
 * Update node in the graph visualization
 * This function is now a wrapper around the store's updateNodeAndSelect method
 * 
 * @param nodeId - ID of the node to update
 * @param entityId - ID of the entity
 * @param propertyName - Name of the property being updated
 * @param newValue - New value for the property
 */
export const updateGraphNode = async (nodeId: string, entityId: string, propertyName: string, newValue: string) => {
  try {
    // Call the store method that handles both data update and UI state
    await useGraphStore.getState().updateNodeAndSelect(nodeId, entityId, propertyName, newValue)
  } catch (error) {
    console.error('Error updating node in graph:', error)
    throw new Error('Failed to update node in graph')
  }
}

/**
 * Update edge in the graph visualization
 * This function is now a wrapper around the store's updateEdgeAndSelect method
 *
 * @param edgeId - ID of the edge
 * @param dynamicId - Dynamic ID of the edge in sigma graph
 * @param sourceId - ID of the source node
 * @param targetId - ID of the target node
 * @param propertyName - Name of the property being updated
 * @param newValue - New value for the property
 */
export const updateGraphEdge = async (edgeId: string, dynamicId: string, sourceId: string, targetId: string, propertyName: string, newValue: string) => {
  try {
    // Call the store method that handles both data update and UI state
    await useGraphStore.getState().updateEdgeAndSelect(edgeId, dynamicId, sourceId, targetId, propertyName, newValue)
  } catch (error) {
    console.error(`Error updating edge ${sourceId}->${targetId} in graph:`, error)
    throw new Error('Failed to update edge in graph')
  }
}
