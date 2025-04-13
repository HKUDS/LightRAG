import { useState, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import Text from '@/components/ui/Text'
import Input from '@/components/ui/Input'
import { toast } from 'sonner'
import { updateEntity, updateRelation, checkEntityNameExists } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import { PencilIcon } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'

/**
 * Interface for the EditablePropertyRow component props
 * Defines all possible properties that can be passed to the component
 */
interface EditablePropertyRowProps {
  name: string                  // Property name to display and edit
  value: any                    // Initial value of the property
  onClick?: () => void          // Optional click handler for the property value
  tooltip?: string              // Optional tooltip text
  entityId?: string             // ID of the entity (for node type)
  entityType?: 'node' | 'edge'  // Type of graph entity
  sourceId?: string            // Source node ID (for edge type)
  targetId?: string            // Target node ID (for edge type)
  onValueChange?: (newValue: any) => void  // Optional callback when value changes
  isEditable?: boolean         // Whether this property can be edited
}

/**
 * Interface for tracking edges that need updating when a node ID changes
 */
interface EdgeToUpdate {
  originalDynamicId: string;
  newEdgeId: string;
  edgeIndex: number;
}

/**
 * EditablePropertyRow component that supports double-click to edit property values
 * This component is used in the graph properties panel to display and edit entity properties
 *
 * @component
 */
const EditablePropertyRow = ({
  name,
  value: initialValue,
  onClick,
  tooltip,
  entityId,
  entityType,
  sourceId,
  targetId,
  onValueChange,
  isEditable = false
}: EditablePropertyRowProps) => {
  // Component state
  const { t } = useTranslation()
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [currentValue, setCurrentValue] = useState(initialValue)
  const inputRef = useRef<HTMLInputElement>(null)

  /**
   * Update currentValue when initialValue changes from parent
   */
  useEffect(() => {
    setCurrentValue(initialValue)
  }, [initialValue])

  /**
   * Initialize edit value and focus input when entering edit mode
   */
  useEffect(() => {
    if (isEditing) {
      setEditValue(String(currentValue))
      // Focus the input element when entering edit mode with a small delay
      // to ensure the input is rendered before focusing
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus()
          inputRef.current.select()
        }
      }, 50)
    }
  }, [isEditing, currentValue])

  /**
   * Get translated property name from i18n
   * Falls back to the original name if no translation is found
   */
  const getPropertyNameTranslation = (propName: string) => {
    const translationKey = `graphPanel.propertiesView.node.propertyNames.${propName}`
    const translation = t(translationKey)
    return translation === translationKey ? propName : translation
  }

  /**
   * Handle double-click event to enter edit mode
   */
  const handleDoubleClick = () => {
    if (isEditable && !isEditing) {
      setIsEditing(true)
    }
  }

  /**
   * Handle keyboard events in the input field
   * - Enter: Save changes
   * - Escape: Cancel editing
   */
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave()
    } else if (e.key === 'Escape') {
      setIsEditing(false)
    }
  }

  /**
   * Update node in the graph visualization after API update
   * Handles both property updates and entity ID changes
   *
   * @param nodeId - ID of the node to update
   * @param propertyName - Name of the property being updated
   * @param newValue - New value for the property
   */
  const updateGraphNode = async (nodeId: string, propertyName: string, newValue: string) => {
    // Get graph state from store
    const sigmaInstance = useGraphStore.getState().sigmaInstance
    const sigmaGraph = useGraphStore.getState().sigmaGraph
    const rawGraph = useGraphStore.getState().rawGraph

    // Validate graph state
    if (!sigmaInstance || !sigmaGraph || !rawGraph || !sigmaGraph.hasNode(String(nodeId))) {
      return
    }

    try {
      const nodeAttributes = sigmaGraph.getNodeAttributes(String(nodeId))

      // Special handling for entity_id changes (node renaming)
      if (propertyName === 'entity_id') {
        // Create new node with updated ID but same attributes
        sigmaGraph.addNode(newValue, { ...nodeAttributes, label: newValue })

        const edgesToUpdate: EdgeToUpdate[] = [];

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
   * Update edge in the graph visualization after API update
   *
   * @param sourceId - ID of the source node
   * @param targetId - ID of the target node
   * @param propertyName - Name of the property being updated
   * @param newValue - New value for the property
   */
  const updateGraphEdge = async (sourceId: string, targetId: string, propertyName: string, newValue: string) => {
    // Get graph state from store
    const sigmaInstance = useGraphStore.getState().sigmaInstance
    const sigmaGraph = useGraphStore.getState().sigmaGraph
    const rawGraph = useGraphStore.getState().rawGraph

    // Validate graph state
    if (!sigmaInstance || !sigmaGraph || !rawGraph) {
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
          sigmaGraph.setEdgeAttribute(keyToUse, 'label', newValue);
        } else {
          sigmaGraph.setEdgeAttribute(keyToUse, propertyName, newValue);
        }

        // Update edge in raw graph data using dynamic ID mapping
        if (keyToUse && rawGraph.edgeDynamicIdMap[keyToUse] !== undefined) {
           const edgeIndex = rawGraph.edgeDynamicIdMap[keyToUse];
           if (rawGraph.edges[edgeIndex]) {
               rawGraph.edges[edgeIndex].properties[propertyName] = newValue;
           } else {
               console.warn(`Edge index ${edgeIndex} found but edge data missing in rawGraph for dynamicId ${entityId}`);
           }
        } else {
          // Fallback: try to find edge by key in edge ID map
          console.warn(`Could not find edge with dynamicId ${entityId} in rawGraph.edgeDynamicIdMap to update properties.`);
          if (keyToUse !== null) {
            const edgeIndexByKey = rawGraph.edgeIdMap[keyToUse];
            if (edgeIndexByKey !== undefined && rawGraph.edges[edgeIndexByKey]) {
                rawGraph.edges[edgeIndexByKey].properties[propertyName] = newValue;
                console.log(`Updated rawGraph edge using constructed key ${keyToUse}`);
            } else {
                console.warn(`Could not find edge in rawGraph using key ${keyToUse} either.`);
            }
          } else {
            console.warn('Cannot update edge properties: edge key is null');
          }
        }
      } else {
        console.warn(`Edge not found in sigmaGraph with key ${keyToUse}`);
      }
    } catch (error) {
      // Log the specific edge key that caused the error
      console.error(`Error updating edge ${sourceId}->${targetId} in graph:`, error);
      throw new Error('Failed to update edge in graph')
    }
  }

  /**
   * Save changes to the property value
   * Updates both the API and the graph visualization
   */
  const handleSave = async () => {
    // Prevent duplicate submissions
    if (isSubmitting) return

    // Skip if value hasn't changed
    if (editValue === String(currentValue)) {
      setIsEditing(false)
      return
    }

    setIsSubmitting(true)

    try {
      // Handle node property updates
      if (entityType === 'node' && entityId) {
        let updatedData = { [name]: editValue }

        // Special handling for entity_id (name) changes
        if (name === 'entity_id') {
          // Check if the new name already exists
          const exists = await checkEntityNameExists(editValue)
          if (exists) {
            toast.error(t('graphPanel.propertiesView.errors.duplicateName'))
            setIsSubmitting(false)
            return
          }
          // For entity_id, we update entity_name in the API
          updatedData = { 'entity_name': editValue }
        }

        // Update entity in API
        await updateEntity(entityId, updatedData, true)
        // Update graph visualization
        await updateGraphNode(entityId, name, editValue)
        toast.success(t('graphPanel.propertiesView.success.entityUpdated'))
      }
      // Handle edge property updates
      else if (entityType === 'edge' && sourceId && targetId) {
        const updatedData = { [name]: editValue }
        // Update relation in API
        await updateRelation(sourceId, targetId, updatedData)
        // Update graph visualization
        await updateGraphEdge(sourceId, targetId, name, editValue)
        toast.success(t('graphPanel.propertiesView.success.relationUpdated'))
      }

      // Update local state
      setIsEditing(false)
      setCurrentValue(editValue)

      // Notify parent component if callback provided
      if (onValueChange) {
        onValueChange(editValue)
      }
    } catch (error) {
      console.error('Error updating property:', error)
      toast.error(t('graphPanel.propertiesView.errors.updateFailed'))
    } finally {
      setIsSubmitting(false)
    }
  }

  /**
   * Render the property row with edit functionality
   * Shows property name, edit icon, and either the editable input or the current value
   */
  return (
    <div className="flex items-center gap-1" onDoubleClick={handleDoubleClick}>
      {/* Property name with translation */}
      <span className="text-primary/60 tracking-wide whitespace-nowrap">
        {getPropertyNameTranslation(name)}
      </span>

      {/* Edit icon with tooltip */}
      <TooltipProvider delayDuration={200}>
        <Tooltip>
          <TooltipTrigger asChild>
            <div>
              <PencilIcon
                className="h-3 w-3 text-gray-500 hover:text-gray-700 cursor-pointer"
                onClick={() => setIsEditing(true)}
              />
            </div>
          </TooltipTrigger>
          <TooltipContent side="top">
            {t('graphPanel.propertiesView.doubleClickToEdit')}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>:

      {/* Conditional rendering based on edit state */}
      {isEditing ? (
        // Input field for editing
        <Input
          ref={inputRef}
          type="text"
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={handleSave}
          className="h-6 text-xs"
          disabled={isSubmitting}
        />
      ) : (
        // Text display when not editing
        <div className="flex items-center gap-1">
          <Text
            className="hover:bg-primary/20 rounded p-1 overflow-hidden text-ellipsis"
            tooltipClassName="max-w-80"
            text={currentValue}
            tooltip={tooltip || (typeof currentValue === 'string' ? currentValue : JSON.stringify(currentValue, null, 2))}
            side="left"
            onClick={onClick}
          />
        </div>
      )}
    </div>
  )
}

export default EditablePropertyRow
