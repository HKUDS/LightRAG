import { useState, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import Text from '@/components/ui/Text'
import Input from '@/components/ui/Input'
import { toast } from 'sonner'
import { updateEntity, updateRelation, checkEntityNameExists } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'

interface EditablePropertyRowProps {
  name: string
  value: any
  onClick?: () => void
  tooltip?: string
  entityId?: string
  entityType?: 'node' | 'edge'
  sourceId?: string
  targetId?: string
  onValueChange?: (newValue: any) => void
  isEditable?: boolean
}

/**
 * EditablePropertyRow component that supports double-click to edit property values
 * Specifically designed for editing 'description' and entity name fields
 */
const EditablePropertyRow = ({
  name,
  value,
  onClick,
  tooltip,
  entityId,
  entityType,
  sourceId,
  targetId,
  onValueChange,
  isEditable = false
}: EditablePropertyRowProps) => {
  const { t } = useTranslation()
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Initialize edit value when entering edit mode
  useEffect(() => {
    if (isEditing) {
      setEditValue(String(value))
      // Focus the input element when entering edit mode
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus()
          inputRef.current.select()
        }
      }, 50)
    }
  }, [isEditing, value])

  const getPropertyNameTranslation = (propName: string) => {
    const translationKey = `graphPanel.propertiesView.node.propertyNames.${propName}`
    const translation = t(translationKey)
    return translation === translationKey ? propName : translation
  }

  const handleDoubleClick = () => {
    if (isEditable && !isEditing) {
      setIsEditing(true)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave()
    } else if (e.key === 'Escape') {
      setIsEditing(false)
    }
  }

  const handleSave = async () => {
    if (isSubmitting) return

    // Don't save if value hasn't changed
    if (editValue === String(value)) {
      setIsEditing(false)
      return
    }

    setIsSubmitting(true)

    try {
      // Special handling for entity_id (name) field to check for duplicates
      if (name === 'entity_id' && entityType === 'node') {
        // Ensure we are not checking the original name against itself if it's protected
        if (editValue !== String(value)) {
            const exists = await checkEntityNameExists(editValue);
            if (exists) {
                toast.error(t('graphPanel.propertiesView.errors.duplicateName'));
                setIsSubmitting(false);
                return;
            }
        }
      }

      // Update the entity or relation in the database
      if (entityType === 'node' && entityId) {
        // For nodes, we need to determine if we're updating the name or description
        const updatedData: Record<string, any> = {}

        if (name === 'entity_id') {
          // For entity name updates
          updatedData['entity_name'] = editValue
          await updateEntity(String(value), updatedData, true) // Pass original name (value) as identifier

          // Update node label in the graph directly instead of reloading the entire graph
          const sigmaInstance = useGraphStore.getState().sigmaInstance
          const sigmaGraph = useGraphStore.getState().sigmaGraph
          const rawGraph = useGraphStore.getState().rawGraph

          if (sigmaInstance && sigmaGraph && rawGraph) {
            // Update the node in sigma graph
            if (sigmaGraph.hasNode(String(value))) {
              // Update the node label in the sigma graph
              sigmaGraph.setNodeAttribute(String(value), 'label', editValue)

              // Also update the node in the raw graph
              const nodeIndex = rawGraph.nodeIdMap[String(value)]
              if (nodeIndex !== undefined) {
                rawGraph.nodes[nodeIndex].id = editValue
                // Update the node ID map
                delete rawGraph.nodeIdMap[String(value)]
                rawGraph.nodeIdMap[editValue] = nodeIndex
              }

              // Refresh the sigma instance to reflect changes
              sigmaInstance.refresh()

              // Update selected node ID if it was the edited node
              const selectedNode = useGraphStore.getState().selectedNode
              if (selectedNode === String(value)) {
                useGraphStore.getState().setSelectedNode(editValue)
              }
            }
          } else {
            // Fallback to full graph reload if direct update is not possible
            useGraphStore.getState().setGraphDataFetchAttempted(false)
            useGraphStore.getState().setLabelsFetchAttempted(false)

            // Get current label to trigger reload
            const currentLabel = useSettingsStore.getState().queryLabel
            if (currentLabel) {
              // Trigger data reload by temporarily clearing and resetting the label
              useSettingsStore.getState().setQueryLabel('')
              setTimeout(() => {
                useSettingsStore.getState().setQueryLabel(currentLabel)
              }, 0)
            }
          }
        } else if (name === 'description') {
          // For description updates
          updatedData['description'] = editValue
          await updateEntity(entityId, updatedData) // Pass entityId as identifier
        } else {
          // For other property updates
          updatedData[name] = editValue
          await updateEntity(entityId, updatedData) // Pass entityId as identifier
        }

        toast.success(t('graphPanel.propertiesView.success.entityUpdated'))
      } else if (entityType === 'edge' && sourceId && targetId) {
        // For edges, update the relation
        const updatedData: Record<string, any> = {}
        updatedData[name] = editValue
        await updateRelation(sourceId, targetId, updatedData)
        toast.success(t('graphPanel.propertiesView.success.relationUpdated'))
      }

      // Notify parent component about the value change
      if (onValueChange) {
        onValueChange(editValue)
      }
    } catch (error: any) { // Keep type as any to access potential response properties
      console.error('Error updating property:', error);

      // Attempt to extract a more specific error message
      let detailMessage = t('graphPanel.propertiesView.errors.updateFailed'); // Default message
      if (error.response?.data?.detail) {
          // Use the detailed message from the backend response if available
          detailMessage = error.response.data.detail;
      } else if (error.message) {
          // Use the error object's message if no backend detail
          detailMessage = error.message;
      }

      toast.error(detailMessage); // Show the determined error message

    } finally {
      setIsSubmitting(false)
      setIsEditing(false)
    }
  }

  // Determine if this property should be editable
  // Currently only 'description' and 'entity_id' fields are editable
  const isEditableField = isEditable && (name === 'description' || name === 'entity_id')

  return (
    <div className="flex items-center gap-2">
      <span className="text-primary/60 tracking-wide whitespace-nowrap">
        {getPropertyNameTranslation(name)}
      </span>:
      {isEditing ? (
        <div className="flex-1">
          <Input
            ref={inputRef}
            className="h-7 text-xs w-full"
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={handleSave}
            onKeyDown={handleKeyDown}
            disabled={isSubmitting}
          />
        </div>
      ) : (
        // Wrap Text component in a div to handle onDoubleClick
        <div
          className={`flex-1 overflow-hidden ${isEditableField ? 'cursor-text' : ''}`} // Apply cursor style to wrapper
          onDoubleClick={isEditableField ? handleDoubleClick : undefined}
        >
          <Text
            className="hover:bg-primary/20 rounded p-1 overflow-hidden text-ellipsis block w-full" // Ensure Text fills the div
            tooltipClassName="max-w-80"
            // Ensure the text prop always receives a string representation
            text={String(value)}
            tooltip={tooltip || (typeof value === 'string' ? value : JSON.stringify(value, null, 2))}
            side="left"
            onClick={onClick}
            // Removed onDoubleClick from Text component
          />
        </div>
      )}
    </div>
  )
}

export default EditablePropertyRow
