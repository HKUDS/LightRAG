import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { updateEntity, updateRelation, checkEntityNameExists } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import { SearchHistoryManager } from '@/utils/SearchHistoryManager'
import { PropertyName, EditIcon, PropertyValue } from './PropertyRowComponents'
import PropertyEditDialog from './PropertyEditDialog'
import MergeDialog from './MergeDialog'

/**
 * Interface for the EditablePropertyRow component props
 */
interface EditablePropertyRowProps {
  name: string                  // Property name to display and edit
  value: any                    // Initial value of the property
  onClick?: () => void          // Optional click handler for the property value
  nodeId?: string               // ID of the node (for node type)
  entityId?: string             // ID of the entity (for node type)
  edgeId?: string               // ID of the edge (for edge type)
  dynamicId?: string
  entityType?: 'node' | 'edge'  // Type of graph entity
  sourceId?: string            // Source node ID (for edge type)
  targetId?: string            // Target node ID (for edge type)
  onValueChange?: (newValue: any) => void  // Optional callback when value changes
  isEditable?: boolean         // Whether this property can be edited
  tooltip?: string             // Optional tooltip to display on hover
}

/**
 * EditablePropertyRow component that supports editing property values
 * This component is used in the graph properties panel to display and edit entity properties
 */
const EditablePropertyRow = ({
  name,
  value: initialValue,
  onClick,
  nodeId,
  edgeId,
  entityId,
  dynamicId,
  entityType,
  sourceId,
  targetId,
  onValueChange,
  isEditable = false,
  tooltip
}: EditablePropertyRowProps) => {
  const { t } = useTranslation()
  const [isEditing, setIsEditing] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [currentValue, setCurrentValue] = useState(initialValue)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [mergeDialogOpen, setMergeDialogOpen] = useState(false)
  const [mergeDialogInfo, setMergeDialogInfo] = useState<{
    targetEntity: string
    sourceEntity: string
  } | null>(null)

  useEffect(() => {
    setCurrentValue(initialValue)
  }, [initialValue])

  const handleEditClick = () => {
    if (isEditable && !isEditing) {
      setIsEditing(true)
      setErrorMessage(null)
    }
  }

  const handleCancel = () => {
    setIsEditing(false)
    setErrorMessage(null)
  }

  const handleSave = async (value: string, options?: { allowMerge?: boolean }) => {
    if (isSubmitting || value === String(currentValue)) {
      setIsEditing(false)
      setErrorMessage(null)
      return
    }

    setIsSubmitting(true)
    setErrorMessage(null)

    try {
      if (entityType === 'node' && entityId && nodeId) {
        let updatedData = { [name]: value }
        const allowMerge = options?.allowMerge ?? false

        if (name === 'entity_id') {
          if (!allowMerge) {
            const exists = await checkEntityNameExists(value)
            if (exists) {
              const errorMsg = t('graphPanel.propertiesView.errors.duplicateName')
              setErrorMessage(errorMsg)
              toast.error(errorMsg)
              return
            }
          }
          updatedData = { 'entity_name': value }
        }

        const response = await updateEntity(entityId, updatedData, true, allowMerge)
        const operationSummary = response.operation_summary
        const operationStatus = operationSummary?.operation_status || 'complete_success'
        const finalValue = operationSummary?.final_entity ?? value

        // Handle different operation statuses
        if (operationStatus === 'success') {
          if (operationSummary?.merged) {
            // Node was successfully merged into an existing entity
            setMergeDialogInfo({
              targetEntity: finalValue,
              sourceEntity: entityId,
            })
            setMergeDialogOpen(true)

            // Remove old entity name from search history
            SearchHistoryManager.removeLabel(entityId)

            // Note: Search Label update is deferred until user clicks refresh button in merge dialog

            toast.success(t('graphPanel.propertiesView.success.entityMerged'))
          } else {
            // Node was updated/renamed normally
            try {
              await useGraphStore
                .getState()
                .updateNodeAndSelect(nodeId, entityId, name, finalValue)
            } catch (error) {
              console.error('Error updating node in graph:', error)
              throw new Error('Failed to update node in graph')
            }

            // Update search history: remove old name, add new name
            if (name === 'entity_id') {
              const currentLabel = useSettingsStore.getState().queryLabel

              SearchHistoryManager.removeLabel(entityId)
              SearchHistoryManager.addToHistory(finalValue)

              // Trigger dropdown refresh to show updated search history
              useSettingsStore.getState().triggerSearchLabelDropdownRefresh()

              // If current queryLabel is the old entity name, update to new name
              if (currentLabel === entityId) {
                useSettingsStore.getState().setQueryLabel(finalValue)
              }
            }

            toast.success(t('graphPanel.propertiesView.success.entityUpdated'))
          }

          // Update local state and notify parent component
          // For entity_id updates, use finalValue (which may be different due to merging)
          // For other properties, use the original value the user entered
          const valueToSet = name === 'entity_id' ? finalValue : value
          setCurrentValue(valueToSet)
          onValueChange?.(valueToSet)

        } else if (operationStatus === 'partial_success') {
          // Partial success: update succeeded but merge failed
          // Do NOT update graph data to keep frontend in sync with backend
          const mergeError = operationSummary?.merge_error || 'Unknown error'

          const errorMsg = t('graphPanel.propertiesView.errors.updateSuccessButMergeFailed', {
            error: mergeError
          })
          setErrorMessage(errorMsg)
          toast.error(errorMsg)
          // Do not update currentValue or call onValueChange
          return

        } else {
          // Complete failure or unknown status
          // Check if this was a merge attempt or just a regular update
          if (operationSummary?.merge_status === 'failed') {
            // Merge operation was attempted but failed
            const mergeError = operationSummary?.merge_error || 'Unknown error'
            const errorMsg = t('graphPanel.propertiesView.errors.mergeFailed', {
              error: mergeError
            })
            setErrorMessage(errorMsg)
            toast.error(errorMsg)
          } else {
            // Regular update failed (no merge involved)
            const errorMsg = t('graphPanel.propertiesView.errors.updateFailed')
            setErrorMessage(errorMsg)
            toast.error(errorMsg)
          }
          // Do not update currentValue or call onValueChange
          return
        }
      } else if (entityType === 'edge' && sourceId && targetId && edgeId && dynamicId) {
        const updatedData = { [name]: value }
        await updateRelation(sourceId, targetId, updatedData)
        try {
          await useGraphStore.getState().updateEdgeAndSelect(edgeId, dynamicId, sourceId, targetId, name, value)
        } catch (error) {
          console.error(`Error updating edge ${sourceId}->${targetId} in graph:`, error)
          throw new Error('Failed to update edge in graph')
        }
        toast.success(t('graphPanel.propertiesView.success.relationUpdated'))
        setCurrentValue(value)
        onValueChange?.(value)
      }

      setIsEditing(false)
    } catch (error) {
      console.error('Error updating property:', error)
      const errorMsg = error instanceof Error ? error.message : t('graphPanel.propertiesView.errors.updateFailed')
      setErrorMessage(errorMsg)
      toast.error(errorMsg)
      return
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleMergeRefresh = (useMergedStart: boolean) => {
    const info = mergeDialogInfo
    const graphState = useGraphStore.getState()
    const settingsState = useSettingsStore.getState()
    const currentLabel = settingsState.queryLabel

    // Clear graph state
    graphState.clearSelection()
    graphState.setGraphDataFetchAttempted(false)
    graphState.setLastSuccessfulQueryLabel('')

    if (useMergedStart && info?.targetEntity) {
      // Use merged entity as new start point (might already be set in handleSave)
      settingsState.setQueryLabel(info.targetEntity)
    } else {
      // Keep current start point - refresh by resetting and restoring label
      // This handles the case where user wants to stay with current label
      settingsState.setQueryLabel('')
      setTimeout(() => {
        settingsState.setQueryLabel(currentLabel)
      }, 50)
    }

    // Force graph re-render and reset zoom/scale (same as refresh button behavior)
    graphState.incrementGraphDataVersion()

    setMergeDialogOpen(false)
    setMergeDialogInfo(null)
    toast.info(t('graphPanel.propertiesView.mergeDialog.refreshing'))
  }

  return (
    <div className="flex items-center gap-1 overflow-hidden">
      <PropertyName name={name} />
      <EditIcon onClick={handleEditClick} />:
      <PropertyValue
        value={currentValue}
        onClick={onClick}
        tooltip={tooltip || (typeof currentValue === 'string' ? currentValue : JSON.stringify(currentValue, null, 2))}
      />
      <PropertyEditDialog
        isOpen={isEditing}
        onClose={handleCancel}
        onSave={handleSave}
        propertyName={name}
        initialValue={String(currentValue)}
        isSubmitting={isSubmitting}
        errorMessage={errorMessage}
      />

      <MergeDialog
        mergeDialogOpen={mergeDialogOpen}
        mergeDialogInfo={mergeDialogInfo}
        onOpenChange={(open) => {
          setMergeDialogOpen(open)
          if (!open) {
            setMergeDialogInfo(null)
          }
        }}
        onRefresh={handleMergeRefresh}
      />
    </div>
  )
}

export default EditablePropertyRow
