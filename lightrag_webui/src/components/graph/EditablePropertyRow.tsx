import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { updateEntity, updateRelation, checkEntityNameExists } from '@/api/lightrag'
import { updateGraphNode, updateGraphEdge } from '@/utils/graphOperations'
import { PropertyName, EditIcon, PropertyValue } from './PropertyRowComponents'
import PropertyEditDialog from './PropertyEditDialog'

/**
 * Interface for the EditablePropertyRow component props
 */
interface EditablePropertyRowProps {
  name: string                  // Property name to display and edit
  value: any                    // Initial value of the property
  onClick?: () => void          // Optional click handler for the property value
  entityId?: string             // ID of the entity (for node type)
  entityType?: 'node' | 'edge'  // Type of graph entity
  sourceId?: string            // Source node ID (for edge type)
  targetId?: string            // Target node ID (for edge type)
  onValueChange?: (newValue: any) => void  // Optional callback when value changes
  isEditable?: boolean         // Whether this property can be edited
}

/**
 * EditablePropertyRow component that supports editing property values
 * This component is used in the graph properties panel to display and edit entity properties
 */
const EditablePropertyRow = ({
  name,
  value: initialValue,
  onClick,
  entityId,
  entityType,
  sourceId,
  targetId,
  onValueChange,
  isEditable = false
}: EditablePropertyRowProps) => {
  const { t } = useTranslation()
  const [isEditing, setIsEditing] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [currentValue, setCurrentValue] = useState(initialValue)

  useEffect(() => {
    setCurrentValue(initialValue)
  }, [initialValue])

  const handleEditClick = () => {
    if (isEditable && !isEditing) {
      setIsEditing(true)
    }
  }

  const handleCancel = () => {
    setIsEditing(false)
  }

  const handleSave = async (value: string) => {
    if (isSubmitting || value === String(currentValue)) {
      setIsEditing(false)
      return
    }

    setIsSubmitting(true)

    try {
      if (entityType === 'node' && entityId) {
        let updatedData = { [name]: value }

        if (name === 'entity_id') {
          const exists = await checkEntityNameExists(value)
          if (exists) {
            toast.error(t('graphPanel.propertiesView.errors.duplicateName'))
            return
          }
          updatedData = { 'entity_name': value }
        }

        await updateEntity(entityId, updatedData, true)
        await updateGraphNode(entityId, name, value)
        toast.success(t('graphPanel.propertiesView.success.entityUpdated'))
      } else if (entityType === 'edge' && sourceId && targetId) {
        const updatedData = { [name]: value }
        await updateRelation(sourceId, targetId, updatedData)
        await updateGraphEdge(sourceId, targetId, name, value)
        toast.success(t('graphPanel.propertiesView.success.relationUpdated'))
      }

      setIsEditing(false)
      setCurrentValue(value)
      onValueChange?.(value)
    } catch (error) {
      console.error('Error updating property:', error)
      toast.error(t('graphPanel.propertiesView.errors.updateFailed'))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="flex items-center gap-1">
      <PropertyName name={name} />
      <EditIcon onClick={handleEditClick} />:
      <PropertyValue value={currentValue} onClick={onClick} />
      <PropertyEditDialog
        isOpen={isEditing}
        onClose={handleCancel}
        onSave={handleSave}
        propertyName={name}
        initialValue={String(currentValue)}
        isSubmitting={isSubmitting}
      />
    </div>
  )
}

export default EditablePropertyRow
