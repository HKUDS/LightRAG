import { useState, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import Text from '@/components/ui/Text'
import Input from '@/components/ui/Input'
import { toast } from 'sonner'
import { updateEntity, updateRelation, checkEntityNameExists } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import { PencilIcon } from 'lucide-react'

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

  const updateGraphNode = async (nodeId: string, propertyName: string, newValue: string) => {
    const sigmaInstance = useGraphStore.getState().sigmaInstance
    const sigmaGraph = useGraphStore.getState().sigmaGraph
    const rawGraph = useGraphStore.getState().rawGraph

    if (!sigmaInstance || !sigmaGraph || !rawGraph || !sigmaGraph.hasNode(String(nodeId))) {
      return
    }

    try {
      const nodeAttributes = sigmaGraph.getNodeAttributes(String(nodeId))

      if (propertyName === 'entity_id') {
        sigmaGraph.addNode(newValue, { ...nodeAttributes, label: newValue })

        sigmaGraph.forEachEdge(String(nodeId), (edge, attributes, source, target) => {
          const otherNode = source === String(nodeId) ? target : source
          const isOutgoing = source === String(nodeId)
          sigmaGraph.addEdge(isOutgoing ? newValue : otherNode, isOutgoing ? otherNode : newValue, attributes)
          sigmaGraph.dropEdge(edge)
        })

        sigmaGraph.dropNode(String(nodeId))

        const nodeIndex = rawGraph.nodeIdMap[String(nodeId)]
        if (nodeIndex !== undefined) {
          rawGraph.nodes[nodeIndex].id = newValue
          rawGraph.nodes[nodeIndex].properties.entity_id = newValue
          delete rawGraph.nodeIdMap[String(nodeId)]
          rawGraph.nodeIdMap[newValue] = nodeIndex
        }
      } else {
        const updatedAttributes = { ...nodeAttributes }
        if (propertyName === 'description') {
          updatedAttributes.description = newValue
        }
        Object.entries(updatedAttributes).forEach(([key, value]) => {
          sigmaGraph.setNodeAttribute(String(nodeId), key, value)
        })

        const nodeIndex = rawGraph.nodeIdMap[String(nodeId)]
        if (nodeIndex !== undefined) {
          rawGraph.nodes[nodeIndex].properties[propertyName] = newValue
        }
      }

      const selectedNode = useGraphStore.getState().selectedNode
      if (selectedNode === String(nodeId)) {
        useGraphStore.getState().setSelectedNode(newValue)
      }

      const focusedNode = useGraphStore.getState().focusedNode
      if (focusedNode === String(nodeId)) {
        useGraphStore.getState().setFocusedNode(newValue)
      }

      sigmaInstance.refresh()
    } catch (error) {
      console.error('Error updating node in graph:', error)
      throw new Error('Failed to update node in graph')
    }
  }

  const handleSave = async () => {
    if (isSubmitting) return

    if (editValue === String(value)) {
      setIsEditing(false)
      return
    }

    setIsSubmitting(true)

    try {
      const updatedData: Record<string, any> = {}

      if (entityType === 'node' && entityId) {
        if (name === 'entity_id') {
          if (editValue !== String(value)) {
            const exists = await checkEntityNameExists(editValue)
            if (exists) {
              toast.error(t('graphPanel.propertiesView.errors.duplicateName'))
              return
            }
          }
          updatedData['entity_name'] = editValue
          await updateEntity(String(value), updatedData, true)
          await updateGraphNode(String(value), 'entity_id', editValue)
        } else {
          updatedData[name] = editValue
          await updateEntity(entityId, updatedData)
          if (name === 'description') {
            await updateGraphNode(entityId, name, editValue)
          }
        }
        toast.success(t('graphPanel.propertiesView.success.entityUpdated'))
      } else if (entityType === 'edge' && sourceId && targetId) {
        updatedData[name] = editValue
        await updateRelation(sourceId, targetId, updatedData)
        toast.success(t('graphPanel.propertiesView.success.relationUpdated'))
      }

      if (onValueChange) {
        onValueChange(editValue)
      }

      useGraphStore.getState().setGraphDataFetchAttempted(false)
      useGraphStore.getState().setLabelsFetchAttempted(false)

      const currentNodeId = name === 'entity_id' ? editValue : (entityId || '')
      useGraphStore.getState().setSelectedNode(null)
      useGraphStore.getState().setSelectedNode(currentNodeId)
    } catch (error: any) {
      console.error('Error updating property:', error)
      let detailMessage = t('graphPanel.propertiesView.errors.updateFailed')

      if (error.response?.data?.detail) {
        detailMessage = error.response.data.detail
      } else if (error.response?.data?.message) {
        detailMessage = error.response.data.message
      } else if (error.message) {
        detailMessage = error.message
      }

      console.error('Update failed:', {
        entityType,
        entityId,
        propertyName: name,
        newValue: editValue,
        error: error.response?.data || error.message
      })

      toast.error(detailMessage, {
        description: t('graphPanel.propertiesView.errors.tryAgainLater')
      })
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
      <div className="flex items-center gap-1 text-primary/60 tracking-wide whitespace-nowrap">
        {getPropertyNameTranslation(name)}
        {isEditableField && (
          <div className="group relative">
            <PencilIcon className="w-3 h-3 opacity-50 hover:opacity-100" />
            <div className="absolute left-5 transform -translate-y-full -top-2 bg-primary/90 text-white text-xs px-3 py-1.5 rounded shadow-lg border border-primary/20 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-[100]">
              {t('graphPanel.propertiesView.doubleClickToEdit')}
            </div>
          </div>
        )}
      </div>:
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
        <div
          className={`flex-1 overflow-hidden ${isEditableField ? 'cursor-text' : ''}`}
          onDoubleClick={isEditableField ? handleDoubleClick : undefined}
        >
          <Text
            className="hover:bg-primary/20 rounded p-1 overflow-hidden text-ellipsis block w-full"
            tooltipClassName="max-w-80"
            text={String(value)}
            tooltip={tooltip || (typeof value === 'string' ? value : JSON.stringify(value, null, 2))}
            side="left"
            onClick={onClick}
          />
        </div>
      )}
    </div>
  )
}

export default EditablePropertyRow
