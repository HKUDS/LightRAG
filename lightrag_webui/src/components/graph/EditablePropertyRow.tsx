import { useState, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import Text from '@/components/ui/Text'
import Input from '@/components/ui/Input'
import { toast } from 'sonner'
import { updateEntity, updateRelation, checkEntityNameExists } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import { PencilIcon } from 'lucide-react'
import { tr } from '@faker-js/faker'

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
  const { t } = useTranslation()
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [currentValue, setCurrentValue] = useState(initialValue)
  const inputRef = useRef<HTMLInputElement>(null)

  // Update currentValue when initialValue changes
  useEffect(() => {
    setCurrentValue(initialValue)
  }, [initialValue])

  // Initialize edit value when entering edit mode
  useEffect(() => {
    if (isEditing) {
      setEditValue(String(currentValue))
      // Focus the input element when entering edit mode
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus()
          inputRef.current.select()
        }
      }, 50)
    }
  }, [isEditing, currentValue])

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

        interface EdgeToUpdate {
          originalDynamicId: string;
          newEdgeId: string;
          edgeIndex: number;
        }

        const edgesToUpdate: EdgeToUpdate[] = [];

        sigmaGraph.forEachEdge(String(nodeId), (edge, attributes, source, target) => {
          const otherNode = source === String(nodeId) ? target : source
          const isOutgoing = source === String(nodeId)

          // 获取原始边的dynamicId，以便后续更新edgeDynamicIdMap
          const originalEdgeDynamicId = edge
          const edgeIndexInRawGraph = rawGraph.edgeDynamicIdMap[originalEdgeDynamicId]

          // 创建新边并获取新边的ID
          const newEdgeId = sigmaGraph.addEdge(isOutgoing ? newValue : otherNode, isOutgoing ? otherNode : newValue, attributes)

          // 存储需要更新的边信息
          if (edgeIndexInRawGraph !== undefined) {
            edgesToUpdate.push({
              originalDynamicId: originalEdgeDynamicId,
              newEdgeId: newEdgeId,
              edgeIndex: edgeIndexInRawGraph
            })
          }

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

        // 更新边的引用关系
        edgesToUpdate.forEach(({ originalDynamicId, newEdgeId, edgeIndex }) => {
          // 更新边的source和target
          if (rawGraph.edges[edgeIndex]) {
            if (rawGraph.edges[edgeIndex].source === String(nodeId)) {
              rawGraph.edges[edgeIndex].source = newValue
            }
            if (rawGraph.edges[edgeIndex].target === String(nodeId)) {
              rawGraph.edges[edgeIndex].target = newValue
            }

            // 更新dynamicId映射
            rawGraph.edges[edgeIndex].dynamicId = newEdgeId
            delete rawGraph.edgeDynamicIdMap[originalDynamicId]
            rawGraph.edgeDynamicIdMap[newEdgeId] = edgeIndex
          }
        })

        useGraphStore.getState().setSelectedNode(editValue)
      } else {
        // const updatedAttributes = { ...nodeAttributes }
        // if (propertyName === 'description') {
        //   updatedAttributes.description = newValue
        // }
        // Object.entries(updatedAttributes).forEach(([key, value]) => {
        //   sigmaGraph.setNodeAttribute(String(nodeId), key, value)
        // })

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

  const updateGraphEdge = async (sourceId: string, targetId: string, propertyName: string, newValue: string) => {
    const sigmaInstance = useGraphStore.getState().sigmaInstance
    const sigmaGraph = useGraphStore.getState().sigmaGraph
    const rawGraph = useGraphStore.getState().rawGraph

    if (!sigmaInstance || !sigmaGraph || !rawGraph) {
      return
    }

    try {
      const allEdges = sigmaGraph.edges()
      let keyToUse = null

      for (const edge of allEdges) {
        const edgeSource = sigmaGraph.source(edge)
        const edgeTarget = sigmaGraph.target(edge)

        if ((edgeSource === sourceId && edgeTarget === targetId) ||
            (edgeSource === targetId && edgeTarget === sourceId)) {
          keyToUse = edge
          break
        }
      }

      if (keyToUse !== null) {
        if(propertyName === 'keywords') {
          sigmaGraph.setEdgeAttribute(keyToUse, 'label', newValue);
        } else {
          sigmaGraph.setEdgeAttribute(keyToUse, propertyName, newValue);
        }

        if (keyToUse && rawGraph.edgeDynamicIdMap[keyToUse] !== undefined) {
           const edgeIndex = rawGraph.edgeDynamicIdMap[keyToUse];
           if (rawGraph.edges[edgeIndex]) {
               rawGraph.edges[edgeIndex].properties[propertyName] = newValue;
           } else {
               console.warn(`Edge index ${edgeIndex} found but edge data missing in rawGraph for dynamicId ${entityId}`);
           }
        } else {
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
      // Log the specific edge key that caused the error if possible
      console.error(`Error updating edge ${sourceId}->${targetId} in graph:`, error);
      throw new Error('Failed to update edge in graph')
    }
  }

  const handleSave = async () => {
    if (isSubmitting) return

    if (editValue === String(currentValue)) {
      setIsEditing(false)
      return
    }

    setIsSubmitting(true)

    try {
      if (entityType === 'node' && entityId) {
        let updatedData = { [name]: editValue }

        if (name === 'entity_id') {
          const exists = await checkEntityNameExists(editValue)
          if (exists) {
            toast.error(t('graphPanel.propertiesView.errors.duplicateName'))
            setIsSubmitting(false)
            return
          }
          updatedData = { 'entity_name': editValue }
        }
        await updateEntity(entityId, updatedData, true)
        await updateGraphNode(entityId, name, editValue)
        toast.success(t('graphPanel.propertiesView.success.entityUpdated'))
      } else if (entityType === 'edge' && sourceId && targetId) {
        const updatedData = { [name]: editValue }
        await updateRelation(sourceId, targetId, updatedData)
        await updateGraphEdge(sourceId, targetId, name, editValue)
        toast.success(t('graphPanel.propertiesView.success.relationUpdated'))
      }

      setIsEditing(false)
      setCurrentValue(editValue)
    } catch (error) {
      console.error('Error updating property:', error)
      toast.error(t('graphPanel.propertiesView.errors.updateFailed'))
    } finally {
      setIsSubmitting(false)
    }
  }

  // Always render the property name label and edit icon, regardless of edit state
  return (
    <div className="flex items-center gap-1" onDoubleClick={handleDoubleClick}>
      <span className="text-primary/60 tracking-wide whitespace-nowrap">{getPropertyNameTranslation(name)}</span>
      <div className="group relative">
        <PencilIcon
          className="h-3 w-3 text-gray-500 hover:text-gray-700 cursor-pointer"
          onClick={() => setIsEditing(true)}
        />
        <div className="absolute left-5 transform -translate-y-full -top-2 bg-primary/90 text-white text-xs px-3 py-1.5 rounded shadow-lg border border-primary/20 opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap z-[100]">
          {t('graphPanel.propertiesView.doubleClickToEdit')}
        </div>
      </div>:
      {isEditing ? (
        // Render input field when editing
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
        // Render text component when not editing
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
