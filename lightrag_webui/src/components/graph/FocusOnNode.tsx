import { useCamera, useSigma } from '@react-sigma/core'
import { useEffect } from 'react'
import { useGraphStore } from '@/stores/graph'

/**
 * Component that highlights a node and centers the camera on it.
 */
const FocusOnNode = ({ node, move }: { node: string | null; move?: boolean }) => {
  const sigma = useSigma()
  const { gotoNode } = useCamera()

  /**
   * When the selected item changes, highlighted the node and center the camera on it.
   */
  useEffect(() => {
    if (move) {
      if (node) {
        sigma.getGraph().setNodeAttribute(node, 'highlighted', true)
        gotoNode(node)
      } else {
        // If no node is selected but move is true, reset to default view
        sigma.setCustomBBox(null)
        sigma.getCamera().animate({ x: 0.5, y: 0.5, ratio: 1 }, { duration: 0 })
      }
      useGraphStore.getState().setMoveToSelectedNode(false)
    } else if (node) {
      sigma.getGraph().setNodeAttribute(node, 'highlighted', true)
    }

    return () => {
      if (node) {
        sigma.getGraph().setNodeAttribute(node, 'highlighted', false)
      }
    }
  }, [node, move, sigma, gotoNode])

  return null
}

export default FocusOnNode
