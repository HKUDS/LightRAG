import { useCamera, useSigma } from '@react-sigma/core'
import { useEffect } from 'react'

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
    if (!node) return
    sigma.getGraph().setNodeAttribute(node, 'highlighted', true)
    if (move) gotoNode(node)

    return () => {
      sigma.getGraph().setNodeAttribute(node, 'highlighted', false)
    }
  }, [node, move, sigma, gotoNode])

  return null
}

export default FocusOnNode