import { useCamera, useSigma } from '@react-sigma/core'
import { useCallback } from 'react'
import Button from '@/components/ui/Button'
import { ZoomInIcon, ZoomOutIcon, FullscreenIcon } from 'lucide-react'
import { controlButtonVariant } from '@/lib/constants'
import { useTranslation } from 'react-i18next';

/**
 * Component that provides zoom controls for the graph viewer.
 */
const ZoomControl = () => {
  const { zoomIn, zoomOut, reset } = useCamera({ duration: 200, factor: 1.5 })
  const sigma = useSigma()
  const { t } = useTranslation();

  const handleZoomIn = useCallback(() => zoomIn(), [zoomIn])
  const handleZoomOut = useCallback(() => zoomOut(), [zoomOut])
  const handleResetZoom = useCallback(() => {
    if (!sigma) return

    try {
      // First clear any custom bounding box and refresh
      sigma.setCustomBBox(null)
      sigma.refresh()

      // Get graph after refresh
      const graph = sigma.getGraph()

      // Check if graph has nodes before accessing them
      if (!graph || graph.nodes().length === 0) {
        reset()
        return
      }

      // Get container dimensions for aspect ratio
      const container = sigma.getContainer()
      const containerWidth = container.offsetWidth
      const containerHeight = container.offsetHeight
      console.log('Container W:', containerWidth, 'H:', containerHeight)

      // Get all node positions
      const nodePositions = graph.nodes().map(node => ({
        x: graph.getNodeAttribute(node, 'x'),
        y: graph.getNodeAttribute(node, 'y')
      }))

      // Calculate bounding box
      const minX = Math.min(...nodePositions.map(pos => pos.x))
      const maxX = Math.max(...nodePositions.map(pos => pos.x))
      const minY = Math.min(...nodePositions.map(pos => pos.y))
      const maxY = Math.max(...nodePositions.map(pos => pos.y))

      // Calculate graph dimensions with minimal padding
      const width = maxX - minX
      const height = maxY - minY
      const padding = Math.max(width, height) * 0.05
      console.log('Graph W:', Math.round(width*100)/100, 'H:', Math.round(height*100)/100)

      // Calculate base scale
      const scale = Math.min(
        containerWidth / (width + padding * 2),
        containerHeight / (height + padding * 2)
      )
      // Apply scaling factor (just don't know why)
      const ratio = (1 / scale) * 10

      console.log('scale:', Math.round(scale*100)/100, 'ratio:', Math.round(ratio*100)/100)

      // Animate to center with calculated ratio
      sigma.getCamera().animate(
        { x: 0.5, y: 0.5, ratio },
        { duration: 500 }
      )
    } catch (error) {
      console.error('Error resetting zoom:', error)
    }
  }, [sigma, reset])

  return (
    <>
      <Button variant={controlButtonVariant} onClick={handleZoomIn} tooltip={t('graphPanel.sideBar.zoomControl.zoomIn')} size="icon">
        <ZoomInIcon />
      </Button>
      <Button variant={controlButtonVariant} onClick={handleZoomOut} tooltip={t('graphPanel.sideBar.zoomControl.zoomOut')} size="icon">
        <ZoomOutIcon />
      </Button>
      <Button
        variant={controlButtonVariant}
        onClick={handleResetZoom}
        tooltip={t('graphPanel.sideBar.zoomControl.resetZoom')}
        size="icon"
      >
        <FullscreenIcon />
      </Button>
    </>
  )
}

export default ZoomControl
