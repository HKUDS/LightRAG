import { useCamera, useSigma } from '@react-sigma/core'
import { FullscreenIcon, RotateCcwIcon, RotateCwIcon, ZoomInIcon, ZoomOutIcon } from 'lucide-react'
import { useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import { controlButtonVariant } from '@/lib/constants'

/**
 * Component that provides zoom controls for the graph viewer.
 */
const ZoomControl = () => {
  const { zoomIn, zoomOut, reset } = useCamera({ duration: 200, factor: 1.5 })
  const sigma = useSigma()
  const { t } = useTranslation()

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
      if (!graph?.order || graph.nodes().length === 0) {
        // Use reset() for empty graph case
        reset()
        return
      }

      sigma.getCamera().animate({ x: 0.5, y: 0.5, ratio: 1.1 }, { duration: 1000 })
    } catch (error) {
      console.error('Error resetting zoom:', error)
      // Use reset() as fallback on error
      reset()
    }
  }, [sigma, reset])

  const handleRotate = useCallback(() => {
    if (!sigma) return

    const camera = sigma.getCamera()
    const currentAngle = camera.angle
    const newAngle = currentAngle + Math.PI / 8

    camera.animate({ angle: newAngle }, { duration: 200 })
  }, [sigma])

  const handleRotateCounterClockwise = useCallback(() => {
    if (!sigma) return

    const camera = sigma.getCamera()
    const currentAngle = camera.angle
    const newAngle = currentAngle - Math.PI / 8

    camera.animate({ angle: newAngle }, { duration: 200 })
  }, [sigma])

  return (
    <>
      <Button
        variant={controlButtonVariant}
        onClick={handleRotate}
        tooltip={t('graphPanel.sideBar.zoomControl.rotateCamera')}
        size="icon"
      >
        <RotateCwIcon />
      </Button>
      <Button
        variant={controlButtonVariant}
        onClick={handleRotateCounterClockwise}
        tooltip={t('graphPanel.sideBar.zoomControl.rotateCameraCounterClockwise')}
        size="icon"
      >
        <RotateCcwIcon />
      </Button>
      <Button
        variant={controlButtonVariant}
        onClick={handleResetZoom}
        tooltip={t('graphPanel.sideBar.zoomControl.resetZoom')}
        size="icon"
      >
        <FullscreenIcon />
      </Button>
      <Button
        variant={controlButtonVariant}
        onClick={handleZoomIn}
        tooltip={t('graphPanel.sideBar.zoomControl.zoomIn')}
        size="icon"
      >
        <ZoomInIcon />
      </Button>
      <Button
        variant={controlButtonVariant}
        onClick={handleZoomOut}
        tooltip={t('graphPanel.sideBar.zoomControl.zoomOut')}
        size="icon"
      >
        <ZoomOutIcon />
      </Button>
    </>
  )
}

export default ZoomControl
