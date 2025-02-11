import { useCamera } from '@react-sigma/core'
import { useCallback } from 'react'
import Button from '@/components/ui/Button'
import { ZoomInIcon, ZoomOutIcon, FullscreenIcon } from 'lucide-react'
import { controlButtonVariant } from '@/lib/constants'

/**
 * Component that provides zoom controls for the graph viewer.
 */
const ZoomControl = () => {
  const { zoomIn, zoomOut, reset } = useCamera({ duration: 200, factor: 1.5 })

  const handleZoomIn = useCallback(() => zoomIn(), [zoomIn])
  const handleZoomOut = useCallback(() => zoomOut(), [zoomOut])
  const handleResetZoom = useCallback(() => reset(), [reset])

  return (
    <>
      <Button variant={controlButtonVariant} onClick={handleZoomIn} tooltip="Zoom In" size="icon">
        <ZoomInIcon />
      </Button>
      <Button variant={controlButtonVariant} onClick={handleZoomOut} tooltip="Zoom Out" size="icon">
        <ZoomOutIcon />
      </Button>
      <Button
        variant={controlButtonVariant}
        onClick={handleResetZoom}
        tooltip="Reset Zoom"
        size="icon"
      >
        <FullscreenIcon />
      </Button>
    </>
  )
}

export default ZoomControl
