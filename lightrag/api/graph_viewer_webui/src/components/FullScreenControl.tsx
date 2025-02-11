import { useFullScreen } from '@react-sigma/core'
import { MaximizeIcon, MinimizeIcon } from 'lucide-react'
import { controlButtonVariant } from '@/lib/constants'
import Button from '@/components/ui/Button'

/**
 * Component that toggles full screen mode.
 */
const FullScreenControl = () => {
  const { isFullScreen, toggle } = useFullScreen()

  return (
    <>
      {isFullScreen ? (
        <Button variant={controlButtonVariant} onClick={toggle} tooltip="Windowed" size="icon">
          <MinimizeIcon />
        </Button>
      ) : (
        <Button variant={controlButtonVariant} onClick={toggle} tooltip="Full Screen" size="icon">
          <MaximizeIcon />
        </Button>
      )}
    </>
  )
}

export default FullScreenControl
