import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert'
import { useBackendState } from '@/stores/state'
import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

// import Button from '@/components/ui/Button'
// import { controlButtonVariant } from '@/lib/constants'

import { AlertCircle } from 'lucide-react'

const MessageAlert = () => {
  const health = useBackendState.use.health()
  const message = useBackendState.use.message()
  const messageTitle = useBackendState.use.messageTitle()
  const [isMounted, setIsMounted] = useState(false)

  useEffect(() => {
    setTimeout(() => {
      setIsMounted(true)
    }, 50)
  }, [])

  return (
    <Alert
      // variant={health ? 'default' : 'destructive'}
      className={cn(
        'bg-background/90 absolute top-12 left-1/2 flex w-auto max-w-lg -translate-x-1/2 transform items-center gap-4 shadow-md backdrop-blur-lg transition-all duration-500 ease-in-out',
        isMounted ? 'translate-y-0 opacity-100' : '-translate-y-20 opacity-0',
        !health && 'bg-red-700 text-white'
      )}
    >
      {!health && (
        <div>
          <AlertCircle className="size-4" />
        </div>
      )}
      <div>
        <AlertTitle className="font-bold">{messageTitle}</AlertTitle>
        <AlertDescription>{message}</AlertDescription>
      </div>
      {/* <div className="flex">
        <div className="flex-auto" />
        <Button
          size="sm"
          variant={controlButtonVariant}
          className="border-primary max-h-8 border !p-2 text-xs"
          onClick={() => useBackendState.getState().clear()}
        >
          Close
        </Button>
      </div> */}
    </Alert>
  )
}

export default MessageAlert
