import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert'
import Button from '@/components/ui/Button'
import { useBackendState } from '@/stores/state'
import { controlButtonVariant } from '@/lib/constants'

import { AlertCircle } from 'lucide-react'

const MessageAlert = () => {
  const health = useBackendState.use.health()
  const message = useBackendState.use.message()
  const messageTitle = useBackendState.use.messageTitle()

  return (
    <Alert
      variant={health ? 'default' : 'destructive'}
      className="bg-background/90 absolute top-1/2 left-1/2 w-auto -translate-x-1/2 -translate-y-1/2 transform backdrop-blur-lg"
    >
      {!health && <AlertCircle className="h-4 w-4" />}
      <AlertTitle>{messageTitle}</AlertTitle>

      <AlertDescription>{message}</AlertDescription>
      <div className="h-2" />
      <div className="flex">
        <div className="flex-auto" />
        <Button
          size="sm"
          variant={controlButtonVariant}
          className="text-primary max-h-8 border !p-2 text-xs"
          onClick={() => useBackendState.getState().clear()}
        >
          Continue
        </Button>
      </div>
    </Alert>
  )
}

export default MessageAlert
