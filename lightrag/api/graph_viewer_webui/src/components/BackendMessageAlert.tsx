import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert'
import { useBackendState } from '@/stores/state'
import { AlertCircle } from 'lucide-react'

const BackendMessageAlert = () => {
  const health = useBackendState.use.health()
  const message = useBackendState.use.message()
  const messageTitle = useBackendState.use.messageTitle()

  return (
    <Alert
      variant={health ? 'default' : 'destructive'}
      className="absolute top-1/2 left-1/2 w-auto -translate-x-1/2 -translate-y-1/2 transform"
    >
      {!health && <AlertCircle className="h-4 w-4" />}
      <AlertTitle>{messageTitle}</AlertTitle>
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  )
}

export default BackendMessageAlert
