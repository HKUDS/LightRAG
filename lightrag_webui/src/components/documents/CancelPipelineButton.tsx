import { cancelPipeline } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import { errorMessage } from '@/lib/utils'
import { BanIcon } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'

type CancelPipelineButtonProps = {
  busy: boolean
}

export default function CancelPipelineButton({ busy }: CancelPipelineButtonProps) {
  const { t } = useTranslation()
  const [isCancelling, setIsCancelling] = useState(false)

  return (
    <Button
      variant="destructive"
      size="sm"
      side="bottom"
      tooltip={t('documentPanel.documentManager.cancelPipelineTooltip')}
      disabled={!busy || isCancelling}
      onClick={async () => {
        if (
          !window.confirm(t('documentPanel.pipelineStatus.cancelConfirmDescription'))
        ) {
          return
        }

        setIsCancelling(true)
        try {
          const response = await cancelPipeline()
          if (response.status === 'cancellation_requested') {
            toast.success(t('documentPanel.pipelineStatus.cancelSuccess'))
          } else {
            toast.info(t('documentPanel.pipelineStatus.cancelNotBusy'))
          }
        } catch (error) {
          toast.error(
            t('documentPanel.pipelineStatus.cancelFailed', {
              error: errorMessage(error)
            })
          )
        } finally {
          setIsCancelling(false)
        }
      }}
    >
      <BanIcon />
      {t('documentPanel.documentManager.cancelPipelineButton')}
    </Button>
  )
}
