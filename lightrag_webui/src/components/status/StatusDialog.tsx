import { LightragStatus } from '@/api/lightrag'
import { useTranslation } from 'react-i18next'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/Dialog'
import StatusCard from './StatusCard'

interface StatusDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  status: LightragStatus | null
}

const StatusDialog = ({ open, onOpenChange, status }: StatusDialogProps) => {
  const { t } = useTranslation()

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-[920px]">
        <DialogHeader className="pr-6">
          <DialogTitle>{t('graphPanel.statusDialog.title')}</DialogTitle>
          <DialogDescription>
            {t('graphPanel.statusDialog.description')}
          </DialogDescription>
        </DialogHeader>
        <StatusCard status={status} />
      </DialogContent>
    </Dialog>
  )
}

export default StatusDialog
