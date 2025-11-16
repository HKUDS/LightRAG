import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import Button from '@/components/ui/Button'

interface MergeDialogProps {
  mergeDialogOpen: boolean
  mergeDialogInfo: {
    targetEntity: string
    sourceEntity: string
  } | null
  onOpenChange: (open: boolean) => void
  onRefresh: (useMergedStart: boolean) => void
}

/**
 * MergeDialog component that appears after a successful entity merge
 * Allows user to choose whether to use the merged entity or keep current start point
 */
const MergeDialog = ({
  mergeDialogOpen,
  mergeDialogInfo,
  onOpenChange,
  onRefresh
}: MergeDialogProps) => {
  const { t } = useTranslation()
  const currentQueryLabel = useSettingsStore.use.queryLabel()

  return (
    <Dialog open={mergeDialogOpen} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('graphPanel.propertiesView.mergeDialog.title')}</DialogTitle>
          <DialogDescription>
            {t('graphPanel.propertiesView.mergeDialog.description', {
              source: mergeDialogInfo?.sourceEntity ?? '',
              target: mergeDialogInfo?.targetEntity ?? '',
            })}
          </DialogDescription>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">
          {t('graphPanel.propertiesView.mergeDialog.refreshHint')}
        </p>
        <DialogFooter className="mt-4 flex-col gap-2 sm:flex-row sm:justify-end">
          {currentQueryLabel !== mergeDialogInfo?.sourceEntity && (
            <Button
              type="button"
              variant="outline"
              onClick={() => onRefresh(false)}
            >
              {t('graphPanel.propertiesView.mergeDialog.keepCurrentStart')}
            </Button>
          )}
          <Button type="button" onClick={() => onRefresh(true)}>
            {t('graphPanel.propertiesView.mergeDialog.useMergedStart')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default MergeDialog
