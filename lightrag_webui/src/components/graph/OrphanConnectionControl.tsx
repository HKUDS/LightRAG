import { useState } from 'react'
import { Link } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import Button from '@/components/ui/Button'
import OrphanConnectionDialog from './OrphanConnectionDialog'
import { controlButtonVariant } from '@/lib/constants'
import { useBackendState } from '@/stores/state'

/**
 * Control button for orphan entity connection.
 * Only visible when AUTO_CONNECT_ORPHANS is disabled (manual mode).
 */
export default function OrphanConnectionControl() {
  const { t } = useTranslation()
  const [showDialog, setShowDialog] = useState(false)
  const status = useBackendState.use.status()

  // Only show when auto_connect_orphans is explicitly false (manual mode)
  if (status?.configuration?.auto_connect_orphans !== false) {
    return null
  }

  return (
    <>
      <Button
        variant={controlButtonVariant}
        tooltip={t('graphPanel.orphanConnection.tooltip')}
        size="icon"
        onClick={() => setShowDialog(true)}
      >
        <Link />
      </Button>
      <OrphanConnectionDialog
        open={showDialog}
        onOpenChange={setShowDialog}
      />
    </>
  )
}
