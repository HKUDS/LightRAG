import { useState, useCallback, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle
} from '@/components/ui/AlertDialog'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'
import { InvalidApiKeyError, RequireApiKeError } from '@/api/lightrag'

/**
 * Why a close was requested. A host that reacts to the close needs to tell
 * the two apart: 'save' committed a new key and is worth re-verifying,
 * 'dismiss' changed nothing — re-verifying it can only fail again, and a
 * host that reopens on failure would trap the user in the dialog.
 */
export type ApiKeyAlertCloseReason = 'save' | 'dismiss'

interface ApiKeyAlertProps {
  open: boolean;
  onOpenChange: (open: boolean, reason?: ApiKeyAlertCloseReason) => void;
}

const ApiKeyAlert = ({ open: opened, onOpenChange: setOpened }: ApiKeyAlertProps) => {
  const { t } = useTranslation()
  const apiKey = useSettingsStore.use.apiKey()
  const [tempApiKey, setTempApiKey] = useState<string>(apiKey || '')
  const message = useBackendState.use.message()

  // Sync draft input with latest store value whenever the dialog opens or apiKey changes
  useEffect(() => {
    const timer = setTimeout(() => setTempApiKey(apiKey || ''), 0)
    return () => clearTimeout(timer)
  }, [apiKey, opened])

  useEffect(() => {
    if (message) {
      if (message.includes(InvalidApiKeyError) || message.includes(RequireApiKeError)) {
        setOpened(true)
      }
    }
  }, [message, setOpened])

  const setApiKey = useCallback(() => {
    useSettingsStore.setState({ apiKey: tempApiKey || null })
    setOpened(false, 'save')
  }, [tempApiKey, setOpened])

  // Radix reports only the boolean; every close it drives (the cancel button,
  // Escape) is a dismissal — the save path calls setOpened itself.
  const handleOpenChange = useCallback(
    (next: boolean) => setOpened(next, next ? undefined : 'dismiss'),
    [setOpened]
  )

  const handleTempApiKeyChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setTempApiKey(e.target.value)
    },
    [setTempApiKey]
  )

  return (
    <AlertDialog open={opened} onOpenChange={handleOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t('apiKeyAlert.title')}</AlertDialogTitle>
          <AlertDialogDescription>
            {t('apiKeyAlert.description')}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="flex flex-col gap-4">
          <form className="flex gap-2" onSubmit={(e) => e.preventDefault()}>
            <Input
              type="password"
              value={tempApiKey}
              onChange={handleTempApiKeyChange}
              placeholder={t('apiKeyAlert.placeholder')}
              className="max-h-full w-full min-w-0"
              autoComplete="off"
            />

            <Button onClick={setApiKey} variant="outline" size="sm">
              {t('apiKeyAlert.save')}
            </Button>
          </form>
          {message && (
            <div className="text-sm text-red-500">
              {message}
            </div>
          )}
        </div>
        {/* The only way out for someone who has no valid key: this dialog is
            modal and its overlay covers the header, and Escape is not
            reachable on the touch devices the workspace entry targets. */}
        <AlertDialogFooter>
          <AlertDialogCancel className="min-h-11">
            {t('common.cancel')}
          </AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

export default ApiKeyAlert
