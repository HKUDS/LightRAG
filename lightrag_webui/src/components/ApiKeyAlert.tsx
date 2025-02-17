import { useState, useCallback, useEffect } from 'react'
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogTitle
} from '@/components/ui/AlertDialog'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'
import { InvalidApiKeyError, RequireApiKeError } from '@/api/lightrag'

import { toast } from 'sonner'

const ApiKeyAlert = () => {
  const [opened, setOpened] = useState<boolean>(true)
  const apiKey = useSettingsStore.use.apiKey()
  const [tempApiKey, setTempApiKey] = useState<string>('')
  const message = useBackendState.use.message()

  useEffect(() => {
    setTempApiKey(apiKey || '')
  }, [apiKey, opened])

  useEffect(() => {
    if (message) {
      if (message.includes(InvalidApiKeyError) || message.includes(RequireApiKeError)) {
        setOpened(true)
      }
    }
  }, [message, setOpened])

  const setApiKey = useCallback(async () => {
    useSettingsStore.setState({ apiKey: tempApiKey || null })
    if (await useBackendState.getState().check()) {
      setOpened(false)
      return
    }
    toast.error('API Key is invalid')
  }, [tempApiKey])

  const handleTempApiKeyChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setTempApiKey(e.target.value)
    },
    [setTempApiKey]
  )

  return (
    <AlertDialog open={opened} onOpenChange={setOpened}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>API Key is required</AlertDialogTitle>
          <AlertDialogDescription>Please enter your API key</AlertDialogDescription>
        </AlertDialogHeader>
        <form className="flex gap-2" onSubmit={(e) => e.preventDefault()}>
          <Input
            type="password"
            value={tempApiKey}
            onChange={handleTempApiKeyChange}
            placeholder="Enter your API key"
            className="max-h-full w-full min-w-0"
            autoComplete="off"
          />

          <Button onClick={setApiKey} variant="outline" size="sm">
            Save
          </Button>
        </form>
      </AlertDialogContent>
    </AlertDialog>
  )
}

export default ApiKeyAlert
