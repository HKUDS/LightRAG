import { useState, useCallback } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { clearDocuments } from '@/api/lightrag'

import { EraserIcon } from 'lucide-react'

export default function ClearDocumentsDialog() {
  const [open, setOpen] = useState(false)

  const handleClear = useCallback(async () => {
    try {
      const result = await clearDocuments()
      if (result.status === 'success') {
        toast.success('Documents cleared successfully')
        setOpen(false)
      } else {
        toast.error(`Clear Documents Failed:\n${result.message}`)
      }
    } catch (err) {
      toast.error('Clear Documents Failed:\n' + errorMessage(err))
    }
  }, [setOpen])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" side="bottom" tooltip='Clear documents' size="sm">
          <EraserIcon/> Clear
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>Clear documents</DialogTitle>
          <DialogDescription>Do you really want to clear all documents?</DialogDescription>
        </DialogHeader>
        <Button variant="destructive" onClick={handleClear}>
          YES
        </Button>
      </DialogContent>
    </Dialog>
  )
}
