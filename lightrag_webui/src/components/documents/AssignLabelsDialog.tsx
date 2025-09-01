import { useState, useEffect } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/Dialog'
import LabelSelector from '@/components/ui/LabelSelector'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { assignLabelsToDocuments } from '@/api/lightrag'
import { TagIcon, CheckIcon } from 'lucide-react'

interface AssignLabelsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  selectedDocIds: string[]
  onLabelsAssigned: () => void
}

export default function AssignLabelsDialog({
  open,
  onOpenChange,
  selectedDocIds,
  onLabelsAssigned
}: AssignLabelsDialogProps) {
  const [selectedLabels, setSelectedLabels] = useState<string[]>([])
  const [isAssigning, setIsAssigning] = useState(false)

  useEffect(() => {
    if (open) {
      // Reset state when dialog opens
      setSelectedLabels([])
    }
  }, [open])

  const handleAssignLabels = async () => {
    if (selectedLabels.length === 0) {
      toast.error('Please select at least one label')
      return
    }

    try {
      setIsAssigning(true)

      const response = await assignLabelsToDocuments({
        doc_ids: selectedDocIds,
        label_names: selectedLabels
      })

      // Count successful assignments
      const successCount = Object.values(response.results).filter(
        (result: any) => result.success
      ).length

      if (successCount === selectedDocIds.length) {
        toast.success(`Successfully assigned labels to ${successCount} documents`)
      } else {
        toast.warning(`Assigned labels to ${successCount} of ${selectedDocIds.length} documents`)
      }

      onLabelsAssigned()
      onOpenChange(false)

    } catch (error) {
      console.error('Error assigning labels:', error)
      toast.error(`Failed to assign labels: ${errorMessage(error)}`)
    } finally {
      setIsAssigning(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TagIcon className="h-5 w-5" />
            Assign Labels to Documents
          </DialogTitle>
          <DialogDescription>
            Select labels to assign to {selectedDocIds.length} selected document{selectedDocIds.length > 1 ? 's' : ''}
          </DialogDescription>
        </DialogHeader>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Select Labels</CardTitle>
          </CardHeader>
          <CardContent>
            <LabelSelector
              selectedLabels={selectedLabels}
              onLabelsChange={setSelectedLabels}
              placeholder="Choose labels to assign..."
              multiple={true}
            />
          </CardContent>
        </Card>

        <DialogFooter>
          <Button
            variant="secondary"
            onClick={() => onOpenChange(false)}
            disabled={isAssigning}
          >
            Cancel
          </Button>
          <Button
            onClick={handleAssignLabels}
            disabled={isAssigning || selectedLabels.length === 0}
            className="gap-2"
          >
            {isAssigning ? (
              'Assigning...'
            ) : (
              <>
                <CheckIcon className="h-4 w-4" />
                Assign Labels
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
