import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'

type PromptVersionDiffDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  diffData: {
    changes: Record<string, { before: unknown; after: unknown }>
  } | null
}

export default function PromptVersionDiffDialog({
  open,
  onOpenChange,
  diffData
}: PromptVersionDiffDialogProps) {
  const entries = Object.entries(diffData?.changes || {})

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Version Diff</DialogTitle>
          <DialogDescription>
            Compare the selected version against the active version or chosen base.
          </DialogDescription>
        </DialogHeader>
        <div className="max-h-[70vh] space-y-4 overflow-auto">
          {entries.length === 0 ? (
            <div className="text-sm text-muted-foreground">No changes to display.</div>
          ) : (
            entries.map(([key, value]) => (
              <div key={key} className="rounded-lg border p-3">
                <div className="mb-2 text-sm font-semibold">{key}</div>
                <div className="grid gap-3 md:grid-cols-2">
                  <pre className="overflow-auto rounded bg-muted/50 p-3 text-xs whitespace-pre-wrap">
                    {JSON.stringify(value.before, null, 2)}
                  </pre>
                  <pre className="overflow-auto rounded bg-muted/50 p-3 text-xs whitespace-pre-wrap">
                    {JSON.stringify(value.after, null, 2)}
                  </pre>
                </div>
              </div>
            ))
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
