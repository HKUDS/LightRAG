import { useMemo } from 'react'
import { FileTextIcon, XIcon } from 'lucide-react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import { cn } from '@/lib/utils'
import { WORKFLOW_STEPS, type KGMaintenanceWorkflowStepId } from './kgMaintenanceArtifacts'

export type DrawerArtifactStatus = 'generated' | 'missing'

export type DrawerArtifact = {
  key: string
  title: string
  sourceFile: string
  zhFile: string
  step: KGMaintenanceWorkflowStepId
  status: DrawerArtifactStatus
}

interface ArtifactDrawerProps {
  open: boolean
  artifacts: DrawerArtifact[]
  onClose: () => void
  onOpenArtifact: (artifactKey: string) => void
}

const stepLabels: Record<KGMaintenanceWorkflowStepId, string> = {
  check: '检查知识库',
  'llm-review': 'LLM 审阅',
  approval: 'Proposal 审批',
  execute: '执行变更',
  validate: '验证结果'
}

const statusPresentation: Record<
  DrawerArtifactStatus,
  {
    label: string
    className: string
  }
> = {
  generated: {
    label: '已生成',
    className: 'bg-emerald-500/10 text-emerald-700'
  },
  missing: {
    label: '缺失',
    className: 'bg-muted text-muted-foreground'
  }
}

export function ArtifactDrawer({ open, artifacts, onClose, onOpenArtifact }: ArtifactDrawerProps) {
  const artifactsByStep = useMemo(
    () =>
      WORKFLOW_STEPS.map((step) => ({
        step,
        label: stepLabels[step],
        artifacts: artifacts.filter((artifact) => artifact.step === step)
      })),
    [artifacts]
  )

  if (!open) return null

  const drawerSurface = (
    <div className="flex h-full min-h-0 flex-col">
      <DialogHeader className="border-border/70 flex-row items-start justify-between gap-3 space-y-0 border-b p-4 text-left">
        <div className="min-w-0">
          <DialogTitle className="text-sm">全部产物</DialogTitle>
          <DialogDescription className="mt-1 text-xs">
            按维护流程查看已生成和缺失的工作产物。
          </DialogDescription>
        </div>
        <DialogClose asChild>
          <Button type="button" variant="outline" size="sm">
            <XIcon className="size-4" />
            关闭
          </Button>
        </DialogClose>
      </DialogHeader>

      <div className="min-h-0 flex-1 overflow-auto p-4">
        <div className="space-y-4">
          {artifactsByStep.map((group) => (
            <section key={group.step} aria-labelledby={`artifact-step-${group.step}`}>
              <div className="mb-2 flex items-center justify-between gap-3">
                <h3 id={`artifact-step-${group.step}`} className="text-sm font-semibold">
                  {group.label}
                </h3>
                <span className="text-muted-foreground text-xs">
                  {group.artifacts.length} 个产物
                </span>
              </div>

              {group.artifacts.length === 0 ? (
                <p className="text-muted-foreground rounded-md border border-dashed px-3 py-2 text-sm">
                  暂无产物。
                </p>
              ) : (
                <div className="space-y-2">
                  {group.artifacts.map((artifact) => {
                    const status = statusPresentation[artifact.status]

                    return (
                      <button
                        key={artifact.key}
                        type="button"
                        onClick={() => onOpenArtifact(artifact.key)}
                        className="border-border/70 hover:bg-muted/40 focus-visible:ring-ring flex w-full items-start gap-3 rounded-md border px-3 py-2 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none"
                      >
                        <FileTextIcon className="text-muted-foreground mt-0.5 size-4 shrink-0" />
                        <span className="min-w-0 flex-1">
                          <span className="flex flex-wrap items-center gap-2">
                            <span className="text-sm font-medium">{artifact.title}</span>
                            <span
                              className={cn('rounded px-1.5 py-0.5 text-xs', status.className)}
                            >
                              {status.label}
                            </span>
                          </span>
                          <span className="text-muted-foreground mt-1 block truncate text-xs">
                            {artifact.zhFile}
                          </span>
                          <span className="text-muted-foreground mt-0.5 block truncate text-xs">
                            {artifact.sourceFile}
                          </span>
                        </span>
                      </button>
                    )
                  })}
                </div>
              )}
            </section>
          ))}
        </div>
      </div>
    </div>
  )

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <DialogContent
        aria-label="全部产物"
        className="top-0 right-0 left-auto h-dvh max-h-dvh w-full max-w-lg translate-x-0 translate-y-0 gap-0 border-y-0 border-r-0 border-l p-0 shadow-lg duration-200 sm:rounded-none [&>button.absolute]:hidden"
      >
        {drawerSurface}
      </DialogContent>
      {typeof window === 'undefined' && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="全部产物"
          className="bg-background border-border fixed inset-y-0 right-0 z-50 flex w-full max-w-lg flex-col border-l shadow-lg"
        >
          {drawerSurface}
        </div>
      )}
    </Dialog>
  )
}
