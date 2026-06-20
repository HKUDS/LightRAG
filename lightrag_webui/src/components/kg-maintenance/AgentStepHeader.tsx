import Button from '@/components/ui/Button'
import { ArrowRightIcon } from 'lucide-react'
import type { KGMaintenanceNextAction } from './kgMaintenanceNextAction'

export type AgentStepHeaderBadge = {
  label: string
  value: string
}

export interface AgentStepHeaderProps {
  title: string
  description: string
  action: KGMaintenanceNextAction
  badges?: AgentStepHeaderBadge[]
  disabled?: boolean
  loading?: boolean
  busyLabel?: string
  onAction: (action: KGMaintenanceNextAction) => void
}

export function AgentStepHeader({
  title,
  description,
  action,
  badges = [],
  disabled = false,
  loading = false,
  busyLabel,
  onAction
}: AgentStepHeaderProps) {
  const actionDisabled = disabled || loading
  const actionLabel = actionDisabled && busyLabel ? busyLabel : action.label

  return (
    <header className="border-border/70 flex flex-col gap-3 border-b pb-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0 space-y-2">
        <div className="space-y-1">
          <h2 className="text-foreground text-base font-semibold">{title}</h2>
          <p className="text-muted-foreground text-sm">{description}</p>
        </div>
        <p className="text-foreground text-sm">{action.reason}</p>
        {badges.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {badges.map((badge, index) => (
              <span
                key={index}
                className="border-border bg-muted/40 text-muted-foreground inline-flex h-7 items-center rounded-md border px-2 text-xs font-medium"
              >
                {badge.label} {badge.value}
              </span>
            ))}
          </div>
        )}
      </div>

      <Button
        type="button"
        size="sm"
        className="shrink-0 self-start"
        disabled={actionDisabled}
        onClick={() => {
          if (actionDisabled) return
          onAction(action)
        }}
      >
        {actionLabel}
        <ArrowRightIcon className="size-4" />
      </Button>
    </header>
  )
}

export default AgentStepHeader
