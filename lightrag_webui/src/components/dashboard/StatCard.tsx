import * as React from 'react'
import { cn } from '@/lib/utils'
import { Card } from '@/components/ui/Card'
import type { LucideIcon } from 'lucide-react'

/**
 * Semantic accent for a stat card. Maps onto the `--status-*` design tokens so
 * colours stay consistent between cards, badges and charts.
 */
export type StatTone = 'brand' | 'success' | 'warning' | 'danger' | 'info' | 'neutral'

const TONE_STYLES: Record<StatTone, { icon: string; glow: string; value: string }> = {
  brand: {
    icon: 'bg-cyan-500/12 text-cyan-400 ring-cyan-500/22',
    glow: 'from-cyan-500/16',
    value: 'text-foreground'
  },
  success: {
    icon: 'bg-emerald-500/12 text-emerald-400 ring-emerald-500/22',
    glow: 'from-emerald-500/16',
    value: 'text-foreground'
  },
  warning: {
    icon: 'bg-amber-500/12 text-amber-400 ring-amber-500/22',
    glow: 'from-amber-500/16',
    value: 'text-foreground'
  },
  danger: {
    icon: 'bg-rose-500/12 text-rose-400 ring-rose-500/22',
    glow: 'from-rose-500/16',
    value: 'text-foreground'
  },
  info: {
    icon: 'bg-blue-500/12 text-blue-400 ring-blue-500/22',
    glow: 'from-blue-500/16',
    value: 'text-foreground'
  },
  neutral: {
    icon: 'bg-slate-500/12 text-slate-300 ring-slate-500/22',
    glow: 'from-slate-500/12',
    value: 'text-foreground'
  }
}

export interface StatCardProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'> {
  label: string
  value: number | string
  icon?: LucideIcon
  tone?: StatTone
  /** Secondary line under the value, e.g. "3 processing". */
  hint?: string
  /** Renders a shimmer placeholder instead of the value. */
  loading?: boolean
  onClick?: () => void
}

/**
 * A single dashboard metric tile.
 *
 * Numbers are locale-formatted so large counts stay readable. When `onClick`
 * is provided the tile becomes a real keyboard-accessible button.
 */
export const StatCard = React.memo(function StatCard({
  label,
  value,
  icon: Icon,
  tone = 'brand',
  hint,
  loading = false,
  onClick,
  className,
  ...props
}: StatCardProps) {
  const styles = TONE_STYLES[tone]
  const isInteractive = typeof onClick === 'function'

  const displayValue = typeof value === 'number' ? value.toLocaleString() : value

  return (
    <Card
      variant="glass"
      interactive={isInteractive}
      className={cn('glass-sheen relative overflow-hidden p-4', className)}
      onClick={onClick}
      role={isInteractive ? 'button' : undefined}
      tabIndex={isInteractive ? 0 : undefined}
      onKeyDown={
        isInteractive
          ? (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              onClick?.()
            }
          }
          : undefined
      }
      {...props}
    >
      {/* Corner glow — purely decorative. */}
      <div
        aria-hidden="true"
        className={cn(
          'pointer-events-none absolute -top-12 -right-12 h-28 w-28 rounded-full bg-gradient-to-br to-transparent blur-2xl',
          styles.glow
        )}
      />

      <div className="relative flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-muted-foreground truncate text-xs font-medium tracking-wide uppercase">
            {label}
          </p>

          {loading ? (
            <div
              className="bg-muted/50 mt-2 h-8 w-20 animate-pulse rounded-md"
              aria-hidden="true"
            />
          ) : (
            <p
              className={cn(
                'mt-1.5 text-2xl leading-none font-semibold tabular-nums',
                styles.value
              )}
            >
              {displayValue}
            </p>
          )}

          {hint && !loading && (
            <p className="text-muted-foreground/80 mt-1.5 truncate text-xs">{hint}</p>
          )}
        </div>

        {Icon && (
          <div
            className={cn(
              'flex size-9 shrink-0 items-center justify-center rounded-lg ring-1 ring-inset',
              styles.icon
            )}
          >
            <Icon className="size-4.5" aria-hidden="true" />
          </div>
        )}
      </div>
    </Card>
  )
})

export default StatCard
