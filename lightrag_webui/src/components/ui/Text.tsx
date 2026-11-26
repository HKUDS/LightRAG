import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { cn } from '@/lib/utils'

const Text = ({
  text,
  className,
  tooltipClassName,
  tooltip,
  side,
  onClick
}: {
  text: string
  className?: string
  tooltipClassName?: string
  tooltip?: string
  side?: 'top' | 'right' | 'bottom' | 'left'
  onClick?: () => void
}) => {
  const trigger =
    onClick !== undefined ? (
      <button
        type="button"
        className={cn('appearance-none border-0 bg-transparent p-0 text-left [font:inherit] cursor-pointer', className)}
        onClick={onClick}
      >
        {text}
      </button>
    ) : (
      <label className={className} tabIndex={tooltip ? 0 : undefined}>
        {text}
      </label>
    )

  if (!tooltip) {
    return trigger
  }

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>{trigger}</TooltipTrigger>
        <TooltipContent side={side} className={tooltipClassName}>
          {tooltip}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export default Text
