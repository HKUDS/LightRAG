import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { cn } from '@/lib/utils'

type TextSize = 'xs' | 'sm' | 'base' | 'lg' | 'xl'
type TextWeight = 'normal' | 'medium' | 'semibold' | 'bold'

const sizeClasses: Record<TextSize, string> = {
  xs: 'text-xs',
  sm: 'text-sm',
  base: 'text-base',
  lg: 'text-lg',
  xl: 'text-xl',
}

const weightClasses: Record<TextWeight, string> = {
  normal: 'font-normal',
  medium: 'font-medium',
  semibold: 'font-semibold',
  bold: 'font-bold',
}

const Text = ({
  text,
  children,
  className,
  tooltipClassName,
  tooltip,
  side,
  size,
  weight,
  onClick,
}: {
  text?: string
  children?: React.ReactNode
  className?: string
  tooltipClassName?: string
  tooltip?: string
  side?: 'top' | 'right' | 'bottom' | 'left'
  size?: TextSize
  weight?: TextWeight
  onClick?: () => void
}) => {
  const content = text ?? children
  const combinedClassName = cn(
    size && sizeClasses[size],
    weight && weightClasses[weight],
    className,
    onClick !== undefined ? 'cursor-pointer' : undefined
  )

  // Use button for clickable elements, span for non-clickable
  const TextElement = onClick ? (
    <button
      type="button"
      className={cn(combinedClassName, 'bg-transparent border-none p-0 text-left')}
      onClick={onClick}
    >
      {content}
    </button>
  ) : (
    <span className={combinedClassName}>{content}</span>
  )

  if (!tooltip) {
    return TextElement
  }

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>{TextElement}</TooltipTrigger>
        <TooltipContent side={side} className={tooltipClassName}>
          {tooltip}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export default Text
