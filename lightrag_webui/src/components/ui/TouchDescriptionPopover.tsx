import { InfoIcon } from 'lucide-react'
import Button from '@/components/ui/Button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'

interface TouchDescriptionPopoverProps {
  description: string | null
}

export default function TouchDescriptionPopover({ description }: TouchDescriptionPopoverProps) {
  if (!description) return null

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="hidden shrink-0 [@media(any-hover:none)]:inline-flex"
          aria-label={description}
        >
          <InfoIcon aria-hidden="true" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="bottom"
        align="start"
        collisionPadding={16}
        avoidCollisions={true}
        className="max-h-[60vh] max-w-[calc(100vw-2rem)] overflow-y-auto whitespace-pre-wrap break-words p-3 text-sm"
      >
        {description}
      </PopoverContent>
    </Popover>
  )
}
