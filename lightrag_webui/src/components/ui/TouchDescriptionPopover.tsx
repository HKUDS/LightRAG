import { InfoIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'

interface TouchDescriptionPopoverProps {
  description: string | null
}

export default function TouchDescriptionPopover({ description }: TouchDescriptionPopoverProps) {
  const { t } = useTranslation()

  if (!description) return null

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="relative hidden h-9 w-9 shrink-0 after:absolute after:-inset-1 after:content-[''] [@media(any-hover:none)]:inline-flex"
          aria-label={t('header.deploymentInfo')}
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
