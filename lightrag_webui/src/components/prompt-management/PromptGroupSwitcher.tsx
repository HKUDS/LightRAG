import Button from '@/components/ui/Button'
import { PromptConfigGroup } from '@/api/lightrag'
import { cn } from '@/lib/utils'

type PromptGroupSwitcherProps = {
  value: PromptConfigGroup
  onChange: (group: PromptConfigGroup) => void
}

export default function PromptGroupSwitcher({ value, onChange }: PromptGroupSwitcherProps) {
  return (
    <div className="flex gap-2">
      <Button
        type="button"
        variant="outline"
        className={cn(value === 'retrieval' && 'bg-emerald-400 text-zinc-50 hover:bg-emerald-500 hover:text-zinc-50')}
        onClick={() => onChange('retrieval')}
      >
        Retrieval
      </Button>
      <Button
        type="button"
        variant="outline"
        className={cn(value === 'indexing' && 'bg-emerald-400 text-zinc-50 hover:bg-emerald-500 hover:text-zinc-50')}
        onClick={() => onChange('indexing')}
      >
        Indexing
      </Button>
    </div>
  )
}
