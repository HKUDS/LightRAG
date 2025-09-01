import { useState, useEffect } from 'react'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/Popover'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
} from '@/components/ui/Command'
import Checkbox from '@/components/ui/Checkbox'
import { getAllLabels, LabelType } from '@/api/lightrag'
import { TagIcon, XIcon, PlusIcon, CheckIcon } from 'lucide-react'
import { errorMessage } from '@/lib/utils'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'

interface LabelSelectorProps {
  selectedLabels: string[]
  onLabelsChange: (labels: string[]) => void
  placeholder?: string
  className?: string
  disabled?: boolean
  multiple?: boolean
  showMatchAllOption?: boolean
  matchAll?: boolean
  onMatchAllChange?: (matchAll: boolean) => void
}

export default function LabelSelector({
  selectedLabels,
  onLabelsChange,
  placeholder = "Select labels...",
  className,
  disabled = false,
  multiple = true,
  showMatchAllOption = false,
  matchAll = false,
  onMatchAllChange
}: LabelSelectorProps) {
  const [open, setOpen] = useState(false)
  const [labels, setLabels] = useState<Record<string, LabelType>>({})
  const [loading, setLoading] = useState(false)
  const [searchValue, setSearchValue] = useState("")

  const loadLabels = async () => {
    try {
      setLoading(true)
      const labelsData = await getAllLabels()
      setLabels(labelsData)
    } catch (error) {
      console.error('Error loading labels:', error)
      toast.error(`Failed to load labels: ${errorMessage(error)}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (open && Object.keys(labels).length === 0) {
      loadLabels()
    }
  }, [open, labels])

  const labelList = Object.values(labels)
  const filteredLabels = labelList.filter(label =>
    label.name.toLowerCase().includes(searchValue.toLowerCase()) ||
    label.description.toLowerCase().includes(searchValue.toLowerCase())
  )

  const handleLabelToggle = (labelName: string) => {
    if (multiple) {
      const newLabels = selectedLabels.includes(labelName)
        ? selectedLabels.filter(l => l !== labelName)
        : [...selectedLabels, labelName]
      onLabelsChange(newLabels)
    } else {
      onLabelsChange(selectedLabels.includes(labelName) ? [] : [labelName])
    }
  }

  const handleRemoveLabel = (labelName: string) => {
    onLabelsChange(selectedLabels.filter(l => l !== labelName))
  }

  const selectedLabelObjects = selectedLabels.map(name => labels[name]).filter(Boolean)

  return (
    <div className={cn("space-y-2", className)}>
      {/* Match All Option */}
      {showMatchAllOption && selectedLabels.length > 1 && (
        <div className="flex items-center space-x-2">
          <Checkbox
            id="match-all"
            checked={matchAll}
            onCheckedChange={onMatchAllChange}
            disabled={disabled}
          />
          <label htmlFor="match-all" className="text-sm">
            Documents must have ALL selected labels
          </label>
        </div>
      )}

      {/* Label Selector */}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between"
            disabled={disabled}
          >
            <div className="flex items-center gap-1">
              <TagIcon className="h-4 w-4" />
              {selectedLabels.length === 0 
                ? placeholder
                : `${selectedLabels.length} label${selectedLabels.length > 1 ? 's' : ''} selected`
              }
            </div>
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-full p-0" align="start">
          <Command>
            <CommandInput
              placeholder="Search labels..."
              value={searchValue}
              onValueChange={setSearchValue}
            />
            <CommandEmpty>
              {loading ? "Loading labels..." : "No labels found."}
            </CommandEmpty>
            <CommandGroup className="max-h-64 overflow-auto">
              {filteredLabels.map((label) => {
                const isSelected = selectedLabels.includes(label.name)
                return (
                  <CommandItem
                    key={label.name}
                    value={label.name}
                    onSelect={() => handleLabelToggle(label.name)}
                    className="flex items-center gap-2"
                  >
                    <div
                      className="w-3 h-3 rounded-full border flex-shrink-0"
                      style={{ backgroundColor: label.color }}
                    />
                    <span className="flex-1 truncate">{label.name}</span>
                    {label.description && (
                      <span className="text-xs text-muted-foreground truncate max-w-32">
                        {label.description}
                      </span>
                    )}
                    {isSelected && <CheckIcon className="h-4 w-4 text-primary flex-shrink-0" />}
                  </CommandItem>
                )
              })}
            </CommandGroup>
          </Command>
        </PopoverContent>
      </Popover>

      {/* Selected Labels Display */}
      {selectedLabels.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selectedLabelObjects.map((label) => (
            <Badge
              key={label.name}
              variant="secondary"
              className="flex items-center gap-1 pl-2"
              style={{ borderLeftColor: label.color, borderLeftWidth: '3px' }}
            >
              {label.name}
              {!disabled && (
                <XIcon
                  className="h-3 w-3 cursor-pointer hover:text-destructive"
                  onClick={() => handleRemoveLabel(label.name)}
                />
              )}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}