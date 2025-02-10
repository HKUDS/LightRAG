import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { Checkbox } from '@/components/ui/Checkbox'
import Button from '@/components/ui/Button'
import { useState, useCallback } from 'react'
import { controlButtonVariant } from '@/lib/constants'
import { useSettingsStore } from '@/stores/settings'

import { SettingsIcon } from 'lucide-react'

import * as Api from '@/api/lightrag'

/**
 * Component that displays a checkbox with a label.
 */
const LabeledCheckBox = ({
  checked,
  onCheckedChange,
  label
}: {
  checked: boolean
  onCheckedChange: () => void
  label: string
}) => {
  return (
    <div className="flex gap-2">
      <Checkbox checked={checked} onCheckedChange={onCheckedChange} />
      <label
        htmlFor="terms"
        className="text-sm leading-none font-medium peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
      >
        {label}
      </label>
    </div>
  )
}

/**
 * Component that displays a popover with settings options.
 */
export default function Settings() {
  const [opened, setOpened] = useState<boolean>(false)

  const enableEdgeEvents = useSettingsStore.use.enableEdgeEvents()
  const enableNodeDrag = useSettingsStore.use.enableNodeDrag()
  const enableHideUnselectedEdges = useSettingsStore.use.enableHideUnselectedEdges()
  const showEdgeLabel = useSettingsStore.use.showEdgeLabel()

  const setEnableNodeDrag = useCallback(
    () => useSettingsStore.setState((pre) => ({ enableNodeDrag: !pre.enableNodeDrag })),
    []
  )
  const setEnableEdgeEvents = useCallback(
    () => useSettingsStore.setState((pre) => ({ enableEdgeEvents: !pre.enableEdgeEvents })),
    []
  )
  const setEnableHideUnselectedEdges = useCallback(
    () =>
      useSettingsStore.setState((pre) => ({
        enableHideUnselectedEdges: !pre.enableHideUnselectedEdges
      })),
    []
  )
  const setShowEdgeLabel = useCallback(
    () =>
      useSettingsStore.setState((pre) => ({
        showEdgeLabel: !pre.showEdgeLabel
      })),
    []
  )

  return (
    <Popover open={opened} onOpenChange={setOpened}>
      <PopoverTrigger asChild>
        <Button variant={controlButtonVariant} tooltip="Settings">
          <SettingsIcon />
        </Button>
      </PopoverTrigger>
      <PopoverContent side="right" align="start" className="p-2">
        <div className="flex flex-col gap-2">
          <LabeledCheckBox
            checked={enableNodeDrag}
            onCheckedChange={setEnableNodeDrag}
            label="Node Draggable"
          />
          <LabeledCheckBox
            checked={enableEdgeEvents}
            onCheckedChange={setEnableEdgeEvents}
            label="Edge Events"
          />
          <LabeledCheckBox
            checked={enableHideUnselectedEdges}
            onCheckedChange={setEnableHideUnselectedEdges}
            label="Hide Unselected Edges"
          />
          <LabeledCheckBox
            checked={showEdgeLabel}
            onCheckedChange={setShowEdgeLabel}
            label="Show Edge Label"
          />
          <Button
            onClick={async () => {
              console.log(Api.checkHealth())
            }}
          >
            Test Api
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  )
}
