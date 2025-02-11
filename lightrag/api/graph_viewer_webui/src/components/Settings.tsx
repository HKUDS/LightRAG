import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { Checkbox } from '@/components/ui/Checkbox'
import Button from '@/components/ui/Button'
import Separator from '@/components/ui/Separator'
import Input from '@/components/ui/Input'
import { useState, useCallback, useEffect } from 'react'
import { controlButtonVariant } from '@/lib/constants'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'

import { SettingsIcon } from 'lucide-react'

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
    <div className="flex items-center gap-2">
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
  const [tempApiKey, setTempApiKey] = useState<string>('') // 用于临时存储输入的API Key

  const showPropertyPanel = useSettingsStore.use.showPropertyPanel()
  const showNodeSearchBar = useSettingsStore.use.showNodeSearchBar()
  const showNodeLabel = useSettingsStore.use.showNodeLabel()

  const enableEdgeEvents = useSettingsStore.use.enableEdgeEvents()
  const enableNodeDrag = useSettingsStore.use.enableNodeDrag()
  const enableHideUnselectedEdges = useSettingsStore.use.enableHideUnselectedEdges()
  const showEdgeLabel = useSettingsStore.use.showEdgeLabel()

  const enableHealthCheck = useSettingsStore.use.enableHealthCheck()
  const apiKey = useSettingsStore.use.apiKey()

  useEffect(() => {
    setTempApiKey(apiKey || '')
  }, [apiKey, opened])

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

  //
  const setShowPropertyPanel = useCallback(
    () => useSettingsStore.setState((pre) => ({ showPropertyPanel: !pre.showPropertyPanel })),
    []
  )

  const setShowNodeSearchBar = useCallback(
    () => useSettingsStore.setState((pre) => ({ showNodeSearchBar: !pre.showNodeSearchBar })),
    []
  )

  const setShowNodeLabel = useCallback(
    () => useSettingsStore.setState((pre) => ({ showNodeLabel: !pre.showNodeLabel })),
    []
  )

  const setEnableHealthCheck = useCallback(
    () => useSettingsStore.setState((pre) => ({ enableHealthCheck: !pre.enableHealthCheck })),
    []
  )

  const setApiKey = useCallback(async () => {
    useSettingsStore.setState({ apiKey: tempApiKey || null })
    await useBackendState.getState().check()
    setOpened(false)
  }, [tempApiKey])

  const handleTempApiKeyChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setTempApiKey(e.target.value)
    },
    [setTempApiKey]
  )

  return (
    <Popover open={opened} onOpenChange={setOpened}>
      <PopoverTrigger asChild>
        <Button variant={controlButtonVariant} tooltip="Settings" size="icon">
          <SettingsIcon />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="right"
        align="start"
        className="mb-2 p-2"
        onCloseAutoFocus={(e) => e.preventDefault()}
      >
        <div className="flex flex-col gap-2">
          <LabeledCheckBox
            checked={showPropertyPanel}
            onCheckedChange={setShowPropertyPanel}
            label="Show Property Panel"
          />
          <LabeledCheckBox
            checked={showNodeSearchBar}
            onCheckedChange={setShowNodeSearchBar}
            label="Show Search Bar"
          />

          <Separator />

          <LabeledCheckBox
            checked={showNodeLabel}
            onCheckedChange={setShowNodeLabel}
            label="Show Node Label"
          />
          <LabeledCheckBox
            checked={enableNodeDrag}
            onCheckedChange={setEnableNodeDrag}
            label="Node Draggable"
          />

          <Separator />

          <LabeledCheckBox
            checked={showEdgeLabel}
            onCheckedChange={setShowEdgeLabel}
            label="Show Edge Label"
          />
          <LabeledCheckBox
            checked={enableHideUnselectedEdges}
            onCheckedChange={setEnableHideUnselectedEdges}
            label="Hide Unselected Edges"
          />
          <LabeledCheckBox
            checked={enableEdgeEvents}
            onCheckedChange={setEnableEdgeEvents}
            label="Edge Events"
          />

          <Separator />

          <LabeledCheckBox
            checked={enableHealthCheck}
            onCheckedChange={setEnableHealthCheck}
            label="Health Check"
          />

          <Separator />

          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">API Key</label>
            <form className="flex h-6 gap-2" onSubmit={(e) => e.preventDefault()}>
              <div className="w-0 flex-1">
                <Input
                  type="password"
                  value={tempApiKey}
                  onChange={handleTempApiKeyChange}
                  placeholder="Enter your API key"
                  className="max-h-full w-full min-w-0"
                  autoComplete="off"
                />
              </div>
              <Button
                onClick={setApiKey}
                variant="outline"
                size="sm"
                className="max-h-full shrink-0"
              >
                Save
              </Button>
            </form>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}
