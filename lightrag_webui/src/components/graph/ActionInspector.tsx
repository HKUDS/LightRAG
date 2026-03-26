import { ReactNode, useMemo, useState } from 'react'
import { RawEdgeType, RawNodeType, useGraphStore } from '@/stores/graph'
import { useGraphWorkbenchStore } from '@/stores/graphWorkbench'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs'
import Button from '@/components/ui/Button'
import PropertiesView from './PropertiesView'
import CreateNodeForm from './CreateNodeForm'
import CreateRelationForm from './CreateRelationForm'
import DeleteGraphObjectPanel from './DeleteGraphObjectPanel'
import MergeEntityPanel from './MergeEntityPanel'

export const ACTION_INSPECTOR_TABS = ['inspect', 'create', 'delete', 'merge'] as const

export type ActionInspectorTab = (typeof ACTION_INSPECTOR_TABS)[number]

export type ActionInspectorNode = RawNodeType & {
  revision_token?: string
}

export type ActionInspectorEdge = RawEdgeType & {
  revision_token?: string
  sourceNode?: RawNodeType
  targetNode?: RawNodeType
}

export type ActionInspectorSelection =
  | { kind: 'node'; node: ActionInspectorNode }
  | { kind: 'edge'; edge: ActionInspectorEdge }

type ActionInspectorProps = {
  initialTab?: ActionInspectorTab
  selection?: ActionInspectorSelection | null
  inspectPane?: ReactNode
}

const isActionInspectorTab = (value: string): value is ActionInspectorTab =>
  ACTION_INSPECTOR_TABS.some((tab) => tab === value)

export const resolveActionInspectorTab = (
  current: ActionInspectorTab,
  next: string
): ActionInspectorTab => {
  return isActionInspectorTab(next) ? next : current
}

const resolveSelectionFromGraph = ({
  selectedNode,
  focusedNode,
  selectedEdge,
  focusedEdge,
  getNode,
  getEdge
}: {
  selectedNode: string | null
  focusedNode: string | null
  selectedEdge: string | null
  focusedEdge: string | null
  getNode: (id: string) => RawNodeType | null
  getEdge: (id: string, dynamicId?: boolean) => RawEdgeType | null
}): ActionInspectorSelection | null => {
  if (focusedNode || selectedNode) {
    const node = getNode(focusedNode ?? selectedNode ?? '')
    if (node) {
      return {
        kind: 'node',
        node: node as ActionInspectorNode
      }
    }
  }

  if (focusedEdge || selectedEdge) {
    const edge = getEdge(focusedEdge ?? selectedEdge ?? '', true)
    if (edge) {
      return {
        kind: 'edge',
        edge: {
          ...(edge as ActionInspectorEdge),
          sourceNode: getNode(edge.source) ?? undefined,
          targetNode: getNode(edge.target) ?? undefined
        }
      }
    }
  }

  return null
}

const ActionInspector = ({
  initialTab = 'inspect',
  selection,
  inspectPane
}: ActionInspectorProps) => {
  const [activeTab, setActiveTab] = useState<ActionInspectorTab>(initialTab)
  const selectedNode = useGraphStore.use.selectedNode()
  const focusedNode = useGraphStore.use.focusedNode()
  const selectedEdge = useGraphStore.use.selectedEdge()
  const focusedEdge = useGraphStore.use.focusedEdge()
  const graphDataVersion = useGraphStore.use.graphDataVersion()
  const rawGraph = useGraphStore.use.rawGraph()
  const mutationError = useGraphWorkbenchStore.use.mutationError()
  const conflictError = useGraphWorkbenchStore.use.conflictError()
  const clearMutationError = useGraphWorkbenchStore.use.clearMutationError()
  const getNode = (id: string) => rawGraph?.getNode(id) || null
  const getEdge = (id: string, dynamicId: boolean = true) =>
    rawGraph?.getEdge(id, dynamicId) || null

  const currentSelection = useMemo(() => {
    if (selection !== undefined) {
      return selection
    }

    return resolveSelectionFromGraph({
      selectedNode,
      focusedNode,
      selectedEdge,
      focusedEdge,
      getNode,
      getEdge
    })
  }, [
    selection,
    selectedNode,
    focusedNode,
    selectedEdge,
    focusedEdge,
    graphDataVersion,
    rawGraph,
    getNode,
    getEdge
  ])

  const errorText = conflictError ?? mutationError
  const errorClassName = conflictError
    ? 'border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-300'
    : 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300'

  return (
    <div className="bg-background/80 h-full rounded-xl border backdrop-blur-sm">
      <div className="flex h-full flex-col gap-3 p-3">
        <div>
          <h2 className="text-sm font-semibold tracking-wide uppercase">Action Inspector</h2>
          <p className="text-muted-foreground mt-1 text-xs">
            Inspect, mutate, and validate graph objects without leaving this workbench.
          </p>
        </div>

        {errorText && (
          <div className={`rounded-md border px-3 py-2 text-xs ${errorClassName}`}>
            <div className="flex items-start justify-between gap-2">
              <p>{errorText}</p>
              <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={clearMutationError}>
                Clear
              </Button>
            </div>
          </div>
        )}

        <Tabs
          value={activeTab}
          onValueChange={(next) => setActiveTab((current) => resolveActionInspectorTab(current, next))}
          className="min-h-0 flex-1"
        >
          <TabsList className="grid h-auto grid-cols-4">
            <TabsTrigger value="inspect" className="text-xs">
              Inspect
            </TabsTrigger>
            <TabsTrigger value="create" className="text-xs">
              Create
            </TabsTrigger>
            <TabsTrigger value="delete" className="text-xs">
              Delete
            </TabsTrigger>
            <TabsTrigger value="merge" className="text-xs">
              Merge
            </TabsTrigger>
          </TabsList>

          <TabsContent value="inspect" className="mt-3 min-h-0">
            {inspectPane ?? <PropertiesView panelClassName="max-w-none rounded-lg border p-2 text-xs" />}
          </TabsContent>

          <TabsContent value="create" className="mt-3 min-h-0">
            <div className="space-y-3">
              <CreateNodeForm />
              <CreateRelationForm selection={currentSelection} />
            </div>
          </TabsContent>

          <TabsContent value="delete" className="mt-3 min-h-0">
            <DeleteGraphObjectPanel selection={currentSelection} />
          </TabsContent>

          <TabsContent value="merge" className="mt-3 min-h-0">
            <MergeEntityPanel selection={currentSelection} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export { ActionInspector }
export default ActionInspector
