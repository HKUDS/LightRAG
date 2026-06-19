import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'
import {
  ArchiveIcon,
  ClipboardCheckIcon,
  FileStackIcon,
  ListChecksIcon,
  RefreshCwIcon,
  SearchCheckIcon,
  SparklesIcon
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

type SectionItem = {
  id: KGMaintenanceSection
  label: string
  icon: LucideIcon
}

const sections: SectionItem[] = [
  { id: 'check', label: '检查知识库', icon: SearchCheckIcon },
  { id: 'llm-review', label: 'LLM 审阅', icon: SparklesIcon },
  { id: 'approval', label: 'Proposal 审批', icon: ClipboardCheckIcon },
  { id: 'execute', label: '执行变更', icon: ListChecksIcon },
  { id: 'validate', label: '验证结果', icon: ArchiveIcon }
]

interface KGMaintenanceShellProps {
  activeSection: KGMaintenanceSection
  onSectionChange: (section: KGMaintenanceSection) => void
  workspaces: string[]
  selectedWorkspace: string | null
  onWorkspaceChange: (workspace: string) => void
  onRefresh: () => void
  onOpenArtifacts?: () => void
  loading: boolean
  running: boolean
  error: string | null
  children: ReactNode
}

export default function KGMaintenanceShell({
  activeSection,
  onSectionChange,
  workspaces,
  selectedWorkspace,
  onWorkspaceChange,
  onRefresh,
  onOpenArtifacts,
  loading,
  running,
  error,
  children
}: KGMaintenanceShellProps) {
  const workspaceOptions = Array.isArray(workspaces) ? workspaces : []

  return (
    <section className="bg-background flex h-full min-h-0 flex-col">
      <div className="border-border/70 flex min-h-14 shrink-0 flex-wrap items-center justify-between gap-2 border-b px-4 py-2">
        <div className="min-w-0">
          <h1 className="text-base font-semibold">知识库迭代 Agent</h1>
          <p className="text-muted-foreground truncate text-xs">
            {selectedWorkspace || '未选择 workspace'}
          </p>
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <select
            value={selectedWorkspace || ''}
            onChange={(event) => onWorkspaceChange(event.target.value)}
            className="border-input bg-background h-9 min-w-0 rounded-md border px-3 text-sm sm:min-w-52"
            aria-label="workspace"
          >
            {workspaceOptions.length === 0 && <option value="">未选择 workspace</option>}
            {workspaceOptions.map((workspace) => (
              <option key={workspace} value={workspace}>
                {workspace}
              </option>
            ))}
          </select>
          <Button variant="outline" size="sm" onClick={onRefresh} disabled={loading}>
            <RefreshCwIcon className={cn('size-4', loading && 'animate-spin')} />
            刷新
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onOpenArtifacts}
            disabled={!selectedWorkspace}
          >
            <FileStackIcon className="size-4" />
            全部产物
          </Button>
        </div>
      </div>

      {error && (
        <div className="border-destructive/30 bg-destructive/10 text-destructive mx-4 mt-3 flex flex-wrap items-center justify-between gap-2 rounded-md border px-3 py-2 text-sm">
          <span>{error}</span>
          <button type="button" className="underline" onClick={onRefresh}>
            重试 / 查看服务日志
          </button>
        </div>
      )}

      {(running || loading) && !error && (
        <div className="border-border/70 bg-muted/30 mx-4 mt-3 rounded-md border px-3 py-2 text-sm">
          {running ? '正在处理知识库维护任务。' : '正在加载维护产物...'}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-0 overflow-hidden lg:grid-cols-[240px_minmax(0,1fr)]">
        <aside className="border-border/70 bg-muted/20 min-h-0 overflow-auto border-b p-3 lg:border-r lg:border-b-0">
          <div className="space-y-1">
            {sections.map((item, index) => {
              const Icon = item.icon
              const active = activeSection === item.id
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSectionChange(item.id)}
                  className={cn(
                    'flex h-10 w-full items-center gap-2 rounded-md px-2 text-left text-sm transition-colors',
                    active ? 'bg-emerald-500 text-white' : 'hover:bg-accent text-foreground'
                  )}
                >
                  <span
                    className={cn(
                      'flex size-5 shrink-0 items-center justify-center rounded text-[11px] font-medium',
                      active ? 'bg-white/20 text-white' : 'bg-muted text-muted-foreground'
                    )}
                  >
                    {index + 1}
                  </span>
                  <Icon className="size-4 shrink-0" />
                  <span className="truncate">{item.label}</span>
                </button>
              )
            })}
          </div>
        </aside>

        <main className="min-h-0 overflow-auto p-4">{children}</main>
      </div>
    </section>
  )
}
