import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'
import {
  BookOpenIcon,
  ClipboardCheckIcon,
  FileSearchIcon,
  HistoryIcon,
  LayoutDashboardIcon,
  ListTreeIcon,
  RefreshCwIcon,
  ShieldCheckIcon
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

type SectionItem = {
  id: KGMaintenanceSection
  label: string
  group: string
  icon: LucideIcon
}

const sections: SectionItem[] = [
  { id: 'overview', label: '审阅包概览', group: '知识库迭代', icon: LayoutDashboardIcon },
  { id: 'stage', label: '当前阶段', group: '知识库迭代', icon: HistoryIcon },
  { id: 'kb-summary', label: '当前 KB 摘要', group: '知识库迭代', icon: BookOpenIcon },
  { id: 'quality', label: '质量检查', group: '质量与快照', icon: ShieldCheckIcon },
  { id: 'snapshot', label: '快照审阅', group: '质量与快照', icon: FileSearchIcon },
  { id: 'approval', label: 'Proposal 审批', group: '人工审阅', icon: ClipboardCheckIcon },
  { id: 'backlog', label: '改进 backlog', group: '人工审阅', icon: ListTreeIcon },
  { id: 'memory', label: '决策记忆', group: '人工审阅', icon: BookOpenIcon },
  { id: 'llm-review', label: 'LLM 审阅材料', group: '辅助材料', icon: FileSearchIcon }
]

interface KGMaintenanceShellProps {
  activeSection: KGMaintenanceSection
  onSectionChange: (section: KGMaintenanceSection) => void
  workspaces: string[]
  selectedWorkspace: string | null
  onWorkspaceChange: (workspace: string) => void
  onRefresh: () => void
  onRunReview: () => void
  loading: boolean
  running: boolean
  error: string | null
  children: ReactNode
  inspector: ReactNode
}

export default function KGMaintenanceShell({
  activeSection,
  onSectionChange,
  workspaces,
  selectedWorkspace,
  onWorkspaceChange,
  onRefresh,
  onRunReview,
  loading,
  running,
  error,
  children,
  inspector
}: KGMaintenanceShellProps) {
  const workspaceOptions = Array.isArray(workspaces) ? workspaces : []
  const grouped = sections.reduce<Record<string, SectionItem[]>>((acc, section) => {
    acc[section.group] = [...(acc[section.group] || []), section]
    return acc
  }, {})

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
          <Button size="sm" onClick={onRunReview} disabled={!selectedWorkspace || running}>
            <ShieldCheckIcon className={cn('size-4', running && 'animate-pulse')} />
            {running ? '运行中' : '运行审阅包'}
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
          {running
            ? '正在生成 KB 审阅包。产物会进入人工审阅，不会自动修改 KG。'
            : '正在加载审阅包产物...'}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-0 overflow-hidden lg:grid-cols-[220px_minmax(0,1fr)_320px]">
        <aside className="border-border/70 bg-muted/20 min-h-0 overflow-auto border-b p-3 lg:border-r lg:border-b-0">
          {Object.entries(grouped).map(([group, items]) => (
            <div key={group} className="mb-4">
              <div className="text-muted-foreground mb-2 px-2 text-xs font-medium">{group}</div>
              <div className="space-y-1">
                {items.map((item) => {
                  const Icon = item.icon
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => onSectionChange(item.id)}
                      className={cn(
                        'flex h-9 w-full items-center gap-2 rounded-md px-2 text-left text-sm transition-colors',
                        activeSection === item.id
                          ? 'bg-emerald-500 text-white'
                          : 'hover:bg-accent text-foreground'
                      )}
                    >
                      <Icon className="size-4 shrink-0" />
                      <span className="truncate">{item.label}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          ))}
        </aside>

        <main className="min-h-0 overflow-auto p-4">{children}</main>

        <aside className="border-border/70 bg-muted/10 min-h-0 overflow-auto border-t p-4 lg:border-t-0 lg:border-l">
          {inspector}
        </aside>
      </div>
    </section>
  )
}
