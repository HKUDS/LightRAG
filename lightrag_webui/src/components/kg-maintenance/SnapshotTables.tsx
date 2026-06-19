import { useEffect, useMemo, useState, type ChangeEvent } from 'react'
import { cn } from '@/lib/utils'
import { buildEvidenceIssueRows } from './kgMaintenanceDisplay'

type SnapshotTablesProps = {
  snapshot: Record<string, any> | null
}

type TableKey = 'nodes' | 'relations' | 'evidence'

type TableColumn = {
  key: string
  label: string
}

const tabs: Array<{ key: TableKey; label: string }> = [
  { key: 'nodes', label: '节点' },
  { key: 'relations', label: '关系' },
  { key: 'evidence', label: '证据问题' }
]

const nodeColumns: TableColumn[] = [
  { key: 'id', label: 'ID' },
  { key: 'label', label: '标签' },
  { key: 'entity_type', label: '实体类型' },
  { key: 'source_id', label: '证据来源 ID' },
  { key: 'file_path', label: '来源文件' }
]

const relationColumns: TableColumn[] = [
  { key: 'id', label: 'ID' },
  { key: 'source', label: '源节点' },
  { key: 'target', label: '目标节点' },
  { key: 'keywords', label: '关键词' },
  { key: 'source_id', label: '证据来源 ID' },
  { key: 'file_path', label: '来源文件' }
]

const evidenceColumns: TableColumn[] = [
  { key: 'itemType', label: '类型' },
  { key: 'itemId', label: '项目 ID' },
  { key: 'issue', label: '问题' }
]

const DEFAULT_VISIBLE_LIMIT = 100

export function SnapshotTables({ snapshot }: SnapshotTablesProps) {
  const [activeTab, setActiveTab] = useState<TableKey>('nodes')
  const [query, setQuery] = useState('')
  const [visibleLimit, setVisibleLimit] = useState(DEFAULT_VISIBLE_LIMIT)

  const nodes = useMemo(() => normalizeCollection(snapshot?.nodes ?? snapshot?.entities), [snapshot])
  const relations = useMemo(
    () =>
      normalizeCollection(snapshot?.edges ?? snapshot?.relations ?? snapshot?.links).map(
        (relation, index) => ({
          ...relation,
          id: relationId(relation, index)
        })
      ),
    [snapshot]
  )
  const evidenceRows = useMemo(() => buildEvidenceIssueRows(snapshot), [snapshot])

  const table = useMemo(() => {
    if (activeTab === 'relations') {
      return { columns: relationColumns, rows: relations }
    }
    if (activeTab === 'evidence') {
      return { columns: evidenceColumns, rows: evidenceRows }
    }
    return { columns: nodeColumns, rows: nodes }
  }, [activeTab, evidenceRows, nodes, relations])

  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    if (!normalizedQuery) return table.rows

    return table.rows.filter((row) =>
      table.columns.some((column) =>
        formatCellValue(row[column.key]).toLowerCase().includes(normalizedQuery)
      ) || safePrettyJson(row).toLowerCase().includes(normalizedQuery)
    )
  }, [query, table])
  const visibleRows = filteredRows.slice(0, visibleLimit)
  const hasMoreRows = visibleRows.length < filteredRows.length
  const emptyMessage = table.rows.length === 0 ? '暂无数据' : '没有匹配结果'

  useEffect(() => {
    setVisibleLimit(DEFAULT_VISIBLE_LIMIT)
  }, [activeTab, query, snapshot])

  const handleTabClick = (tab: TableKey) => {
    setActiveTab(tab)
    setVisibleLimit(DEFAULT_VISIBLE_LIMIT)
  }

  const handleQueryChange = (event: ChangeEvent<HTMLInputElement>) => {
    setQuery(event.target.value)
    setVisibleLimit(DEFAULT_VISIBLE_LIMIT)
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="bg-muted flex rounded-md p-1">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => handleTabClick(tab.key)}
              className={cn(
                'rounded-sm px-3 py-1.5 text-sm transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none',
                activeTab === tab.key
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <input
          type="search"
          aria-label="搜索"
          placeholder="搜索"
          value={query}
          onChange={handleQueryChange}
          className="border-input bg-background focus-visible:ring-ring h-9 w-full rounded-md border px-3 text-sm outline-none transition-shadow placeholder:text-muted-foreground focus-visible:ring-2 sm:w-64"
        />
      </div>

      <div className="text-muted-foreground flex flex-wrap items-center justify-between gap-2 text-xs">
        <span>
          显示 {visibleRows.length} / {filteredRows.length} 行
        </span>
        {hasMoreRows ? (
          <button
            type="button"
            onClick={() => setVisibleLimit((current) => current + DEFAULT_VISIBLE_LIMIT)}
            className="border-border bg-background hover:bg-muted/40 focus-visible:ring-ring rounded-md border px-2.5 py-1 text-xs text-foreground transition-colors focus-visible:ring-2 focus-visible:outline-none"
          >
            显示更多
          </button>
        ) : null}
      </div>

      <div className="border-border/70 h-[520px] overflow-auto rounded-md border">
        <table className="w-full min-w-[880px] border-collapse text-sm">
          <thead className="bg-muted/50 sticky top-0 z-10">
            <tr>
              {table.columns.map((column) => (
                <th
                  key={column.key}
                  scope="col"
                  data-column={column.key}
                  className="border-border/70 border-b px-3 py-2 text-left font-medium whitespace-nowrap"
                >
                  {column.label}
                </th>
              ))}
              <th
                scope="col"
                data-column="details"
                className="border-border/70 border-b px-3 py-2 text-left font-medium whitespace-nowrap"
              >
                详情
              </th>
            </tr>
          </thead>
          <tbody>
            {visibleRows.length > 0 ? (
              visibleRows.map((row, rowIndex) => (
                <tr
                  key={rowKey(activeTab, row, rowIndex)}
                  className="border-border/60 border-b"
                >
                  {table.columns.map((column) => (
                    <td
                      key={column.key}
                      className="text-muted-foreground max-w-[280px] px-3 py-2 align-top break-words"
                    >
                      {formatCellValue(row[column.key])}
                    </td>
                  ))}
                  <td className="text-muted-foreground px-3 py-2 align-top">
                    <details className="max-w-[360px]">
                      <summary className="cursor-pointer text-foreground">查看</summary>
                      <pre className="border-border/70 bg-muted/20 mt-2 max-h-56 overflow-auto rounded-md border p-2 text-xs break-words whitespace-pre-wrap">
                        {safePrettyJson(row)}
                      </pre>
                    </details>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td
                  colSpan={table.columns.length + 1}
                  className="text-muted-foreground px-3 py-8 text-center text-sm"
                >
                  {emptyMessage}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function normalizeCollection(value: unknown): Array<Record<string, any>> {
  if (Array.isArray(value)) {
    return value.filter(isRecord)
  }

  if (!isRecord(value)) return []

  return Object.entries(value).flatMap(([id, item]) => {
    if (!isRecord(item)) return []
    return item.id === undefined || item.id === null || item.id === '' ? [{ ...item, id }] : [item]
  })
}

function relationId(relation: Record<string, any>, index: number): string {
  if (!isEmpty(relation.id)) return formatCellValue(relation.id)

  const source = relation.source ?? relation.from ?? relation.src
  const target = relation.target ?? relation.to ?? relation.dst
  if (!isEmpty(source) && !isEmpty(target)) {
    return `${formatCellValue(source)}->${formatCellValue(target)}`
  }

  return `relation-${index + 1}`
}

function formatCellValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (Array.isArray(value)) return value.map(formatCellValue).join(', ')
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  return safeStringify(value)
}

function rowKey(tableKey: TableKey, row: Record<string, any>, index: number): string {
  if (tableKey === 'evidence') {
    return `${tableKey}:${formatCellValue(row.id ?? row.itemId ?? index)}`
  }

  if (tableKey === 'relations') {
    return `${tableKey}:${relationId(row, index)}`
  }

  return `${tableKey}:${formatCellValue(row.id ?? row.label ?? row.name ?? index)}`
}

function safePrettyJson(value: unknown): string {
  return safeStringify(value, 2)
}

function safeStringify(value: unknown, space?: number): string {
  try {
    const result = JSON.stringify(value, null, space)
    if (result) return result
  } catch {
    return '无法序列化'
  }

  try {
    return String(value)
  } catch {
    return '无法序列化'
  }
}

function isEmpty(value: unknown): boolean {
  if (value === null || value === undefined) return true
  if (typeof value === 'string') return value.trim().length === 0
  return false
}

function isRecord(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
