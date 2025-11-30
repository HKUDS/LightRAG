import { useState, useMemo, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getTableList, getTableSchema, getTableData } from '@/api/lightrag'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/Dialog'
import DataTable from '@/components/ui/DataTable'
import { ColumnDef } from '@tanstack/react-table'
import Button from '@/components/ui/Button'
import { ChevronLeftIcon, ChevronRightIcon, RefreshCwIcon, CopyIcon, CheckIcon } from 'lucide-react'
import { toast } from 'sonner'

const HIDDEN_COLUMNS = ['meta']

// Truncate long values for display
function truncateValue(value: any, maxLength = 50): string {
  if (value === null || value === undefined) return ''

  let strValue: string
  if (typeof value === 'object') {
    strValue = JSON.stringify(value)
  } else {
    strValue = String(value)
  }

  if (strValue.length <= maxLength) return strValue
  return strValue.slice(0, maxLength) + '...'
}

// Format value for display in modal
function formatValue(value: any): string {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'

  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }

  return String(value)
}

// Check if value is JSON-like (object or array)
function isJsonLike(value: any): boolean {
  return typeof value === 'object' && value !== null
}

// Copy to clipboard helper
async function copyToClipboard(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // Fall through to legacy approach
    }
  }
  // Fallback for older browsers
  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  try {
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand('copy')
    return true
  } catch {
    return false
  } finally {
    if (textarea.parentNode) {
      document.body.removeChild(textarea)
    }
  }
}

// Copy button component with feedback
function CopyButton({ text, label }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    const success = await copyToClipboard(text)
    if (success) {
      setCopied(true)
      toast.success(label ? `${label} copied` : 'Copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    } else {
      toast.error('Failed to copy')
    }
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-6 w-6 p-0"
      onClick={handleCopy}
    >
      {copied ? (
        <CheckIcon className="h-3 w-3 text-green-500" />
      ) : (
        <CopyIcon className="h-3 w-3" />
      )}
    </Button>
  )
}

// Row Detail Modal
function RowDetailModal({
  row,
  open,
  onOpenChange
}: {
  row: Record<string, any> | null
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const entries = useMemo(() => (row ? Object.entries(row) : []), [row])
  const fullRowJson = useMemo(() => {
    try {
      return JSON.stringify(row, null, 2)
    } catch {
      return '[Unable to serialize row]'
    }
  }, [row])

  if (!row) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Row Details
            <CopyButton text={fullRowJson} label="Full row" />
          </DialogTitle>
          <DialogDescription>
            Click the copy icon next to any field to copy its value
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-auto space-y-3 pr-2">
          {entries.map(([key, value]) => (
            <div key={key} className="border rounded-lg p-3 bg-muted/30">
              <div className="flex items-center justify-between mb-1">
                <span className="font-medium text-sm text-muted-foreground">{key}</span>
                <CopyButton text={formatValue(value)} label={key} />
              </div>
              <div className={`text-sm ${isJsonLike(value) ? 'font-mono' : ''}`}>
                {isJsonLike(value) ? (
                  <pre className="whitespace-pre-wrap break-all bg-muted p-2 rounded text-xs overflow-auto max-h-[200px]">
                    {formatValue(value)}
                  </pre>
                ) : (
                  <div className="whitespace-pre-wrap break-all">
                    {formatValue(value)}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}

export default function TableExplorer() {
  const [selectedTable, setSelectedTable] = useState<string>('')
  const [page, setPage] = useState(1)
  const [selectedRow, setSelectedRow] = useState<Record<string, any> | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const pageSize = 20

  // Fetch table list
  const { data: tableList } = useQuery({
    queryKey: ['tables', 'list'],
    queryFn: getTableList,
  })

  // Derive effective selection: use state if set, otherwise default to first table
  const effectiveSelectedTable = selectedTable || (tableList?.[0] ?? '')

  // Reset page when table changes
  const handleTableChange = (value: string) => {
    setSelectedTable(value)
    setPage(1)
  }

  // Fetch schema
  const { data: schema } = useQuery({
    queryKey: ['tables', effectiveSelectedTable, 'schema'],
    queryFn: () => getTableSchema(effectiveSelectedTable),
    enabled: !!effectiveSelectedTable,
  })

  // Fetch data
  const { data: tableData, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['tables', effectiveSelectedTable, 'data', page],
    queryFn: () => getTableData(effectiveSelectedTable, page, pageSize),
    enabled: !!effectiveSelectedTable,
  })

  // Handle row click
  const handleRowClick = useCallback((row: Record<string, any>) => {
    setSelectedRow(row)
    setModalOpen(true)
  }, [])

  // Generate columns dynamically from data
  const columns = useMemo<ColumnDef<any>[]>(() => {
    const cols: ColumnDef<any>[] = []
    if (tableData?.data && tableData.data.length > 0) {
      const allKeys = new Set<string>()
      tableData.data.forEach((row: any) => {
        Object.keys(row).forEach(key => allKeys.add(key))
      })

      Array.from(allKeys).sort().forEach((key) => {
        if (HIDDEN_COLUMNS.includes(key)) return // Skip hidden columns
        cols.push({
          accessorKey: key,
          header: () => (
            <div className="font-semibold text-xs truncate max-w-[150px]" title={key}>
              {key}
            </div>
          ),
          cell: ({ row }) => {
            const value = row.getValue(key)
            const displayValue = truncateValue(value, 50)
            const isLong = typeof value === 'object' || (typeof value === 'string' && value.length > 50)

            return (
              <div
                className={`text-xs max-w-[200px] truncate ${isLong ? 'cursor-pointer hover:text-primary' : ''}`}
                title={isLong ? 'Click row to see full value' : displayValue}
              >
                {displayValue}
              </div>
            )
          },
        })
      })
    }
    return cols
  }, [tableData?.data])

  const totalPages = tableData?.total_pages || 0

  return (
    <div className="h-full flex flex-col p-4 gap-4 overflow-hidden">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-medium">Table Explorer</CardTitle>
            <div className="flex items-center gap-2">
              <Select value={effectiveSelectedTable} onValueChange={handleTableChange}>
                <SelectTrigger className="w-[250px]">
                  <SelectValue placeholder={tableList && tableList.length > 0 ? 'Select a table' : 'No tables available'} />
                </SelectTrigger>
                <SelectContent>
                  {tableList && tableList.length > 0 ? (
                    tableList.map((table) => (
                      <SelectItem key={table} value={table}>
                        {table}
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem value="no-tables" disabled>
                      No tables found
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
              <Button variant="outline" size="icon" onClick={() => refetch()}>
                <RefreshCwIcon className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        {schema && (
          <CardContent className="pb-2">
            <details className="text-xs text-muted-foreground cursor-pointer">
              <summary>Show Schema (DDL)</summary>
              <pre className="mt-2 p-2 bg-muted rounded overflow-auto max-h-[200px] font-mono text-xs">
                {schema.ddl}
              </pre>
            </details>
          </CardContent>
        )}
      </Card>

      <Card className="flex-1 overflow-hidden flex flex-col">
        <CardContent className="flex-1 p-0 overflow-auto">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <RefreshCwIcon className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : isError ? (
            <div className="flex flex-col items-center justify-center h-full text-destructive gap-2">
              <p className="font-medium">Failed to load table data</p>
              <p className="text-sm text-muted-foreground">{error instanceof Error ? error.message : 'Unknown error'}</p>
              <Button variant="outline" size="sm" onClick={() => refetch()} className="mt-2">
                Retry
              </Button>
            </div>
          ) : (
            <div className="h-full">
              <DataTable
                columns={columns}
                data={tableData?.data || []}
                onRowClick={handleRowClick}
              />
            </div>
          )}
        </CardContent>

        <div className="border-t p-2 flex items-center justify-between bg-muted/20">
          <div className="text-sm text-muted-foreground">
            {tableData?.total ? (
              <>
                Showing {((page - 1) * pageSize) + 1} to {Math.min(page * pageSize, tableData.total)} of {tableData.total} rows
              </>
            ) : (
              'No results'
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page <= 1 || isLoading}
            >
              <ChevronLeftIcon className="h-4 w-4 mr-1" />
              Previous
            </Button>
            <span className="text-sm font-medium min-w-[3rem] text-center">
              {page} / {totalPages || 1}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages || isLoading}
            >
              Next
              <ChevronRightIcon className="h-4 w-4 ml-1" />
            </Button>
          </div>
        </div>
      </Card>

      <RowDetailModal
        row={selectedRow}
        open={modalOpen}
        onOpenChange={setModalOpen}
      />
    </div>
  )
}
