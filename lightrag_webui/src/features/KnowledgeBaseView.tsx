import * as React from 'react'
import { useTranslation } from 'react-i18next'
import {
  ArrowLeft,
  BookOpen,
  CheckCircle2,
  Clock,
  Copy,
  FileText,
  Loader2,
  LoaderCircle,
  Trash2,
  XCircle,
} from 'lucide-react'
import { toast } from 'sonner'

import { cn } from '@/lib/utils'
import KnowledgeBaseDocuments from '@/features/KnowledgeBaseDocuments'
import { Card } from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger
} from '@/components/ui/Tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import Checkbox from '@/components/ui/Checkbox'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import FileUploader from '@/components/ui/FileUploader'
import {
  getKnowledgeBaseTaskHistory,
  uploadKnowledgeBaseDocument,
  deleteKnowledgeBase,
  deleteKnowledgeBaseDocuments,
  getKnowledgeBase,
  getSupportedFileTypes,
  type KnowledgeBaseTaskItem
} from '@/api/lightrag'

interface KnowledgeBaseViewProps {
  kbId: string
  onBack: () => void
}

// Fallback parser choices when the backend capability matrix cannot be loaded.
const FALLBACK_PARSERS = ['auto', 'docling', 'mineru', 'legacy']

function statusTone(status: string): 'success' | 'warning' | 'danger' | 'info' | 'neutral' {
  const s = status.toLowerCase()
  if (s.includes('complet') || s.includes('succeed') || s === 'processed')
    return 'success'
  if (s.includes('fail') || s.includes('error')) return 'danger'
  if (s.includes('process') || s.includes('pending') || s.includes('parsing'))
    return 'warning'
  return 'neutral'
}

function StatusBadge({ status }: { status: string }) {
  const tone = statusTone(status)
  const map = {
    success: 'bg-emerald-500/12 text-emerald-400 ring-emerald-500/22',
    warning: 'bg-amber-500/12 text-amber-400 ring-amber-500/22',
    danger: 'bg-rose-500/12 text-rose-400 ring-rose-500/22',
    info: 'bg-blue-500/12 text-blue-400 ring-blue-500/22',
    neutral: 'bg-slate-500/12 text-slate-300 ring-slate-500/22'
  } as const
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset',
        map[tone]
      )}
    >
      {tone === 'success' && <CheckCircle2 className="size-3" />}
      {tone === 'warning' && <LoaderCircle className="size-3" />}
      {tone === 'danger' && <XCircle className="size-3" />}
      {tone === 'info' && <Clock className="size-3" />}
      {tone === 'neutral' && <FileText className="size-3" />}
      {status}
    </span>
  )
}

/**
 * Single knowledge base detail view. Mirrors the two-tab design from the
 * reference screenshots: a document browser and an ingestion panel.
 */
export default function KnowledgeBaseView({
  kbId,
  onBack
}: KnowledgeBaseViewProps) {
  const { t } = useTranslation()
  const [tabValue, setTabValue] = React.useState('documents')
  const [kbName, setKbName] = React.useState<string | null>(null)

  // Load the KB detail once so the header can show the display name
  // (defaults to the workspace id when no custom name exists).
  React.useEffect(() => {
    let cancelled = false
    getKnowledgeBase(kbId)
      .then((detail) => {
        if (!cancelled) setKbName(detail.name || kbId)
      })
      .catch(() => {
        // Keep the fallback (kbId) on transient failures.
      })
    return () => {
      cancelled = true
    }
  }, [kbId])

  const [history, setHistory] = React.useState<KnowledgeBaseTaskItem[]>([])
  const [loadingHistory, setLoadingHistory] = React.useState(false)

  const [confirmDelete, setConfirmDelete] = React.useState(false)
  const [deleting, setDeleting] = React.useState(false)

  const [historyDeleteTarget, setHistoryDeleteTarget] =
    React.useState<KnowledgeBaseTaskItem | null>(null)
  const [deletingHistory, setDeletingHistory] = React.useState(false)

  const loadHistory = React.useCallback(async () => {
    setLoadingHistory(true)
    try {
      const h = await getKnowledgeBaseTaskHistory(kbId, 20)
      setHistory(h ?? [])
    } catch (err) {
      toast.error(errorMessage(err) || 'Failed to load history')
    } finally {
      setLoadingHistory(false)
    }
  }, [kbId])

  const copyId = () => {
    navigator.clipboard?.writeText(kbId).then(
      () => toast.success(t('kbDetail.copied', 'ID copied')),
      () => toast.error(t('kbDetail.copyFailed', 'Copy failed'))
    )
  }

  const handleDelete = async () => {
    setDeleting(true)
    try {
      await deleteKnowledgeBase(kbId)
      toast.success(t('kbDetail.deleted', 'Knowledge base deleted'))
      setConfirmDelete(false)
      onBack()
    } catch (err) {
      toast.error(errorMessage(err) || 'Failed to delete')
    } finally {
      setDeleting(false)
    }
  }

  const handleDeleteHistory = async () => {
    if (!historyDeleteTarget) return
    // Only the document id maps to a deletable document; the track id is a
    // scheduling artifact and may not resolve in doc_status.
    const docId = historyDeleteTarget.doc_id
    if (!docId) {
      toast.error(t('kbDetail.deleteHistoryNoId', 'This task has no document ID'))
      return
    }
    setDeletingHistory(true)
    try {
      const res = await deleteKnowledgeBaseDocuments(kbId, [docId])
      toast.success(res.message || t('kbDetail.deleted', 'Deleted'))
      setHistoryDeleteTarget(null)
      await loadHistory()
    } catch (err) {
      toast.error(errorMessage(err) || 'Delete failed')
    } finally {
      setDeletingHistory(false)
    }
  }

  return (
    <div className="space-y-5">
      {/* Breadcrumb */}
      <button
        type="button"
        onClick={onBack}
        className="text-muted-foreground hover:text-foreground flex items-center gap-1 text-sm"
      >
        <ArrowLeft className="size-4" />
        {t('knowledgeBase.title', 'Knowledge Bases')}
      </button>

      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="bg-cyan-500/12 text-cyan-400 ring-cyan-500/22 flex size-12 shrink-0 items-center justify-center rounded-xl ring-1 ring-inset">
            <BookOpen className="size-6" />
          </div>
          <div className="min-w-0 flex-1">
            <h1
              className="text-2xl font-semibold tracking-tight break-all"
              title={kbId}
            >
              {kbName || kbId || t('common.loading', 'Loading…')}
            </h1>
            <div className="text-muted-foreground mt-1 flex items-center gap-2 text-sm">
              <span className="font-mono break-all">ID: {kbId}</span>
              <button
                type="button"
                onClick={copyId}
                aria-label={t('kbDetail.copyId', 'Copy ID')}
                className="hover:text-foreground rounded p-1"
              >
                <Copy className="size-3.5" />
              </button>
            </div>
          </div>
        </div>
        <Button variant="destructive" onClick={() => setConfirmDelete(true)}>
          <Trash2 className="size-4" />
          {t('kbDetail.deleteKb', 'Delete Knowledge Base')}
        </Button>
      </div>

      <Tabs value={tabValue} onValueChange={setTabValue} className="space-y-4">
        <TabsList>
          <TabsTrigger value="documents">
            {t('kbDetail.tabDocuments', 'Documents')}
          </TabsTrigger>
          <TabsTrigger value="upload">
            {t('kbDetail.tabUpload', 'Upload & Ingest')}
          </TabsTrigger>
        </TabsList>

        {/* Documents tab – dedicated KB document browser matching the
            reference screenshot (filter pills + compact table + per-row delete). */}
        <TabsContent value="documents" className="overflow-auto">
          <KnowledgeBaseDocuments
            kbId={kbId}
            onUploadClick={() => setTabValue('upload')}
          />
        </TabsContent>

        {/* Upload & ingest tab */}
        <TabsContent value="upload" className="space-y-4">
          <UploadPanel kbId={kbId} onUploaded={() => void loadHistory()} />
          <Card variant="glass" className="p-5">
            <h3 className="font-semibold">
              {t('kbDetail.historyTitle', 'Recent Ingestion History')}
            </h3>
            <div className="mt-3">
              {loadingHistory ? (
                <div className="py-8 text-center">
                  <Loader2 className="text-muted-foreground mx-auto size-5 animate-spin" />
                </div>
              ) : history.length === 0 ? (
                <p className="text-muted-foreground py-8 text-center text-sm">
                  {t('kbDetail.noHistory', 'No ingestion history yet.')}
                </p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>{t('kbDetail.colFile', 'File Name')}</TableHead>
                      <TableHead>{t('kbDetail.colId', 'ID')}</TableHead>
                      <TableHead>{t('kbDetail.colStatus', 'Status')}</TableHead>
                      <TableHead>{t('kbDetail.colUpdated', 'Updated')}</TableHead>
                      <TableHead className="text-right">
                        {t('kbDetail.colActions', 'Actions')}
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {history.map((h) => {
                      // Only a resolved document id is deletable; track-only
                      // rows cannot be routed to the delete endpoint.
                      const historyId = h.doc_id
                      return (
                        <TableRow key={h.track_id}>
                          <TableCell className="max-w-[260px] truncate font-medium">
                            {h.file_name}
                          </TableCell>
                          <TableCell
                            className="text-muted-foreground max-w-[150px] truncate font-mono text-xs"
                            title={historyId}
                          >
                            {historyId ? historyId.slice(0, 12) : '-'}
                          </TableCell>
                          <TableCell>
                            <StatusBadge status={h.status} />
                          </TableCell>
                          <TableCell className="text-muted-foreground text-sm">
                            {h.updated_at ? formatTime(h.updated_at) : '-'}
                          </TableCell>
                          <TableCell className="text-right">
                            <Button
                              variant="ghost"
                              size="icon"
                              className="text-muted-foreground hover:text-destructive"
                              tooltip={
                                historyId
                                  ? t('kbDetail.delete', 'Delete')
                                  : t('kbDetail.deleteHistoryNoId', 'No document ID')
                              }
                              disabled={!historyId}
                              onClick={() => setHistoryDeleteTarget(h)}
                            >
                              <Trash2 className="size-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              )}
            </div>
          </Card>
        </TabsContent>
      </Tabs>

      <Dialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {t('kbDetail.deleteTitle', 'Delete knowledge base?')}
            </DialogTitle>
            <DialogDescription>
              {t('kbDetail.deleteConfirm', 'This permanently removes all data in "{{id}}".').replace(
                '{{id}}',
                kbId
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmDelete(false)}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button variant="destructive" onClick={() => void handleDelete()} disabled={deleting}>
              {deleting && <Loader2 className="size-4 animate-spin" />}
              {t('kbDetail.deleteKb', 'Delete Knowledge Base')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete ingestion-history task confirm dialog */}
      <Dialog
        open={!!historyDeleteTarget}
        onOpenChange={(o) => !o && setHistoryDeleteTarget(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('kbDetail.deleteDocTitle', 'Delete document?')}</DialogTitle>
            <DialogDescription>
              {t('kbDetail.deleteDocConfirm', 'Delete "{{name}}"? This cannot be undone.').replace(
                '{{name}}',
                historyDeleteTarget?.file_name ?? ''
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setHistoryDeleteTarget(null)}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button
              variant="destructive"
              onClick={() => void handleDeleteHistory()}
              disabled={deletingHistory}
            >
              {deletingHistory && <Loader2 className="size-4 animate-spin" />}
              {t('kbDetail.delete', 'Delete')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

/**
 * Ingestion panel: parser/parse-method selection, content toggles, dropzone
 * and a submit button. Submits to the KB-specific upload endpoint.
 */
function UploadPanel({ kbId, onUploaded }: { kbId: string; onUploaded: () => void }) {
  const { t } = useTranslation()
  const [files, setFiles] = React.useState<File[]>([])
  const [parser, setParser] = React.useState('auto')
  // All content-processing toggles default ON, matching the reference design;
  // the server still gates VLM work behind its own VLM_PROCESS_ENABLE switch.
  const [imageProcessing, setImageProcessing] = React.useState(true)
  const [tableProcessing, setTableProcessing] = React.useState(true)
  const [formulaProcessing, setFormulaProcessing] = React.useState(true)
  const [enableVlm, setEnableVlm] = React.useState(true)
  const [uploading, setUploading] = React.useState(false)
  const [parserOptions, setParserOptions] = React.useState<string[]>(FALLBACK_PARSERS)

  // Load the live parser capability matrix from the document module so the
  // picker only offers engines that are actually configured on this server.
  React.useEffect(() => {
    let cancelled = false
    getSupportedFileTypes()
      .then((data) => {
        if (cancelled) return
        const engines = Object.keys(data.engines ?? {})
        if (engines.length > 0) {
          setParserOptions(['auto', ...engines.filter((e) => e !== 'auto')])
        }
      })
      .catch(() => {
        // Keep the fallback list; the backend still validates the final choice.
      })
    return () => {
      cancelled = true
    }
  }, [])

  const handleUpload = async () => {
    if (files.length === 0) {
      toast.error(t('kbDetail.noFile', 'Please select a file first'))
      return
    }
    setUploading(true)
    let ok = 0
    try {
      for (const file of files) {
        await uploadKnowledgeBaseDocument(kbId, file, {
          parser,
          imageProcessing,
          tableProcessing,
          formulaProcessing,
          enableVlm
        })
        ok += 1
      }
      toast.success(
        t('kbDetail.uploadSuccess', '{{n}} file(s) queued for ingestion').replace(
          '{{n}}',
          String(ok)
        )
      )
      setFiles([])
      onUploaded()
    } catch (err) {
      toast.error(errorMessage(err) || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <Card variant="glass" className="p-5">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <label className="text-sm font-medium">
            {t('kbDetail.parser', 'Parser')}
          </label>
          <Select value={parser} onValueChange={setParser}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {parserOptions.map((p) => (
                <SelectItem key={p} value={p}>
                  {p}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">
            {t('kbDetail.parseMethod', 'Parse Method')}
          </label>
          <Select value="auto" onValueChange={() => {}}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">auto</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Toggle
          label={t('kbDetail.image', 'Image')}
          checked={imageProcessing}
          onChange={setImageProcessing}
        />
        <Toggle
          label={t('kbDetail.table', 'Table')}
          checked={tableProcessing}
          onChange={setTableProcessing}
        />
        <Toggle
          label={t('kbDetail.formula', 'Formula')}
          checked={formulaProcessing}
          onChange={setFormulaProcessing}
        />
        <Toggle
          label={t('kbDetail.vlm', 'Enable VLM')}
          checked={enableVlm}
          onChange={setEnableVlm}
        />
      </div>

      <div className="mt-4">
        <FileUploader
          value={files}
          onValueChange={setFiles}
          multiple
          maxFileCount={20}
          className="glass-panel-strong border-dashed"
        />
      </div>

      <div className="mt-4 flex justify-end">
        <Button onClick={() => void handleUpload()} disabled={uploading || files.length === 0}>
          {uploading && <Loader2 className="size-4 animate-spin" />}
          {t('kbDetail.startIngest', 'Start Ingestion')}
        </Button>
      </div>
    </Card>
  )
}

function Toggle({
  label,
  checked,
  onChange
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-sm">
      <Checkbox checked={checked} onCheckedChange={(v) => onChange(Boolean(v))} />
      {label}
    </label>
  )
}

function formatTime(value: string): string {
  const d = new Date(value)
  if (isNaN(d.getTime())) return value
  return d.toLocaleString()
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message
  if (typeof err === 'string') return err
  return String(err)
}
