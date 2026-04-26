import type {
  LightragQueueStatus,
  LightragRoleLLMConfig,
  LightragStatus
} from '@/api/lightrag'
import { useTranslation } from 'react-i18next'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'

const ROLE_ORDER = ['extract', 'keyword', 'query', 'vlm']

type RoleLLMRow = {
  role: string
  config: LightragRoleLLMConfig
  queue?: LightragQueueStatus
}

const textValue = (value: string | number | null | undefined) => {
  if (value === null || value === undefined || value === '') return '-'
  return String(value)
}

const statValue = (value: number | undefined) => {
  return typeof value === 'number' ? value.toString() : '-'
}

const getModelRows = (status: LightragStatus): RoleLLMRow[] => {
  const configs = status.configuration.role_llm_config || {}
  const queues = status.llm_queue_status || {}
  const discoveredRoles = new Set([...Object.keys(configs), ...Object.keys(queues)])
  const orderedRoles = [
    ...ROLE_ORDER.filter((role) => discoveredRoles.has(role)),
    ...Array.from(discoveredRoles).filter((role) => !ROLE_ORDER.includes(role))
  ]

  const rows: RoleLLMRow[] = orderedRoles.map((role) => ({
    role,
    config: configs[role] || {
      binding: status.configuration.llm_binding,
      model: status.configuration.llm_model,
      host: status.configuration.llm_binding_host,
      max_async: status.configuration.max_async
    },
    queue: queues[role]
  }))

  if (!rows.length) {
    rows.push({
      role: 'base',
      config: {
        binding: status.configuration.llm_binding,
        model: status.configuration.llm_model,
        host: status.configuration.llm_binding_host,
        max_async: status.configuration.max_async
      }
    })
  }

  rows.push({
    role: 'embed',
    config: {
      binding: status.configuration.embedding_binding,
      model: status.configuration.embedding_model,
      host: status.configuration.embedding_binding_host,
      max_async: status.configuration.embedding_func_max_async
    },
    queue: status.embedding_queue_status
  })

  if (status.configuration.enable_rerank || status.rerank_queue_status?.available) {
    rows.push({
      role: 'rerank',
      config: {
        binding: status.configuration.rerank_binding,
        model: status.configuration.rerank_model,
        host: status.configuration.rerank_binding_host,
        max_async: status.rerank_queue_status?.max_async
      },
      queue: status.rerank_queue_status
    })
  }

  return rows
}

const StatusCard = ({ status }: { status: LightragStatus | null }) => {
  const { t } = useTranslation()
  if (!status) {
    return <div className="text-foreground text-xs">{t('graphPanel.statusCard.unavailable')}</div>
  }

  const roleRows = getModelRows(status)
  const storageWorkspaces = status.configuration.storage_workspaces
  const defaultWorkspace = status.configuration.workspace
  const storageColumns = [
    {
      key: 'kv',
      label: t('graphPanel.statusCard.kvStorage'),
      storageClass: status.configuration.kv_storage,
      workspace: storageWorkspaces?.kv_storage ?? defaultWorkspace
    },
    {
      key: 'doc-status',
      label: t('graphPanel.statusCard.docStatusStorage'),
      storageClass: status.configuration.doc_status_storage,
      workspace: storageWorkspaces?.doc_status_storage ?? defaultWorkspace
    },
    {
      key: 'graph',
      label: t('graphPanel.statusCard.graphStorage'),
      storageClass: status.configuration.graph_storage,
      workspace: storageWorkspaces?.graph_storage ?? defaultWorkspace
    },
    {
      key: 'vector',
      label: t('graphPanel.statusCard.vectorStorage'),
      storageClass: status.configuration.vector_storage,
      workspace: storageWorkspaces?.vector_storage ?? defaultWorkspace
    }
  ]

  return (
    <div className="min-w-[300px] space-y-2 text-xs">
      <div className="space-y-1">
        <h4 className="font-medium">{t('graphPanel.statusCard.serverInfo')}</h4>
        <div className="text-foreground grid grid-cols-[160px_1fr] gap-1">
          <span>{t('graphPanel.statusCard.workingDirectory')}:</span>
          <span className="truncate">{status.working_directory}</span>
          <span>{t('graphPanel.statusCard.inputDirectory')}:</span>
          <span className="truncate">{status.input_directory}</span>
          <span>{t('graphPanel.statusCard.summarySettings')}:</span>
          <span>{status.configuration.summary_language} / LLM summary on {status.configuration.force_llm_summary_on_merge.toString()} fragments</span>
          <span>{t('graphPanel.statusCard.threshold')}:</span>
          <span>cosine {status.configuration.cosine_threshold} / rerank_score {status.configuration.min_rerank_score} / max_related {status.configuration.related_chunk_number}</span>
          <span>{t('graphPanel.statusCard.otherSettings')}:</span>
          <span>max_graph_nodes {status.configuration.max_graph_nodes || '-'} / max_parallel_insert {status.configuration.max_parallel_insert}</span>
          {status.keyed_locks && (
            <>
              <span>{t('graphPanel.statusCard.lockStatus')}:</span>
              <span>
                mp {status.keyed_locks.current_status.pending_mp_cleanup}/{status.keyed_locks.current_status.total_mp_locks} |
                async {status.keyed_locks.current_status.pending_async_cleanup}/{status.keyed_locks.current_status.total_async_locks}
                (pid: {status.keyed_locks.process_id})
              </span>
            </>
          )}
        </div>
      </div>

      <div className="space-y-1">
        <h4 className="font-medium">{t('graphPanel.statusCard.llmConfig')}</h4>
        <div className="rounded-md border">
          <Table className="text-xs">
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="h-7 px-2 py-1">role</TableHead>
                <TableHead className="h-7 px-2 py-1">
                  binding/model
                </TableHead>
                <TableHead className="h-7 px-2 py-1">base_url</TableHead>
                <TableHead className="h-7 px-2 py-1 text-right">
                  queued
                </TableHead>
                <TableHead className="h-7 px-2 py-1 text-right">
                  run/max
                </TableHead>
                <TableHead className="h-7 px-2 py-1 text-right">
                  req
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {roleRows.map(({ role, config, queue }) => {
                const maxAsync = queue?.max_async ?? config.max_async
                return (
                  <TableRow key={role} className="hover:bg-muted/30">
                    <TableCell className="px-2 py-1 font-medium capitalize">
                      {role}
                    </TableCell>
                    <TableCell className="max-w-[170px] px-2 py-1">
                      <div className="truncate">{textValue(config.binding)}</div>
                      <div className="text-muted-foreground truncate">
                        {textValue(config.model)}
                      </div>
                    </TableCell>
                    <TableCell className="max-w-[220px] px-2 py-1">
                      <div className="truncate">{textValue(config.host)}</div>
                    </TableCell>
                    <TableCell className="px-2 py-1 text-right tabular-nums">
                      {statValue(queue?.queued)}
                    </TableCell>
                    <TableCell className="px-2 py-1 text-right tabular-nums">
                      {statValue(queue?.running)}/{statValue(maxAsync)}
                    </TableCell>
                    <TableCell className="px-2 py-1 text-right tabular-nums">
                      {statValue(queue?.submitted_total)}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      </div>

      <div className="space-y-1">
        <h4 className="font-medium">{t('graphPanel.statusCard.storageConfig')}</h4>
        <div className="rounded-md border">
          <Table className="text-xs">
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                {storageColumns.map(({ key, label }) => (
                  <TableHead key={key} className="h-7 min-w-[130px] px-2 py-1">
                    {label}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow className="hover:bg-muted/30">
                {storageColumns.map(({ key, storageClass }) => (
                  <TableCell key={`${key}-class`} className="break-all px-2 py-1 align-top font-medium">
                    {textValue(storageClass)}
                  </TableCell>
                ))}
              </TableRow>
              <TableRow className="hover:bg-muted/30">
                {storageColumns.map(({ key, workspace }) => (
                  <TableCell key={`${key}-workspace`} className="text-muted-foreground break-all px-2 py-1 align-top">
                    {textValue(workspace)}
                  </TableCell>
                ))}
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  )
}

export default StatusCard
