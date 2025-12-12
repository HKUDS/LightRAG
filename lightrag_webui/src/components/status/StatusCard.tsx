import { useTranslation } from 'react-i18next'
import type { LightragStatus } from '@/api/lightrag'

const StatusCard = ({ status }: { status: LightragStatus | null }) => {
  const { t } = useTranslation()
  if (!status) {
    return <div className="text-foreground text-xs">{t('graphPanel.statusCard.unavailable')}</div>
  }

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
          <span>
            {status.configuration.summary_language} / LLM summary on{' '}
            {status.configuration.force_llm_summary_on_merge.toString()} fragments
          </span>
          <span>{t('graphPanel.statusCard.threshold')}:</span>
          <span>
            cosine {status.configuration.cosine_threshold} / rerank_score{' '}
            {status.configuration.min_rerank_score} / max_related{' '}
            {status.configuration.related_chunk_number}
          </span>
          <span>{t('graphPanel.statusCard.maxParallelInsert')}:</span>
          <span>{status.configuration.max_parallel_insert}</span>
        </div>
      </div>

      <div className="space-y-1">
        <h4 className="font-medium">{t('graphPanel.statusCard.llmConfig')}</h4>
        <div className="text-foreground grid grid-cols-[160px_1fr] gap-1">
          <span>{t('graphPanel.statusCard.llmBindingHost')}:</span>
          <span>{status.configuration.llm_binding_host}</span>
          <span>{t('graphPanel.statusCard.llmModel')}:</span>
          <span>
            {status.configuration.llm_binding}: {status.configuration.llm_model} (#
            {status.configuration.max_async} Async)
          </span>
        </div>
      </div>

      <div className="space-y-1">
        <h4 className="font-medium">{t('graphPanel.statusCard.embeddingConfig')}</h4>
        <div className="text-foreground grid grid-cols-[160px_1fr] gap-1">
          <span>{t('graphPanel.statusCard.embeddingBindingHost')}:</span>
          <span>{status.configuration.embedding_binding_host}</span>
          <span>{t('graphPanel.statusCard.embeddingModel')}:</span>
          <span>
            {status.configuration.embedding_binding}: {status.configuration.embedding_model} (#
            {status.configuration.embedding_func_max_async} Async *{' '}
            {status.configuration.embedding_batch_num} batches)
          </span>
        </div>
      </div>

      {status.configuration.enable_rerank && (
        <div className="space-y-1">
          <h4 className="font-medium">{t('graphPanel.statusCard.rerankerConfig')}</h4>
          <div className="text-foreground grid grid-cols-[160px_1fr] gap-1">
            <span>{t('graphPanel.statusCard.rerankerBindingHost')}:</span>
            <span>{status.configuration.rerank_binding_host || '-'}</span>
            <span>{t('graphPanel.statusCard.rerankerModel')}:</span>
            <span>
              {status.configuration.rerank_binding || '-'} :{' '}
              {status.configuration.rerank_model || '-'}
            </span>
          </div>
        </div>
      )}

      <div className="space-y-1">
        <h4 className="font-medium">{t('graphPanel.statusCard.storageConfig')}</h4>
        <div className="text-foreground grid grid-cols-[160px_1fr] gap-1">
          <span>{t('graphPanel.statusCard.kvStorage')}:</span>
          <span>{status.configuration.kv_storage}</span>
          <span>{t('graphPanel.statusCard.docStatusStorage')}:</span>
          <span>{status.configuration.doc_status_storage}</span>
          <span>{t('graphPanel.statusCard.graphStorage')}:</span>
          <span>{status.configuration.graph_storage}</span>
          <span>{t('graphPanel.statusCard.vectorStorage')}:</span>
          <span>{status.configuration.vector_storage}</span>
          <span>{t('graphPanel.statusCard.workspace')}:</span>
          <span>{status.configuration.workspace || '-'}</span>
          <span>{t('graphPanel.statusCard.maxGraphNodes')}:</span>
          <span>{status.configuration.max_graph_nodes || '-'}</span>
          {status.keyed_locks && (
            <>
              <span>{t('graphPanel.statusCard.lockStatus')}:</span>
              <span>
                mp {status.keyed_locks.current_status.pending_mp_cleanup}/
                {status.keyed_locks.current_status.total_mp_locks} | async{' '}
                {status.keyed_locks.current_status.pending_async_cleanup}/
                {status.keyed_locks.current_status.total_async_locks}
                (pid: {status.keyed_locks.process_id})
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default StatusCard
