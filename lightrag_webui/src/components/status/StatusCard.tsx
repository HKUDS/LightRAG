import { LightragStatus } from '@/api/lightrag'
import { useTranslation } from 'react-i18next'

const StatusCard = ({ status }: { status: LightragStatus | null }) => {
  const { t } = useTranslation()
  if (!status) {
    return <div className="text-foreground text-xs">{t('graphPanel.statusCard.unavailable')}</div>
  }

  return (
    <div className="min-w-[300px] space-y-2 text-xs">
      <div className="space-y-1">
        <h4 className="font-medium">{t('graphPanel.statusCard.storageInfo')}</h4>
        <div className="text-foreground grid grid-cols-[120px_1fr] gap-1">
          <span>{t('graphPanel.statusCard.workingDirectory')}:</span>
          <span className="truncate">{status.working_directory}</span>
          <span>{t('graphPanel.statusCard.inputDirectory')}:</span>
          <span className="truncate">{status.input_directory}</span>
        </div>
      </div>

      <div className="space-y-1">
        <h4 className="font-medium">{t('graphPanel.statusCard.llmConfig')}</h4>
        <div className="text-foreground grid grid-cols-[120px_1fr] gap-1">
          <span>{t('graphPanel.statusCard.llmBinding')}:</span>
          <span>{status.configuration.llm_binding}</span>
          <span>{t('graphPanel.statusCard.llmBindingHost')}:</span>
          <span>{status.configuration.llm_binding_host}</span>
          <span>{t('graphPanel.statusCard.llmModel')}:</span>
          <span>{status.configuration.llm_model}</span>
          <span>{t('graphPanel.statusCard.maxTokens')}:</span>
          <span>{status.configuration.max_tokens}</span>
        </div>
      </div>

      <div className="space-y-1">
        <h4 className="font-medium">{t('graphPanel.statusCard.embeddingConfig')}</h4>
        <div className="text-foreground grid grid-cols-[120px_1fr] gap-1">
          <span>{t('graphPanel.statusCard.embeddingBinding')}:</span>
          <span>{status.configuration.embedding_binding}</span>
          <span>{t('graphPanel.statusCard.embeddingBindingHost')}:</span>
          <span>{status.configuration.embedding_binding_host}</span>
          <span>{t('graphPanel.statusCard.embeddingModel')}:</span>
          <span>{status.configuration.embedding_model}</span>
        </div>
      </div>

      <div className="space-y-1">
        <h4 className="font-medium">{t('graphPanel.statusCard.storageConfig')}</h4>
        <div className="text-foreground grid grid-cols-[120px_1fr] gap-1">
          <span>{t('graphPanel.statusCard.kvStorage')}:</span>
          <span>{status.configuration.kv_storage}</span>
          <span>{t('graphPanel.statusCard.docStatusStorage')}:</span>
          <span>{status.configuration.doc_status_storage}</span>
          <span>{t('graphPanel.statusCard.graphStorage')}:</span>
          <span>{status.configuration.graph_storage}</span>
          <span>{t('graphPanel.statusCard.vectorStorage')}:</span>
          <span>{status.configuration.vector_storage}</span>
        </div>
      </div>
    </div>
  )
}

export default StatusCard
