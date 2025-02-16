import { LightragStatus } from '@/api/lightrag'

const StatusCard = ({ status }: { status: LightragStatus | null }) => {
  if (!status) {
    return <div className="text-muted-foreground text-sm">Status information unavailable</div>
  }

  return (
    <div className="min-w-[300px] space-y-3 text-sm">
      <div className="space-y-1">
        <h4 className="font-medium">Storage Info</h4>
        <div className="text-muted-foreground grid grid-cols-2 gap-1">
          <span>Working Directory:</span>
          <span className="truncate">{status.working_directory}</span>
          <span>Input Directory:</span>
          <span className="truncate">{status.input_directory}</span>
        </div>
      </div>

      <div className="space-y-1">
        <h4 className="font-medium">LLM Configuration</h4>
        <div className="text-muted-foreground grid grid-cols-2 gap-1">
          <span>LLM Binding:</span>
          <span>{status.configuration.llm_binding}</span>
          <span>LLM Binding Host:</span>
          <span>{status.configuration.llm_binding_host}</span>
          <span>LLM Model:</span>
          <span>{status.configuration.llm_model}</span>
          <span>Max Tokens:</span>
          <span>{status.configuration.max_tokens}</span>
        </div>
      </div>

      <div className="space-y-1">
        <h4 className="font-medium">Embedding Configuration</h4>
        <div className="text-muted-foreground grid grid-cols-2 gap-1">
          <span>Embedding Binding:</span>
          <span>{status.configuration.embedding_binding}</span>
          <span>Embedding Binding Host:</span>
          <span>{status.configuration.embedding_binding_host}</span>
          <span>Embedding Model:</span>
          <span>{status.configuration.embedding_model}</span>
        </div>
      </div>

      <div className="space-y-1">
        <h4 className="font-medium">Storage Configuration</h4>
        <div className="text-muted-foreground grid grid-cols-2 gap-1">
          <span>KV Storage:</span>
          <span>{status.configuration.kv_storage}</span>
          <span>Doc Status Storage:</span>
          <span>{status.configuration.doc_status_storage}</span>
          <span>Graph Storage:</span>
          <span>{status.configuration.graph_storage}</span>
          <span>Vector Storage:</span>
          <span>{status.configuration.vector_storage}</span>
        </div>
      </div>
    </div>
  )
}

export default StatusCard
