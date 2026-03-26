import type { GraphWorkbenchQueryRequest } from '@/api/lightrag'
import { useTranslation } from 'react-i18next'

type GraphWorkbenchSummaryProps = {
  draft: GraphWorkbenchQueryRequest
  appliedQuery: GraphWorkbenchQueryRequest | null
  queryVersion: number
  nodeCount: number
  edgeCount: number
}

const countActiveFilterGroups = (query: GraphWorkbenchQueryRequest): number => {
  let activeGroups = 0

  const hasNodeFilters =
    query.node_filters.entity_types.length > 0 ||
    !!query.node_filters.name_query.trim() ||
    !!query.node_filters.description_query.trim() ||
    query.node_filters.degree_min !== null ||
    query.node_filters.degree_max !== null ||
    query.node_filters.isolated_only
  if (hasNodeFilters) activeGroups += 1

  const hasEdgeFilters =
    query.edge_filters.relation_types.length > 0 ||
    !!query.edge_filters.keyword_query.trim() ||
    query.edge_filters.weight_min !== null ||
    query.edge_filters.weight_max !== null ||
    query.edge_filters.source_entity_types.length > 0 ||
    query.edge_filters.target_entity_types.length > 0
  if (hasEdgeFilters) activeGroups += 1

  const hasSourceFilters =
    !!query.source_filters.source_id_query.trim() ||
    query.source_filters.file_paths.length > 0 ||
    !!query.source_filters.time_from ||
    !!query.source_filters.time_to
  if (hasSourceFilters) activeGroups += 1

  const hasViewOptions =
    query.view_options.show_nodes_only ||
    query.view_options.show_edges_only ||
    query.view_options.hide_low_weight_edges ||
    query.view_options.hide_empty_description ||
    query.view_options.highlight_matches
  if (hasViewOptions) activeGroups += 1

  return activeGroups
}

const GraphWorkbenchSummary = ({
  draft,
  appliedQuery,
  queryVersion,
  nodeCount,
  edgeCount
}: GraphWorkbenchSummaryProps) => {
  const { t } = useTranslation()
  const summaryQuery = appliedQuery ?? draft
  const status = appliedQuery
    ? t('graphPanel.workbench.summary.appliedStatus')
    : t('graphPanel.workbench.summary.draftStatus')
  const activeGroups = countActiveFilterGroups(summaryQuery)

  return (
    <div className="bg-muted/40 rounded-lg border px-3 py-2 text-xs">
      <p className="font-semibold">{status}</p>
      <p className="text-muted-foreground">
        {t('graphPanel.workbench.summary.version', { queryVersion })}
      </p>
      <p className="mt-1">
        {t('graphPanel.workbench.summary.scope', {
          label: summaryQuery.scope.label,
          maxDepth: summaryQuery.scope.max_depth,
          maxNodes: summaryQuery.scope.max_nodes
        })}
      </p>
      <p>{t('graphPanel.workbench.summary.result', { nodeCount, edgeCount })}</p>
      <p>{t('graphPanel.workbench.summary.activeGroups', { activeGroups })}</p>
    </div>
  )
}

export default GraphWorkbenchSummary
