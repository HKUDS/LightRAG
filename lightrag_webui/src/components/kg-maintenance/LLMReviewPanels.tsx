import Button from '@/components/ui/Button'
import type { KBIterationProposalFunnelReport } from '@/api/lightrag'
import { BrainCircuitIcon, FileDiffIcon, ShieldCheckIcon } from 'lucide-react'
import type { ReactNode } from 'react'

export type TraceRound = {
  round_id?: string
  focus?: string[]
  proposal_ids?: string[]
  state?: string
}

export type TraceAttemptLog = {
  attempt?: number
  state?: string
  error?: string
}

export type TraceStage = {
  stage?: string
  state?: string
  attempts?: number
  artifact_keys?: string[]
  proposal_ids?: string[]
  attempt_logs?: TraceAttemptLog[]
}

export type LLMReviewPanelProps = {
  trace: Record<string, any> | null
  report: string
  proposals: string
  issueAnalysis: string
  missingBranchInference: string
  evidenceMap: string
  repairPlan: string
  deterministicProposalReport?: KBIterationProposalFunnelReport | null
  running: boolean
  onRun: () => void
}

type PatchCandidatesPanelProps = {
  proposals: string
  proposalIdSource?: string
  patchText: string
  onLoadPatch: (proposalId: string) => void
}

type LLMJudgePanelProps = {
  report: string
}

export function LLMReviewPanel({
  trace,
  report,
  proposals,
  issueAnalysis,
  missingBranchInference,
  evidenceMap,
  repairPlan,
  deterministicProposalReport,
  running,
  onRun
}: LLMReviewPanelProps) {
  const stopReason = typeof trace?.stop_reason === 'string' ? trace.stop_reason : ''
  const rounds = Array.isArray(trace?.rounds) ? (trace.rounds as TraceRound[]) : []
  const stages = Array.isArray(trace?.stages)
    ? trace.stages.map(normalizeTraceStageEntry).filter(isTraceStage)
    : []
  const subtitle = stopReason
    ? `停止原因：${stopReason}。辅助材料，不会自动修改 KG。`
    : '尚未生成 LLM 审阅 trace。辅助材料，不会自动修改 KG。'

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <PanelHeader
          icon={<BrainCircuitIcon className="size-4 text-sky-500" />}
          title="LLM 审阅材料"
          subtitle={subtitle}
        />
        <Button variant="outline" size="sm" disabled={running} onClick={onRun}>
          <BrainCircuitIcon className="size-4" />
          {running ? '运行中' : '运行 LLM 审阅'}
        </Button>
      </div>

      <div className="border-border/70 bg-muted/20 rounded-lg border p-3 text-sm">
        <span className="text-muted-foreground mr-2">停止原因</span>
        <span className="font-medium">{stopReason || '暂无 stop_reason'}</span>
      </div>

      <div className="border-border/70 bg-muted/20 rounded-lg border p-3 text-sm">
        LLM Agent 只生成分析、proposal、证据位置和修复排序；任何会修改 KG、规则、prompt、workspace
        或 WebUI 的操作都必须经过人工批准，不会自动修改 KG。
      </div>

      <div className="space-y-2">
        <h3 className="text-sm font-medium">多阶段 LLM Agent</h3>
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {TRACE_STAGE_LABELS.map(({ key, label }) => {
            const stage = stages.find((item) => normalizeTraceStage(item.stage) === key)
            return (
              <article key={key} className="border-border/70 rounded-lg border p-3 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{label}</span>
                  {stage?.state ? (
                    <span className="bg-muted rounded-md px-2 py-1 text-xs">{stage.state}</span>
                  ) : null}
                </div>
                {typeof stage?.attempts === 'number' ? (
                  <div className="text-muted-foreground mt-2 text-xs">
                    尝试次数：<span className="font-medium">{stage.attempts}</span>
                  </div>
                ) : null}
                <AttemptLogList logs={stage?.attempt_logs} />
                <RoundField
                  label="产物键"
                  values={stage?.artifact_keys}
                  emptyText="暂无产物键。"
                />
                <RoundField
                  label="提案 ID"
                  values={stage?.proposal_ids}
                  emptyText="暂无提案 ID。"
                />
              </article>
            )
          })}
        </div>
      </div>

      <ArtifactBlock title="问题解释" content={issueAnalysis} emptyText="暂无问题解释。" />
      <DeterministicProposalFunnel report={deterministicProposalReport} />
      <ArtifactBlock
        title="缺失分支推断"
        content={missingBranchInference}
        emptyText="暂无缺失分支推断。"
      />
      <ArtifactBlock title="证据定位" content={evidenceMap} emptyText="暂无证据定位。" />
      <ArtifactBlock title="修复方案排序" content={repairPlan} emptyText="暂无修复方案排序。" />

      <div className="space-y-2">
        <h3 className="text-sm font-medium">审阅轮次</h3>
        {rounds.length ? (
          rounds.map((round, index) => (
            <article
              key={round.round_id || `round-${index}`}
              className="border-border/70 rounded-lg border p-3 text-sm"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">{round.round_id || `round-${index + 1}`}</span>
                {round.state ? (
                  <span className="bg-muted rounded-md px-2 py-1 text-xs">{round.state}</span>
                ) : null}
              </div>
              <RoundField label="关注项" values={round.focus} emptyText="暂无关注项。" />
              <RoundField
                label="提案 ID"
                values={round.proposal_ids}
                emptyText="暂无提案 ID。"
              />
            </article>
          ))
        ) : (
          <EmptyBlock>暂无审阅轮次。</EmptyBlock>
        )}
      </div>

      <ArtifactBlock title="LLM 审阅报告" content={report} emptyText="暂无审阅报告。" />
      <ArtifactBlock title="生成的提案" content={proposals} emptyText="暂无候选提案。" />
    </section>
  )
}

const TRACE_STAGE_LABELS = [
  { key: 'explain', label: '问题解释' },
  { key: 'infer_branches', label: '缺失分支' },
  { key: 'locate_evidence', label: '证据定位' },
  { key: 'propose', label: '生成提案' },
  { key: 'rank_repairs', label: '修复排序' },
  { key: 'judge', label: '评判复核' }
]

const FAMILY_LABELS: Record<string, string> = {
  diagnosis: '诊断',
  treatment: '治疗',
  risk_safety: '风险/安全',
  prevention: '预防',
  clinical_modeling: '临床建模',
  entity_cleanup: '实体清理',
  legacy_schema: '旧关系',
  direction: '方向修正',
  multi_predicate_split: '多谓词拆分',
  alias_role_conflict: '别名冲突'
}

type FunnelFamilyRow = {
  family: string
  rawIssueCount: number
  issueWithCandidateCount: number
  deterministicCandidateIssueCount: number
  actionCandidateCount: number
  deterministicCoveredCount: number
  deterministicProposalCount: number
  llmResidualCount: number
  llmResidualEligibleCount: number
  llmResidualSelectedCount: number
  validLlmProposalCount: number
  blockedCount: number
  schemaBlockedCount: number
  safetyBlockedCount: number
  evidenceBlockedCount: number
  applyBlockedCount: number
  decisionMemoryBlockedCount: number
  conflictCount: number
  deferredBudgetCount: number
  conversionFailureCount: number
  mergeDropCount: number
  selectedProposalCount: number
  selectedApprovalProposalCount: number
  topReasonCode: string
}

function DeterministicProposalFunnel({
  report
}: {
  report?: KBIterationProposalFunnelReport | null
}) {
  const rows = deterministicFunnelRows(report)
  if (!rows.length) {
    return null
  }
  const summary = isRecord(report?.summary) ? report.summary : {}
  const conflictGroups = deterministicConflictGroups(report)

  return (
    <div className="border-border/70 rounded-lg border p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium">按家族漏斗统计</h3>
          <p className="text-muted-foreground mt-1 text-xs">确定性提案漏斗</p>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs md:grid-cols-4">
          <MetricPill label="问题入账率" value={formatRate(summary.issue_accounting_rate)} />
          <MetricPill label="候选校验率" value={formatRate(summary.candidate_validation_rate)} />
          <MetricPill label="候选转提案率" value={formatRate(summary.candidate_to_proposal_rate)} />
          <MetricPill label="队列应用支持率" value={formatRate(summary.queue_apply_support_rate)} />
        </div>
      </div>
      <div className="mt-3 overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-muted/40 text-muted-foreground">
            <tr>
              {[
                '家族',
                '原始问题',
                '有候选',
                '候选动作',
                '确定性覆盖',
                '确定性提案',
                'LLM 剩余',
                'LLM 有效提案',
                '阻塞',
                '延后',
                '已选提案',
                '主要原因'
              ].map((heading) => (
                <th key={heading} className="px-3 py-2 text-left font-medium">
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.family} className="border-border/60 border-t">
                <td className="px-3 py-2 font-medium">{familyLabel(row.family)}</td>
                <NumericCell value={row.rawIssueCount} />
                <NumericCell value={row.issueWithCandidateCount} />
                <NumericCell value={row.actionCandidateCount} />
                <NumericCell value={row.deterministicCoveredCount} />
                <NumericCell value={row.deterministicProposalCount} />
                <NumericCell value={row.llmResidualCount} />
                <NumericCell value={row.validLlmProposalCount} />
                <NumericCell value={row.blockedCount} />
                <NumericCell value={row.deferredBudgetCount} />
                <NumericCell value={row.selectedApprovalProposalCount} />
                <td className="px-3 py-2">{row.topReasonCode}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4 grid gap-3 xl:grid-cols-2">
        <FunnelBreakdownTable title="阻断原因" rows={rows} mode="blockers" />
        <FunnelBreakdownTable title="LLM 剩余已选/延后" rows={rows} mode="residual" />
      </div>
      <div className="mt-4 grid gap-3 xl:grid-cols-2">
        <ConflictGroupsPanel groups={conflictGroups} />
        <RecurrencePanel summary={summary} />
      </div>
    </div>
  )
}

function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-muted/40 rounded-md px-2 py-1">
      <div className="text-muted-foreground">{label}</div>
      <div className="font-medium tabular-nums">{value}</div>
    </div>
  )
}

function NumericCell({ value }: { value: number }) {
  return <td className="px-3 py-2 text-right">{value}</td>
}

function deterministicFunnelRows(report?: KBIterationProposalFunnelReport | null): FunnelFamilyRow[] {
  const families = report?.families
  if (Array.isArray(families)) {
    return families
      .filter(isRecord)
      .map((family) => deterministicFunnelRow(stringValue(family.family), family))
  }
  if (isRecord(families)) {
    return Object.entries(families)
      .filter((entry): entry is [string, Record<string, unknown>] => isRecord(entry[1]))
      .map(([familyKey, metrics]) => deterministicFunnelRow(familyKey, metrics))
  }
  return []
}

function deterministicFunnelRow(
  familyKey: string,
  metrics: Record<string, unknown>
): FunnelFamilyRow {
  const reasonCodeCounts = isRecord(metrics.reason_code_counts)
    ? metrics.reason_code_counts
    : {}
  const schemaBlockedCount = metricNumber(metrics, 'schema_blocked_count', 'blocked_schema_count')
  const safetyBlockedCount = metricNumber(metrics, 'safety_blocked_count', 'blocked_safety_count')
  const evidenceBlockedCount = metricNumber(
    metrics,
    'evidence_blocked_count',
    'blocked_evidence_count'
  )
  const applyBlockedCount = metricNumber(metrics, 'apply_blocked_count', 'blocked_apply_count')
  const decisionMemoryBlockedCount = metricNumber(
    metrics,
    'decision_memory_blocked_count',
    'blocked_decision_memory_count'
  )
  const deferredBudgetCount = metricNumber(
    metrics,
    'deferred_by_family_cap_count',
    'deferred_budget_count'
  )
  const llmResidualEligibleCount = metricNumber(
    metrics,
    'llm_residual_eligible_count',
    'llm_residual_count'
  )
  const selectedApprovalProposalCount = metricNumber(
    metrics,
    'selected_approval_proposal_count',
    'selected_proposal_count'
  )
  return {
    family: stringValue(metrics.family) || familyKey || 'unknown',
    rawIssueCount: numericValue(metrics.raw_issue_count),
    issueWithCandidateCount: numericValue(metrics.issue_with_candidate_count),
    deterministicCandidateIssueCount: metricNumber(
      metrics,
      'deterministic_candidate_issue_count',
      'issue_with_candidate_count'
    ),
    actionCandidateCount: numericValue(metrics.action_candidate_count),
    deterministicCoveredCount: numericValue(metrics.deterministic_covered_count),
    deterministicProposalCount: metricNumber(
      metrics,
      'deterministic_proposal_count',
      'deterministic_covered_count'
    ),
    llmResidualCount: numericValue(metrics.llm_residual_count),
    llmResidualEligibleCount,
    llmResidualSelectedCount: numericValue(metrics.llm_residual_selected_count),
    validLlmProposalCount: numericValue(metrics.valid_llm_proposal_count),
    blockedCount:
      schemaBlockedCount +
      safetyBlockedCount +
      evidenceBlockedCount +
      applyBlockedCount +
      decisionMemoryBlockedCount,
    schemaBlockedCount,
    safetyBlockedCount,
    evidenceBlockedCount,
    applyBlockedCount,
    decisionMemoryBlockedCount,
    conflictCount: numericValue(metrics.conflict_count),
    deferredBudgetCount,
    conversionFailureCount: numericValue(metrics.conversion_failure_count),
    mergeDropCount: numericValue(metrics.merge_drop_count),
    selectedProposalCount: numericValue(metrics.selected_proposal_count),
    selectedApprovalProposalCount,
    topReasonCode: topReasonCode(reasonCodeCounts)
  }
}

function FunnelBreakdownTable({
  title,
  rows,
  mode
}: {
  title: string
  rows: FunnelFamilyRow[]
  mode: 'blockers' | 'residual'
}) {
  const headings =
    mode === 'blockers'
      ? ['家族', '结构', '安全', '证据', '应用', '记忆', '冲突', '转换失败', '合并丢弃']
      : ['家族', '可送 LLM', '已选 LLM', '延后', '有效 LLM 提案', '已进审批']
  return (
    <section>
      <h4 className="text-sm font-medium">{title}</h4>
      <div className="mt-2 overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-muted/40 text-muted-foreground">
            <tr>
              {headings.map((heading) => (
                <th key={heading} className="px-3 py-2 text-left font-medium">
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${title}-${row.family}`} className="border-border/60 border-t">
                <td className="px-3 py-2 font-medium">{familyLabel(row.family)}</td>
                {mode === 'blockers' ? (
                  <>
                    <NumericCell value={row.schemaBlockedCount} />
                    <NumericCell value={row.safetyBlockedCount} />
                    <NumericCell value={row.evidenceBlockedCount} />
                    <NumericCell value={row.applyBlockedCount} />
                    <NumericCell value={row.decisionMemoryBlockedCount} />
                    <NumericCell value={row.conflictCount} />
                    <NumericCell value={row.conversionFailureCount} />
                    <NumericCell value={row.mergeDropCount} />
                  </>
                ) : (
                  <>
                    <NumericCell value={row.llmResidualEligibleCount} />
                    <NumericCell value={row.llmResidualSelectedCount} />
                    <NumericCell value={row.deferredBudgetCount} />
                    <NumericCell value={row.validLlmProposalCount} />
                    <NumericCell value={row.selectedApprovalProposalCount} />
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function ConflictGroupsPanel({ groups }: { groups: Record<string, unknown>[] }) {
  return (
    <section>
      <h4 className="text-sm font-medium">冲突组</h4>
      {groups.length ? (
        <div className="mt-2 overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-muted/40 text-muted-foreground">
              <tr>
                {['目标', '提案', '原因'].map((heading) => (
                  <th key={heading} className="px-3 py-2 text-left font-medium">
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {groups.map((group, index) => (
                <tr key={`conflict-${index}`} className="border-border/60 border-t">
                  <td className="px-3 py-2">{conflictTarget(group)}</td>
                  <td className="px-3 py-2">
                    {stringArray(group.proposal_ids)?.join(', ') || '暂无'}
                  </td>
                  <td className="px-3 py-2">{stringValue(group.reason) || '需要人工复核'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyBlock>暂无冲突组。</EmptyBlock>
      )}
    </section>
  )
}

function RecurrencePanel({ summary }: { summary: Record<string, unknown> }) {
  return (
    <section>
      <h4 className="text-sm font-medium">复现的拒绝记忆命中数</h4>
      <div className="mt-2 grid grid-cols-1 gap-2 text-sm sm:grid-cols-3">
        <MetricPill
          label="硬拒绝复现"
          value={String(numericValue(summary.hard_rejection_recurrence_count))}
        />
        <MetricPill
          label="精确重复复现"
          value={String(numericValue(summary.exact_duplicate_recurrence_count))}
        />
        <MetricPill
          label="已知坏模式"
          value={String(numericValue(summary.known_bad_pattern_count))}
        />
      </div>
    </section>
  )
}

function familyLabel(family: string) {
  return FAMILY_LABELS[family] ?? family
}

function topReasonCode(reasonCodeCounts: Record<string, unknown>) {
  let topKey = ''
  let topCount = -1
  for (const [key, value] of Object.entries(reasonCodeCounts)) {
    const count = numericValue(value)
    if (count > topCount) {
      topKey = key
      topCount = count
    }
  }
  return topKey || '暂无'
}

export function PatchCandidatesPanel({
  proposals,
  proposalIdSource,
  patchText,
  onLoadPatch
}: PatchCandidatesPanelProps) {
  const proposalIds = parseProposalIds(proposalIdSource ?? proposals)

  return (
    <section className="space-y-4">
      <PanelHeader
        icon={<FileDiffIcon className="size-4 text-emerald-500" />}
        title="候选 Patch"
        subtitle="从 proposal 加载 Patch 候选，仅供人工检查，不会自动应用。"
      />

      <div className="space-y-2">
        <h3 className="text-sm font-medium">提案 ID</h3>
        {proposalIds.length ? (
          <div className="flex flex-wrap gap-2">
            {proposalIds.map((proposalId) => (
              <Button
                key={proposalId}
                variant="outline"
                size="sm"
                onClick={() => onLoadPatch(proposalId)}
              >
                {proposalId}
              </Button>
            ))}
          </div>
        ) : (
          <EmptyBlock>暂无可加载的提案 ID。</EmptyBlock>
        )}
      </div>

      <ArtifactBlock title="已选择 Patch" content={patchText} emptyText="尚未选择候选 Patch。" />
      <ArtifactBlock title="提案来源" content={proposals} emptyText="暂无候选提案。" />
    </section>
  )
}

export function LLMJudgePanel({ report }: LLMJudgePanelProps) {
  return (
    <section className="space-y-4">
      <PanelHeader
        icon={<ShieldCheckIcon className="size-4 text-indigo-500" />}
        title="Judge 评判"
        subtitle="LLM judge 报告与人工复核状态"
      />
      <ArtifactBlock title="Judge 报告" content={report} emptyText="暂无 Judge 评判报告。" />
    </section>
  )
}

function PanelHeader({
  icon,
  title,
  subtitle
}: {
  icon: ReactNode
  title: string
  subtitle: string
}) {
  return (
    <div>
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-sm font-semibold">{title}</h2>
      </div>
      <p className="text-muted-foreground mt-1 text-sm">{subtitle}</p>
    </div>
  )
}

function RoundField({
  label,
  values,
  emptyText
}: {
  label: string
  values?: string[]
  emptyText: string
}) {
  const hasValues = Array.isArray(values) && values.length > 0
  return (
    <div className="mt-3">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 flex flex-wrap gap-2">
        {hasValues ? (
          values.map((value) => (
            <span key={value} className="bg-muted rounded-md px-2 py-1 text-xs">
              {value}
            </span>
          ))
        ) : (
          <span className="text-muted-foreground text-sm">{emptyText}</span>
        )}
      </div>
    </div>
  )
}

function AttemptLogList({ logs }: { logs?: TraceAttemptLog[] }) {
  if (!Array.isArray(logs) || logs.length === 0) {
    return null
  }

  return (
    <div className="mt-3 space-y-1">
      <div className="text-muted-foreground text-xs">被拒绝的尝试</div>
      {logs.map((log, index) => (
        <div
          key={`${log.attempt || index}-${log.error || ''}`}
          className="bg-muted rounded-md px-2 py-1 text-xs"
        >
          <span className="font-medium">第 {log.attempt || index + 1} 次</span>
          {log.state ? <span className="text-muted-foreground"> - {log.state}</span> : null}
          {log.error ? <div className="mt-1 break-words">{log.error}</div> : null}
        </div>
      ))}
    </div>
  )
}

function ArtifactBlock({
  title,
  content,
  emptyText
}: {
  title: string
  content: string
  emptyText: string
}) {
  const hasContent = content.trim().length > 0
  return (
    <div className="border-border/70 rounded-lg border p-3">
      <h3 className="text-sm font-medium">{title}</h3>
      <pre className="text-muted-foreground mt-3 max-h-80 overflow-auto text-xs break-words whitespace-pre-wrap">
        {hasContent ? content : emptyText}
      </pre>
    </div>
  )
}

function EmptyBlock({ children }: { children: ReactNode }) {
  return (
    <div className="border-border/70 bg-muted/20 text-muted-foreground rounded-lg border p-3 text-sm">
      {children}
    </div>
  )
}

function isTraceStage(stage: TraceStage | null): stage is TraceStage {
  return stage !== null
}

function numericValue(value: unknown) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function metricNumber(metrics: Record<string, unknown>, key: string, fallbackKey: string) {
  if (key in metrics) {
    return numericValue(metrics[key])
  }
  return numericValue(metrics[fallbackKey])
}

function formatRate(value: unknown) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '暂无'
  }
  return `${Math.round(value * 1000) / 10}%`
}

function stringValue(value: unknown) {
  return typeof value === 'string' ? value.trim() : ''
}

function deterministicConflictGroups(report?: KBIterationProposalFunnelReport | null) {
  const primary = Array.isArray(report?.conflict_groups) ? report?.conflict_groups : undefined
  const fallback = Array.isArray(report?.conflicts) ? report?.conflicts : undefined
  return (primary || fallback || []).filter(isRecord)
}

function conflictTarget(group: Record<string, unknown>) {
  return (
    stringValue(group.target) ||
    stringValue(group.issue_ref) ||
    stringArray(group.candidate_ids)?.join(', ') ||
    '未标注'
  )
}

function normalizeTraceStageEntry(stage: unknown): TraceStage | null {
  if (!isRecord(stage)) {
    return null
  }

  return {
    stage: typeof stage.stage === 'string' ? stage.stage : undefined,
    state: typeof stage.state === 'string' ? stage.state : undefined,
    attempts:
      typeof stage.attempts === 'number' && Number.isFinite(stage.attempts)
        ? stage.attempts
        : undefined,
    artifact_keys: stringArray(stage.artifact_keys),
    proposal_ids: stringArray(stage.proposal_ids),
    attempt_logs: attemptLogArray(stage.attempt_logs)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function stringArray(value: unknown) {
  if (!Array.isArray(value)) {
    return undefined
  }

  const strings = value.filter((item): item is string => typeof item === 'string')
  return strings.length ? strings : undefined
}

function attemptLogArray(value: unknown) {
  if (!Array.isArray(value)) {
    return undefined
  }

  const logs = value
    .map((item): TraceAttemptLog | null => {
      if (!isRecord(item)) {
        return null
      }
      const attempt =
        typeof item.attempt === 'number' && Number.isFinite(item.attempt) ? item.attempt : undefined
      const state = typeof item.state === 'string' ? item.state : undefined
      const error = typeof item.error === 'string' ? item.error : undefined
      if (attempt === undefined && !state && !error) {
        return null
      }
      return { attempt, state, error }
    })
    .filter((item): item is TraceAttemptLog => item !== null)

  return logs.length ? logs : undefined
}

function normalizeTraceStage(stage?: string) {
  return typeof stage === 'string' ? stage.trim().toLowerCase() : ''
}

function parseProposalIds(proposals: string) {
  const ids = new Set<string>()
  const idPattern = /^\s*-?\s*id:\s*['"]?([^'"\s#]+)['"]?/gm
  let match: RegExpExecArray | null

  while ((match = idPattern.exec(proposals)) !== null) {
    ids.add(match[1])
  }

  return Array.from(ids)
}
