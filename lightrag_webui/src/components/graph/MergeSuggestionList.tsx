import Button from '@/components/ui/Button'
import { useTranslation } from 'react-i18next'
import type { GraphMergeSuggestionCandidate, GraphMergeSuggestionReason } from '@/api/lightrag'

type MergeSuggestionListProps = {
  candidates: GraphMergeSuggestionCandidate[]
  selectedTargets?: string[]
  isLoading?: boolean
  errorMessage?: string | null
  onImportCandidate: (candidate: GraphMergeSuggestionCandidate) => void
}

type MergeSuggestionTranslator = (key: string, options?: Record<string, unknown>) => string

const formatReason = (
  reason: GraphMergeSuggestionReason,
  t?: MergeSuggestionTranslator
): string => {
  if (!t) {
    return `${reason.code} (${reason.score.toFixed(2)})`
  }

  const reasonCodeKey = `graphPanel.workbench.merge.suggestions.reasonCodes.${reason.code}`
  const localizedReasonCode = t(reasonCodeKey)
  const reasonLabel = localizedReasonCode === reasonCodeKey ? reason.code : localizedReasonCode
  return t('graphPanel.workbench.merge.suggestions.reasonScore', {
    reason: reasonLabel,
    score: reason.score.toFixed(2)
  })
}

export const buildMergeCandidateEvidence = (
  candidate: GraphMergeSuggestionCandidate,
  t?: MergeSuggestionTranslator
): string => {
  if (!candidate.reasons.length) {
    return t
      ? t('graphPanel.workbench.merge.suggestions.noEvidence')
      : 'No explicit evidence provided.'
  }
  return candidate.reasons.map((reason) => formatReason(reason, t)).join(', ')
}

const MergeSuggestionList = ({
  candidates,
  selectedTargets = [],
  isLoading = false,
  errorMessage = null,
  onImportCandidate
}: MergeSuggestionListProps) => {
  const { t } = useTranslation()
  return (
    <section className="bg-background/60 space-y-3 rounded-lg border p-3">
      <div>
        <h3 className="text-sm font-semibold">{t('graphPanel.workbench.merge.suggestions.title')}</h3>
        <p className="text-muted-foreground mt-1 text-xs">
          {t('graphPanel.workbench.merge.suggestions.description')}
        </p>
      </div>

      {errorMessage && <p className="text-xs text-red-600 dark:text-red-300">{errorMessage}</p>}
      {isLoading && (
        <p className="text-muted-foreground text-xs">
          {t('graphPanel.workbench.merge.suggestions.loading')}
        </p>
      )}

      {!isLoading && !candidates.length && !errorMessage && (
        <p className="text-muted-foreground text-xs">
          {t('graphPanel.workbench.merge.suggestions.empty')}
        </p>
      )}

      {!!candidates.length && (
        <div className="max-h-64 space-y-2 overflow-auto pr-1">
          {candidates.map((candidate) => {
            const isSelected = selectedTargets.includes(candidate.target_entity)
            return (
              <article
                key={`${candidate.target_entity}:${candidate.source_entities.join('|')}`}
                className={`rounded-md border p-2 ${isSelected ? 'border-primary/50 bg-primary/5' : 'bg-muted/20'}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-xs font-medium">
                      {candidate.source_entities.join(', ')} → {candidate.target_entity}
                    </p>
                    <p className="text-muted-foreground mt-1 text-[11px]">
                      {t('graphPanel.workbench.merge.suggestions.score', {
                        score: candidate.score.toFixed(2)
                      })}
                    </p>
                  </div>
                  <Button
                    type="button"
                    size="sm"
                    variant={isSelected ? 'default' : 'outline'}
                    className="h-7 px-2 text-[11px]"
                    onClick={() => onImportCandidate(candidate)}
                  >
                    {isSelected
                      ? t('graphPanel.workbench.merge.suggestions.actions.imported')
                      : t('graphPanel.workbench.merge.suggestions.actions.import')}
                  </Button>
                </div>
                <p className="text-muted-foreground mt-2 text-[11px]">
                  {t('graphPanel.workbench.merge.suggestions.evidence')}{' '}
                  {buildMergeCandidateEvidence(candidate, t)}
                </p>
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}

export default MergeSuggestionList
