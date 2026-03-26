import Button from '@/components/ui/Button'
import type { GraphMergeSuggestionCandidate, GraphMergeSuggestionReason } from '@/api/lightrag'

type MergeSuggestionListProps = {
  candidates: GraphMergeSuggestionCandidate[]
  selectedTargets?: string[]
  isLoading?: boolean
  errorMessage?: string | null
  onImportCandidate: (candidate: GraphMergeSuggestionCandidate) => void
}

const formatReason = (reason: GraphMergeSuggestionReason): string => {
  return `${reason.code} (${reason.score.toFixed(2)})`
}

export const buildMergeCandidateEvidence = (
  candidate: GraphMergeSuggestionCandidate
): string => {
  if (!candidate.reasons.length) {
    return 'No explicit evidence provided.'
  }
  return candidate.reasons.map(formatReason).join(', ')
}

const MergeSuggestionList = ({
  candidates,
  selectedTargets = [],
  isLoading = false,
  errorMessage = null,
  onImportCandidate
}: MergeSuggestionListProps) => {
  return (
    <section className="bg-background/60 space-y-3 rounded-lg border p-3">
      <div>
        <h3 className="text-sm font-semibold">Merge Suggestions</h3>
        <p className="text-muted-foreground mt-1 text-xs">
          Candidate list from current query scope. Import once to prefill merge form.
        </p>
      </div>

      {errorMessage && <p className="text-xs text-red-600 dark:text-red-300">{errorMessage}</p>}
      {isLoading && <p className="text-muted-foreground text-xs">Loading suggestions...</p>}

      {!isLoading && !candidates.length && !errorMessage && (
        <p className="text-muted-foreground text-xs">
          No candidates loaded yet. Click &quot;Load Suggestions&quot; to fetch candidates.
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
                      Score: {candidate.score.toFixed(2)}
                    </p>
                  </div>
                  <Button
                    type="button"
                    size="sm"
                    variant={isSelected ? 'default' : 'outline'}
                    className="h-7 px-2 text-[11px]"
                    onClick={() => onImportCandidate(candidate)}
                  >
                    {isSelected ? 'Imported' : 'Import'}
                  </Button>
                </div>
                <p className="text-muted-foreground mt-2 text-[11px]">
                  Evidence: {buildMergeCandidateEvidence(candidate)}
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
