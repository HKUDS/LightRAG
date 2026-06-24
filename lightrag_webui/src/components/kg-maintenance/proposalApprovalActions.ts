import type { KBIterationProposalDecision } from '@/api/lightrag'
import type { ProposalSummary } from './kgMaintenanceData'

export function requestProposalRevisionFromPanel(
  proposal: ProposalSummary,
  onRequestRevision?: (proposal: ProposalSummary) => void | Promise<void>
) {
  return onRequestRevision?.(proposal)
}

export function acceptAllPendingProposalsFromPanel(
  proposals: ProposalSummary[],
  decisionStates: Record<string, KBIterationProposalDecision>,
  onAcceptAll?: (proposals: ProposalSummary[]) => void | Promise<void>
) {
  const pendingProposals = pendingApprovalProposals(proposals, decisionStates)
  if (!pendingProposals.length) return undefined
  return onAcceptAll?.(pendingProposals)
}

export function pendingApprovalProposals(
  proposals: ProposalSummary[],
  decisionStates: Record<string, KBIterationProposalDecision>
) {
  return proposals.filter((proposal) => proposal.requiresApproval && !decisionStates[proposal.id])
}
