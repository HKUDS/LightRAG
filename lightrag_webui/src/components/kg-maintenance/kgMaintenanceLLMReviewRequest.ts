import type { KBIterationLLMReviewRunRequest } from '@/api/lightrag'

export function buildDefaultLLMReviewRequest(
  profile: string | null
): KBIterationLLMReviewRunRequest {
  return {
    profile,
    mode: 'agent_pipeline',
    max_stage_retries: 1,
    max_review_rounds: 4,
    max_focus_items_per_round: 3,
    max_subagent_tasks: 8,
    max_parallel_subagents: 4,
    max_subagent_issues_per_task: 4,
    max_subagent_proposals_per_task: 2,
    max_proposals_per_run: 200,
    deterministic_family_caps: {
      diagnosis: 40,
      treatment: 40,
      risk_safety: 35,
      prevention: 30,
      clinical_modeling: 30,
      entity_cleanup: 25,
      legacy_schema: 20
    },
    strict_subagent_role_contracts: true,
    prevalidate_action_candidates: true,
    require_candidate_evidence_allowlist: true,
    skip_deterministic_subagent_calls: true,
    allow_llm_judge: true,
    allow_llm_auto_accept: false,
    allow_low_risk_auto_reject: true,
    generate_patch_candidates: false,
    require_human_for_mutation: true
  }
}
