# Superpowers Work Notes

This directory keeps only active LightRAG KG iteration planning material.

Current source of truth:

- `docs/KBIterationAgent.md`
- `docs/KBIterationAgent-zh.md`
- `docs/tutorials/kb-iteration-agent-tutorial-zh.md`
- `docs/tutorials/kb-iteration-agent-proposal-approval-zh.md`
- `docs/superpowers/plans/2026-06-21-kg-agent-multi-agent-proposal-orchestrator-implementation.md`
- `docs/superpowers/plans/2026-06-23-kg-agent-proposal-funnel-quality-implementation.md`
- `docs/superpowers/plans/2026-06-23-kg-agent-proposal-funnel-quality-v3-implementation.md`
- `docs/superpowers/plans/2026-06-23-kg-proposal-yield-p1-implementation.md`

Historical one-off plans and design drafts were removed after the multi-agent proposal orchestrator and proposal-funnel plans replaced the earlier KG maintenance Agent architecture.

Latest maintained status is in `docs/KBIterationAgent.md` and `docs/KBIterationAgent-zh.md`. As of 2026-06-24, the current engineering follow-up is fingerprint-based decision memory so rejected semantic duplicates cannot re-enter the approval queue with modified proposal IDs.
