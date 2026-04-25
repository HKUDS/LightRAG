# LightRAG Enterprise Foundation Architecture

## Positioning

LightRAG is the knowledge layer, retrieval core, and memory engine. It is not
treated as the final CRM, help desk, identity platform, or multi-agent runtime.

The enterprise layer is implemented as `lightrag_enterprise/`, an adjacent
package that preserves:

- `LightRAG` lifecycle: `initialize_storages()` and `finalize_storages()`.
- Workspace isolation and namespace behavior.
- Existing storage abstractions in `lightrag/kg/`.
- Existing query modes: `local`, `global`, `hybrid`, `naive`, `mix`, `bypass`.
- Existing OpenAI-compatible, Ollama, Azure, Gemini, Bedrock, and other bindings.

## Decision

Selected architecture: modular monolith plus workers, model gateway, and agent
contracts.

This is the lowest-risk enterprise path for the current repo because the server,
WebUI, storage adapters, and core package already ship together. The extension
keeps deployment simple while allowing later extraction into services when load,
team ownership, or compliance boundaries justify it.

## Layers

```text
apps / admin console / CRM / internal chat
  -> lightrag_enterprise.admin optional routers
  -> lightrag_enterprise.agents and skills contracts
  -> lightrag_enterprise.security / audit / observability
  -> lightrag_enterprise.model_gateway
  -> lightrag.LightRAG retrieval core
  -> lightrag.kg storage adapters and lightrag.llm bindings
```

## Extension Modules

- `model_gateway/`: runtime model catalog, OpenRouter sync, routing profiles, cost estimates.
- `agents/` and `subagents/`: role declarations and allowed skill surfaces.
- `skills/`: explicit contracts and thin wrappers around LightRAG.
- `workflows/`: bounded planning, scorecards, critic rules, execution guardrails.
- `domain/crm/`: CRM records and repository contract.
- `domain/internal_chat/`: chat workspaces, channels, threads, messages, citations.
- `security/`: RBAC, ACL scope, PII masking, prompt-injection detection.
- `audit/`: structured audit event sink.
- `observability/`: metrics event recorder.
- `connectors/`: connector action/result contract.
- `jobs/`: catalog sync job.
- `admin/`: optional FastAPI router for model catalog and routing.
