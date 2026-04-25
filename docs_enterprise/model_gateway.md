# Model Gateway

## Runtime Catalog

The gateway never hardcodes a fixed model list. OpenRouter models are synced at
runtime via:

- `GET https://openrouter.ai/api/v1/models/user` when `OPENROUTER_API_KEY` is set.
- Fallback to `GET https://openrouter.ai/api/v1/models` when account-scoped sync is unavailable.

The normalized catalog stores:

- model id and canonical slug
- provider/lab and inferred family
- context window
- modalities and capabilities
- tool calling and structured output support
- input, output, request, and image prices as runtime catalog values
- privacy flags and sync timestamp

## Profiles

- `local_private`: local/offline/private models only.
- `cheap_high_volume`: lowest available runtime price among permitted models.
- `balanced_general`: broader context and general reliability.
- `premium_reasoning`: reasoning-capable or highest-context models when policy permits.

## Escalation Policy

Default escalation order:

```text
local_private -> cheap_high_volume -> balanced_general -> premium_reasoning
```

Hosted models are blocked when `contains_private_data=true` and the active policy
requires private routing.

## Job

```bash
uv run python -m lightrag_enterprise.jobs.sync_openrouter_catalog \
  --output rag_storage/model_catalog/openrouter_catalog.json
```
