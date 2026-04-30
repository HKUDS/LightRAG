from decimal import Decimal

from lightrag_enterprise.model_gateway.catalog import (
    ModelCatalog,
    ModelCatalogEntry,
    ModelProfile,
)
from lightrag_enterprise.model_gateway.cost import estimate_request_cost
from lightrag_enterprise.model_gateway.policy import (
    ModelPolicy,
    ModelRoutingContext,
    PolicyModelRouter,
)


OPENROUTER_SAMPLE = {
    "id": "google/gemini-example",
    "canonical_slug": "google/gemini-example-001",
    "name": "Google: Gemini Example",
    "context_length": 128000,
    "architecture": {
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
    },
    "pricing": {
        "prompt": "0.000001",
        "completion": "0.000002",
        "request": "0",
        "image": "0.00001",
    },
    "top_provider": {"context_length": 128000, "is_moderated": True},
    "supported_parameters": ["tools", "response_format", "structured_outputs"],
}


def test_openrouter_catalog_entry_normalizes_runtime_metadata():
    entry = ModelCatalogEntry.from_openrouter_model(OPENROUTER_SAMPLE)

    assert entry.model_id == "google/gemini-example"
    assert entry.slug == "google/gemini-example-001"
    assert entry.provider == "google"
    assert entry.family == "gemini"
    assert entry.context_window == 128000
    assert entry.tool_calling is True
    assert entry.structured_output is True
    assert entry.input_price == Decimal("0.000001")
    assert entry.image_price == Decimal("0.00001")


def test_policy_router_blocks_hosted_for_private_data_and_uses_local():
    catalog = ModelCatalog(
        entries=[
            ModelCatalogEntry.from_openrouter_model(OPENROUTER_SAMPLE),
            ModelCatalogEntry.local(
                "ollama/qwen-local",
                family="qwen",
                context_window=32768,
                capabilities={"tools"},
            ),
        ]
    )
    router = PolicyModelRouter(catalog, ModelPolicy())

    decision = router.route(
        ModelRoutingContext(
            tenant_id="t1",
            workspace="private_ws",
            purpose="internal_rag",
            contains_private_data=True,
            requires_tools=True,
        )
    )

    assert decision.allowed is True
    assert decision.profile == ModelProfile.LOCAL_PRIVATE
    assert decision.model is not None
    assert decision.model.provider == "local"


def test_policy_router_blocks_hosted_only_catalog_for_private_data():
    catalog = ModelCatalog(entries=[ModelCatalogEntry.from_openrouter_model(OPENROUTER_SAMPLE)])
    router = PolicyModelRouter(catalog, ModelPolicy())

    decision = router.route(
        ModelRoutingContext(
            tenant_id="t1",
            workspace="private_ws",
            purpose="internal_rag",
            contains_private_data=True,
            requested_profile=ModelProfile.LOCAL_PRIVATE,
        )
    )

    assert decision.allowed is False
    assert decision.model is None
    assert "No permitted model" in decision.reason


def test_cost_estimate_uses_catalog_prices_only():
    entry = ModelCatalogEntry.from_openrouter_model(OPENROUTER_SAMPLE)

    estimate = estimate_request_cost(entry, input_tokens=1000, output_tokens=500)

    assert estimate.input_cost == Decimal("0.001000")
    assert estimate.output_cost == Decimal("0.001000")
    assert estimate.total_cost == Decimal("0.002000")
