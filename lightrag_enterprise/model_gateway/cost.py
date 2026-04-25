from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .catalog import ModelCatalogEntry


@dataclass(frozen=True)
class CostEstimate:
    """Request cost estimate based only on runtime catalog prices."""

    input_cost: Decimal | None
    output_cost: Decimal | None
    request_cost: Decimal | None
    total_cost: Decimal | None
    currency: str = "openrouter_credits"


def estimate_request_cost(
    model: ModelCatalogEntry,
    *,
    input_tokens: int,
    output_tokens: int,
    image_units: int = 0,
) -> CostEstimate:
    """Estimate cost without hardcoding prices.

    OpenRouter catalog prices are supplied by the provider at sync time. If a
    component price is absent, the total remains unknown rather than guessed.
    """

    input_cost = (
        model.input_price * Decimal(input_tokens)
        if model.input_price is not None
        else None
    )
    output_cost = (
        model.output_price * Decimal(output_tokens)
        if model.output_price is not None
        else None
    )
    request_cost = model.request_price
    image_cost = (
        model.image_price * Decimal(image_units)
        if model.image_price is not None and image_units
        else Decimal("0")
    )
    components = [input_cost, output_cost, request_cost]
    total = None
    if all(component is not None for component in components):
        total = sum(components, Decimal("0")) + image_cost
    return CostEstimate(
        input_cost=input_cost,
        output_cost=output_cost,
        request_cost=request_cost,
        total_cost=total,
    )
