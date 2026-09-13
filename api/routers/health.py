"""Liveness and static vocabulary."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

import business_case as bc
from api.deps import SettingsDep
from api.schemas import HealthResponse, MetaResponse, RowSpecModel

router = APIRouter(tags=["meta"])


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send the bare base URL somewhere useful instead of a bare 404."""
    return RedirectResponse(url="/docs")


@router.get("/healthz", response_model=HealthResponse)
def healthz(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        groq_configured=bool(settings.groq_api_key.get_secret_value()),
    )


@router.get("/v1/meta", response_model=MetaResponse)
def meta(settings: SettingsDep) -> MetaResponse:
    """Every vocabulary a client needs to build a grid without hardcoding it.

    Row labels here still contain the ``{cur}`` placeholder; it is substituted per
    request against the chosen currency.
    """
    return MetaResponse(
        business_models=list(bc.BUSINESS_MODELS),
        model_labels=dict(bc.MODEL_LABELS),
        currency_choices=list(bc.CURRENCY_CHOICES),
        currency_symbols=dict(bc.CURRENCY_SYMBOLS),
        license_model_types=["flat", "per_unit", "tiered"],
        cell_states=["prefilled", "user", "assumed", "computed", "blank"],
        row_kinds=["volume", "churn", "rate", "capex", "opex", "seed", "computed"],
        grid_rows={
            model: [RowSpecModel(**row) for row in rows]
            for model, rows in bc.MODEL_GRID_ROWS.items()
        },
        default_opex_label=bc.DEFAULT_OPEX_LABEL,
        default_llm_model=settings.groq_model,
        max_grid_years=settings.max_grid_years,
    )
