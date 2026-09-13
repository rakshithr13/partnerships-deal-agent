"""Grid scaffolding: the steps the Streamlit UI performs between classification
and the P&L, exposed so an API client doesn't have to reimplement them."""

from __future__ import annotations

from fastapi import APIRouter

import business_case as bc
from api import adapters
from api.deps import GroqDep, SettingsDep, resolve_llm_model
from api.schemas import (
    GridAssumptionsRequest,
    GridAssumptionsResponse,
    GridPrefillRequest,
    GridPrefillResponse,
    GridRecomputeRequest,
    GridRecomputeResponse,
    GridTemplateRequest,
    GridTemplateResponse,
    RowSpecModel,
)

router = APIRouter(prefix="/v1/grid", tags=["grid"])


@router.post("/template", response_model=GridTemplateResponse)
def template(payload: GridTemplateRequest) -> GridTemplateResponse:
    """One-shot 'give me a working grid' call: row specs, an empty values/states
    matrix, and any cells pre-filled from the term sheet. No LLM call."""
    extraction = payload.extraction.model_dump()
    n_years = payload.n_years or bc.default_grid_years(extraction)
    symbol = bc.currency_symbol(payload.currency)

    row_specs = adapters.build_row_specs(
        payload.business_model,
        payload.license_model_type,
        symbol,
        [i.model_dump() for i in payload.opex_items] if payload.opex_items else None,
        payload.row_label_overrides,
    )
    values, states, prefill, blanks = adapters.init_grid(
        payload.business_model, extraction, row_specs, n_years, payload.include_prefill
    )

    return GridTemplateResponse(
        business_model=payload.business_model,
        license_model_type=payload.license_model_type,
        n_years=n_years,
        years=[f"Year {y}" for y in range(n_years + 1)],
        currency=payload.currency,
        currency_symbol=symbol,
        row_specs=[RowSpecModel(**r) for r in row_specs],
        values=values,
        states=states,
        prefill=prefill,
        blank_cells=blanks,
    )


@router.post("/prefill", response_model=GridPrefillResponse)
def prefill(payload: GridPrefillRequest) -> GridPrefillResponse:
    """Re-derive term-sheet prefill against row specs the client has already
    customised, without regenerating the grid and losing those edits."""
    return GridPrefillResponse(
        prefill=bc.prefill_grid_cells(
            payload.business_model,
            payload.extraction.model_dump(),
            [r.model_dump() for r in payload.row_specs],
            payload.n_years,
        )
    )


@router.post("/recompute", response_model=GridRecomputeResponse)
def recompute(payload: GridRecomputeRequest) -> GridRecomputeResponse:
    """Roll the subscription waterfall forward (ending_base = beginning + new - churn).

    /v1/business-case does this automatically; this endpoint exists for a client that
    wants to display a correct ending_base while the user is still editing.
    """
    values, states, computed = adapters.recompute_waterfall(
        payload.values, payload.states, payload.n_years
    )
    return GridRecomputeResponse(values=values, states=states, computed_rows=computed)


@router.post("/assumptions", response_model=GridAssumptionsResponse)
def assumptions(
    payload: GridAssumptionsRequest, groq: GroqDep, settings: SettingsDep
) -> GridAssumptionsResponse:
    """Propose a value for every still-blank cell and (by default) apply them."""
    model = resolve_llm_model(payload.llm_model, settings)
    proposals, values, states, notes, blanks = adapters.apply_assumptions(
        payload.business_model,
        payload.extraction.model_dump(),
        [r.model_dump() for r in payload.row_specs],
        payload.values,
        payload.states,
        payload.n_years,
        groq,
        model,
        payload.apply,
    )
    return GridAssumptionsResponse(
        assumptions=proposals,
        values=values,
        states=states,
        assumption_notes=notes,
        blank_cells=blanks,
        llm_model=model,
    )
