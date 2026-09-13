"""POST /v1/business-case - the confirmed grid -> P&L, NPV and ROI."""

from __future__ import annotations

from fastapi import APIRouter

from api import adapters
from api.schemas import BusinessCaseRequest, BusinessCaseResponse

router = APIRouter(prefix="/v1", tags=["business-case"])


@router.post("/business-case", response_model=BusinessCaseResponse)
def business_case(payload: BusinessCaseRequest) -> BusinessCaseResponse:
    """Pure calculation - no LLM call. Build the grid with /v1/grid/* first.

    ``roi`` is null when total CAPEX + OPEX is 0 (ROI is undefined against a zero
    investment base); that is a normal 200, not an error.
    """
    result = adapters.build_business_case(
        payload.business_model,
        [r.model_dump() for r in payload.row_specs],
        payload.values,
        payload.n_years,
        payload.license_model_type,
        [t.model_dump() for t in payload.tier_breaks] if payload.tier_breaks else None,
        payload.discount_rate,
        payload.auto_recompute_waterfall,
    )
    return BusinessCaseResponse(**result)
