"""POST /v1/risks/triggers (deterministic) and /v1/risks/assess (LLM-backed)."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter

from api import adapters
from api.deps import GroqDep, SettingsDep, resolve_llm_model
from api.schemas import (
    AssessRiskRequest,
    AssessRiskResponse,
    TriggersRequest,
    TriggersResponse,
)

router = APIRouter(prefix="/v1/risks", tags=["risks"])


@router.post("/triggers", response_model=TriggersResponse)
def triggers(payload: TriggersRequest) -> TriggersResponse:
    """Which deterministic rules fire for this term sheet. No LLM call.

    Worth calling before /assess: the returned count is exactly how many serial
    LLM calls /assess will make, so it tells you what that request will cost.
    """
    found = adapters.run_triggers(payload.extraction.model_dump())
    return TriggersResponse(triggers=found, count=len(found))


@router.post(
    "/assess",
    response_model=AssessRiskResponse,
    description=(
        "Builds the full risk register. Makes ONE language-model call per fired "
        "trigger, serially, so a term sheet with several gaps can take 30-90s or "
        "more. Set a client read timeout of at least 180s. Call /v1/risks/triggers "
        "first to see how many calls that will be."
    ),
)
def assess(payload: AssessRiskRequest, groq: GroqDep, settings: SettingsDep) -> AssessRiskResponse:
    model = resolve_llm_model(payload.llm_model, settings)
    risks = adapters.run_risk_register(
        payload.extraction.model_dump(), groq, model, payload.strict
    )
    return AssessRiskResponse(
        risks=risks,
        count=len(risks),
        severity_counts=dict(Counter(r["severity"] for r in risks)),
        llm_model=model,
    )
