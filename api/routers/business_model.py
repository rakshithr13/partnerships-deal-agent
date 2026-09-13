"""POST /v1/business-model/classify and /signals."""

from __future__ import annotations

from fastapi import APIRouter

from api import adapters
from api.deps import GroqDep, SettingsDep, resolve_llm_model
from api.schemas import (
    ClassificationModel,
    ClassifyRequest,
    ClassifyResponse,
    SignalsModel,
    SignalsRequest,
)

router = APIRouter(prefix="/v1/business-model", tags=["business-model"])


@router.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest, groq: GroqDep, settings: SettingsDep) -> ClassifyResponse:
    model = resolve_llm_model(payload.llm_model, settings)
    result = adapters.run_classification(
        payload.extraction.model_dump(),
        [r.model_dump() for r in payload.risks],
        groq,
        model,
    )
    return ClassifyResponse(classification=ClassificationModel(**result), llm_model=model)


@router.post("/signals", response_model=SignalsModel)
def signals(payload: SignalsRequest) -> SignalsModel:
    """Deterministic check that the contract language actually supports a model.

    This is the guardrail behind a user override: if a caller picks a model the term
    sheet has no language for, ``supported`` is False and any resulting numbers are
    hypothetical rather than grounded in the contract. No LLM call.
    """
    return SignalsModel(
        **adapters.run_signals(payload.business_model, payload.extraction.model_dump())
    )
