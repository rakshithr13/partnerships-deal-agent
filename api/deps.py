"""Request-scoped dependencies: the shared Groq client and settings."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from groq import Groq

from api.config import Settings
from api.errors import UnsupportedModelError


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_groq_client(request: Request) -> Groq:
    return request.app.state.groq_client


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
GroqDep = Annotated[Groq, Depends(get_groq_client)]


def resolve_llm_model(requested: str | None, settings: Settings) -> str:
    if requested is None:
        return settings.groq_model
    if requested not in settings.allowed_models:
        raise UnsupportedModelError(
            f"Model {requested!r} is not permitted.",
            detail={"allowed_models": list(settings.allowed_models)},
        )
    return requested
