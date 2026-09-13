"""Partnership Agent HTTP API.

Run from the project root so both ``api`` and the flat backend modules import:

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import anyio.to_thread
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from groq import Groq

# Before Settings() is constructed, and before any backend module's os.getenv
# fallback could be consulted.
load_dotenv()

from api.config import get_settings  # noqa: E402
from api.errors import register_exception_handlers  # noqa: E402
from api.routers import (  # noqa: E402
    business_case,
    business_model,
    extract,
    grid,
    health,
    risks,
)

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Constructing Settings raises if GROQ_API_KEY is absent, which aborts startup
    # rather than letting every request fail with a confusing runtime error.
    settings = get_settings()
    anyio.to_thread.current_default_thread_limiter().total_tokens = settings.thread_limit

    app.state.settings = settings
    app.state.groq_client = Groq(
        api_key=settings.groq_api_key.get_secret_value(),
        timeout=httpx.Timeout(settings.groq_timeout_seconds, connect=10.0),
        max_retries=settings.groq_max_retries,
    )
    try:
        yield
    finally:
        app.state.groq_client.close()


app = FastAPI(
    title="Partnership Agent API",
    version="1.0.0",
    summary="Term sheet extraction, risk register, business-model classification "
            "and grid-driven P&L / NPV / ROI.",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


register_exception_handlers(app)

for module in (health, extract, risks, business_model, grid, business_case):
    app.include_router(module.router)
