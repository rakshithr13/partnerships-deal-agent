"""Error envelope and the exception -> HTTP status mapping.

Every non-2xx response in this service is an ``ErrorResponse``, including FastAPI's
own 404s and validation errors, so clients only ever parse one shape.

Handler lookup walks the raised exception's MRO (see
``starlette/_exception_handler.py``), so ``MissingInputError`` resolves to its own
handler even though it subclasses ``FinancialEngineError``, regardless of the order
these are registered in. Both simply have to be registered.
"""

from __future__ import annotations

import logging
import zipfile
from typing import Any

import groq
from docx.opc.exceptions import OpcError
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from business_case import BusinessCaseError
from financial_engine import FinancialEngineError, MissingInputError
from risk_engine import RiskEngineError
from termsheet_extractor import ExtractionError

logger = logging.getLogger("api")


class ErrorBody(BaseModel):
    code: str
    message: str
    type: str
    detail: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str


class ApiError(Exception):
    """Base for errors raised by the API layer itself (not the backend)."""

    status_code = 400
    code = "api_error"

    def __init__(self, message: str, detail: Any | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class InvalidDocxError(ApiError):
    status_code = 400
    code = "invalid_docx"


class UnsupportedFileTypeError(ApiError):
    status_code = 400
    code = "unsupported_file_type"


class EmptyFileError(ApiError):
    status_code = 400
    code = "empty_file"


class FileTooLargeError(ApiError):
    status_code = 413
    code = "file_too_large"


class UnsupportedModelError(ApiError):
    status_code = 422
    code = "unsupported_model"


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _serializable_errors(errors: list[dict]) -> list[dict]:
    """Make Pydantic's error list JSON-safe.

    A ValueError raised inside a @model_validator comes back as
    ``ctx: {"error": ValueError(...)}`` -- an live exception object -- and ``input``
    can hold raw upload bytes. Either one makes JSONResponse blow up *inside* the
    handler, which turns a clean 422 into a 500.
    """
    cleaned = []
    for err in errors:
        item = {k: v for k, v in err.items() if k not in ("ctx", "input")}
        if "ctx" in err:
            item["ctx"] = {k: str(v) for k, v in err["ctx"].items()}
        if "input" in err:
            item["input"] = jsonable_encoder(
                err["input"], custom_encoder={bytes: lambda b: f"<{len(b)} bytes>"}
            )
        cleaned.append(item)
    return jsonable_encoder(cleaned)


def _envelope(
    request: Request,
    status: int,
    code: str,
    message: str,
    exc: BaseException,
    detail: Any | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, type=type(exc).__name__, detail=detail),
        request_id=_request_id(request),
    )
    # jsonable_encoder, not model_dump(): `detail` is typed Any, and a handler that
    # fails to serialize would itself become a 500.
    return JSONResponse(status_code=status, content=jsonable_encoder(body), headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    def _api_error(request: Request, exc: ApiError):
        return _envelope(request, exc.status_code, exc.code, exc.message, exc, exc.detail)

    @app.exception_handler(RequestValidationError)
    def _validation(request: Request, exc: RequestValidationError):
        return _envelope(
            request, 422, "validation_error", "Request failed validation.", exc,
            detail=_serializable_errors(exc.errors()),
        )

    @app.exception_handler(StarletteHTTPException)
    def _http(request: Request, exc: StarletteHTTPException):
        codes = {404: "not_found", 405: "method_not_allowed", 413: "file_too_large"}
        return _envelope(
            request, exc.status_code, codes.get(exc.status_code, "http_error"),
            str(exc.detail), exc, headers=getattr(exc, "headers", None),
        )

    # --- document parsing -------------------------------------------------
    # python-docx raises PackageNotFoundError for non-zip bytes, but a *valid* zip
    # that isn't a Word file (a renamed .xlsx, say) escapes as a bare ValueError
    # from docx/api.py. Both have to land on 400, not 500.
    @app.exception_handler(OpcError)
    def _opc(request: Request, exc: OpcError):
        return _envelope(
            request, 400, "invalid_docx",
            "Uploaded file is not a readable .docx document.", exc,
        )

    @app.exception_handler(zipfile.BadZipFile)
    def _badzip(request: Request, exc: zipfile.BadZipFile):
        return _envelope(
            request, 400, "invalid_docx",
            "Uploaded file is not a readable .docx document.", exc,
        )

    # --- backend domain errors -------------------------------------------
    @app.exception_handler(ExtractionError)
    def _extraction(request: Request, exc: ExtractionError):
        return _envelope(
            request, 422, "extraction_failed",
            "Could not extract structured terms from this document.", exc,
            detail=str(exc),
        )

    @app.exception_handler(RiskEngineError)
    def _risk(request: Request, exc: RiskEngineError):
        return _envelope(
            request, 502, "upstream_model_error",
            "The language model did not return a usable risk explanation.", exc,
        )

    @app.exception_handler(BusinessCaseError)
    def _business_case(request: Request, exc: BusinessCaseError):
        return _envelope(
            request, 502, "upstream_model_error",
            "The language model did not return a usable classification.", exc,
        )

    @app.exception_handler(MissingInputError)
    def _missing_input(request: Request, exc: MissingInputError):
        return _envelope(
            request, 422, "missing_input",
            "A required value was missing.", exc, detail=str(exc),
        )

    @app.exception_handler(FinancialEngineError)
    def _financial(request: Request, exc: FinancialEngineError):
        return _envelope(
            request, 422, "invalid_input", "A financial input was invalid.", exc,
            detail=str(exc),
        )

    # --- upstream LLM -----------------------------------------------------
    @app.exception_handler(groq.RateLimitError)
    def _rate_limited(request: Request, exc: groq.RateLimitError):
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after")
        return _envelope(
            request, 429, "upstream_rate_limited",
            "The language model provider is rate limiting this service.", exc,
            headers={"Retry-After": retry_after} if retry_after else None,
        )

    @app.exception_handler(groq.APITimeoutError)
    def _timeout(request: Request, exc: groq.APITimeoutError):
        return _envelope(
            request, 504, "upstream_timeout",
            "The language model provider did not respond in time.", exc,
        )

    @app.exception_handler(groq.APIConnectionError)
    def _conn(request: Request, exc: groq.APIConnectionError):
        return _envelope(
            request, 502, "upstream_unavailable",
            "Could not reach the language model provider.", exc,
        )

    # Our key / our request shape -- never the caller's fault, so never a 4xx.
    @app.exception_handler(groq.AuthenticationError)
    def _auth(request: Request, exc: groq.AuthenticationError):
        logger.error("Groq authentication failed (request_id=%s)", _request_id(request))
        return _envelope(
            request, 500, "config_error",
            "The service is misconfigured and could not authenticate upstream.", exc,
        )

    @app.exception_handler(groq.PermissionDeniedError)
    def _perm(request: Request, exc: groq.PermissionDeniedError):
        logger.error("Groq permission denied (request_id=%s)", _request_id(request))
        return _envelope(
            request, 500, "config_error",
            "The service is misconfigured and was denied upstream.", exc,
        )

    @app.exception_handler(groq.BadRequestError)
    def _bad_upstream_request(request: Request, exc: groq.BadRequestError):
        logger.exception("Malformed upstream request (request_id=%s)", _request_id(request))
        return _envelope(
            request, 500, "upstream_request_error",
            "The service built an invalid request to the language model.", exc,
        )

    @app.exception_handler(groq.APIError)
    def _api(request: Request, exc: groq.APIError):
        return _envelope(
            request, 502, "upstream_model_error",
            "The language model provider returned an error.", exc,
        )

    @app.exception_handler(Exception)
    def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error (request_id=%s)", _request_id(request))
        return _envelope(
            request, 500, "internal_error",
            "An internal error occurred.", exc,
        )
