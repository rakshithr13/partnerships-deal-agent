"""POST /v1/extract - term sheet .docx upload -> structured fields."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Annotated

from docx.opc.exceptions import OpcError
from fastapi import APIRouter, File, Form, UploadFile

from api import adapters
from api.deps import GroqDep, SettingsDep, resolve_llm_model
from api.errors import (
    EmptyFileError,
    FileTooLargeError,
    InvalidDocxError,
    UnsupportedFileTypeError,
)
from api.schemas import ExtractResponse

router = APIRouter(prefix="/v1", tags=["extract"])

_DOCX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
# Browsers, curl and Postman all disagree about the docx MIME type, so content-type
# is only a cheap early reject; the extension and the ZIP magic bytes are the real gate.
_ACCEPTED_CONTENT_TYPES = {
    _DOCX_CONTENT_TYPE, "application/octet-stream", "application/zip",
    "application/x-zip-compressed", "", None,
}
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_CHUNK = 64 * 1024


def _read_capped(upload: UploadFile, max_bytes: int) -> bytes:
    buf = io.BytesIO()
    size = 0
    while True:
        chunk = upload.file.read(_CHUNK)
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise FileTooLargeError(
                "Uploaded file exceeds the maximum allowed size.",
                detail={"max_bytes": max_bytes},
            )
        buf.write(chunk)
    return buf.getvalue()


@router.post("/extract", response_model=ExtractResponse)
def extract(
    groq: GroqDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Term sheet (.docx)")],
    llm_model: Annotated[str | None, Form()] = None,
) -> ExtractResponse:
    model = resolve_llm_model(llm_model, settings)

    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".docx":
        raise UnsupportedFileTypeError(
            "Only .docx term sheets are supported.",
            detail={"filename": filename, "accepted_extensions": [".docx"]},
        )
    if file.content_type not in _ACCEPTED_CONTENT_TYPES:
        raise UnsupportedFileTypeError(
            "Unexpected content type for a .docx upload.",
            detail={"content_type": file.content_type},
        )

    try:
        data = _read_capped(file, settings.max_upload_bytes)
    finally:
        file.file.close()

    if not data:
        raise EmptyFileError("Uploaded file is empty.")
    if not data.startswith(_ZIP_MAGIC):
        raise InvalidDocxError("Uploaded file is not a readable .docx document.")

    buffer = io.BytesIO(data)

    # Parse the document in its own narrow try. python-docx signals "not a readable
    # .docx" three different ways: PackageNotFoundError (an OpcError) for non-zip
    # bytes, BadZipFile for a corrupt archive, and -- for a valid ZIP that simply
    # isn't a Word file, e.g. a renamed .xlsx -- a bare KeyError from looking up a
    # main document part that isn't there. Keeping the LLM call outside this block
    # means a KeyError raised anywhere else still surfaces as a 500, not a bogus 400.
    try:
        has_text = adapters.document_has_text(buffer)
    except (OpcError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        # Never echo str(exc): PackageNotFoundError interpolates the "path", which for
        # a BytesIO is a meaningless object repr.
        raise InvalidDocxError("Uploaded file is not a readable .docx document.") from exc

    if not has_text:
        raise InvalidDocxError(
            "The document contains no extractable text.",
            detail={"filename": filename},
        )

    extraction = adapters.run_extraction(buffer, groq, model)

    return ExtractResponse(
        extraction=extraction,
        filename=filename,
        size_bytes=len(data),
        llm_model=model,
    )
