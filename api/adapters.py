"""The seam between HTTP and the backend modules.

This is the only module in ``api`` that imports the backend, and the only place
that knows about the backend's rough edges: the pandas DataFrame in
``build_pl_from_grid``'s return, ``recompute_subscription_waterfall`` mutating its
argument in place, and the row-spec assembly that otherwise lives in the Streamlit
layer.
"""

from __future__ import annotations

import copy
import io
import re
from typing import Any

from groq import Groq

import business_case as bc
from risk_engine import build_risk_register, detect_triggers
from termsheet_extractor import extract_docx_content, extract_termsheet

_LICENSING_RATE_LABELS = {
    "flat": "Annual flat licence fee ({cur})",
    "per_unit": "Royalty per licensed unit ({cur})",
    "tiered": "Royalty per licensed unit ({cur})",
}


def _opex_slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "opex"


# MIRRORS Partnerships_agent.py::_bc_grid_row_specs (lines 238-256) - keep in sync.
# A drift here does not raise: build_pl_from_grid's _grid_row zero-fills an unknown
# row name, which silently produces a wrong P&L rather than an error.
def build_row_specs(
    business_model: str,
    license_model_type: str | None = None,
    currency_symbol: str = "$",
    opex_items: list[dict] | None = None,
    label_overrides: dict[str, str] | None = None,
) -> list[dict]:
    rows = [dict(r) for r in bc.MODEL_GRID_ROWS[business_model]]

    if business_model == "licensing":
        label = _LICENSING_RATE_LABELS.get(license_model_type or "flat")
        for r in rows:
            if r["name"] == "rate_or_fee":
                r["label"] = label

    rows.append(dict(bc.GRID_CAPEX_ROW))

    if opex_items is None:
        opex_items = [{"key": "opex__0__opex", "label": bc.DEFAULT_OPEX_LABEL}]
    for index, item in enumerate(opex_items):
        key = item.get("key") or f"opex__{index}__{_opex_slug(item['label'])}"
        rows.append({"name": key, "label": item["label"], "kind": "opex"})

    for r in rows:
        r["label"] = bc.apply_currency_label(r["label"], currency_symbol)

    for r in rows:
        if label_overrides and r["name"] in label_overrides:
            r["label"] = label_overrides[r["name"]]

    names = [r["name"] for r in rows]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate grid row name(s): {sorted(duplicates)}")

    return rows


def _mark_subscription_computed(states: dict, n_years: int) -> None:
    for row in ("beginning_base", "ending_base"):
        states.setdefault(row, ["blank"] * (n_years + 1))
    for y in range(1, n_years + 1):
        states["beginning_base"][y] = "computed"
    for y in range(n_years + 1):
        states["ending_base"][y] = "computed"


def _blank_cells(states: dict) -> list[dict]:
    return [
        {"row": row, "year": year}
        for row, series in states.items()
        for year, state in enumerate(series)
        if state == "blank"
    ]


def recompute_waterfall(
    values: dict, states: dict | None, n_years: int
) -> tuple[dict, dict, list[str]]:
    """Copy-in/copy-out wrapper: the backend function returns None and mutates."""
    values = copy.deepcopy(values)
    states = copy.deepcopy(states) if states is not None else {}
    bc.recompute_subscription_waterfall(values, n_years)
    for row in ("beginning_base", "ending_base"):
        states.setdefault(row, ["blank"] * (n_years + 1))
    _mark_subscription_computed(states, n_years)
    return values, states, ["beginning_base", "ending_base"]


def run_extraction(buffer: io.BytesIO, groq_client: Groq, model: str) -> dict:
    return extract_termsheet(buffer, groq_client=groq_client, model=model)


def document_has_text(buffer: io.BytesIO) -> bool:
    has_text = bool(extract_docx_content(buffer).strip())
    buffer.seek(0)
    return has_text


def run_triggers(extraction: dict) -> list[dict]:
    return detect_triggers(extraction)


def run_risk_register(
    extraction: dict, groq_client: Groq, model: str, strict: bool
) -> list[dict]:
    return build_risk_register(extraction, groq_client=groq_client, model=model, strict=strict)


def run_classification(
    extraction: dict, risks: list[dict], groq_client: Groq, model: str
) -> dict:
    return bc.classify_business_model(extraction, risks, groq_client, model=model)


def run_signals(business_model: str, extraction: dict) -> dict:
    return bc.extraction_signals_for_model(business_model, extraction)


def init_grid(
    business_model: str,
    extraction: dict,
    row_specs: list[dict],
    n_years: int,
    include_prefill: bool = True,
) -> tuple[dict, dict, dict, list[dict]]:
    values = {r["name"]: [None] * (n_years + 1) for r in row_specs}
    states = {r["name"]: ["blank"] * (n_years + 1) for r in row_specs}

    prefill: dict = {}
    if include_prefill:
        prefill = bc.prefill_grid_cells(business_model, extraction, row_specs, n_years)
        for row, cells in prefill.items():
            for year, info in cells.items():
                values[row][year] = info["value"]
                states[row][year] = "prefilled"

    if business_model == "subscription":
        bc.recompute_subscription_waterfall(values, n_years)
        _mark_subscription_computed(states, n_years)

    return values, states, prefill, _blank_cells(states)


def apply_assumptions(
    business_model: str,
    extraction: dict,
    row_specs: list[dict],
    values: dict,
    states: dict,
    n_years: int,
    groq_client: Groq,
    model: str,
    apply: bool = True,
) -> tuple[dict, dict, dict, dict, list[dict]]:
    values = copy.deepcopy(values)
    states = copy.deepcopy(states)

    proposals = bc.propose_grid_assumptions(
        business_model, extraction, row_specs, values, states, n_years,
        groq_client, model=model,
    )

    notes: dict[str, str] = {}
    if apply:
        for row, cells in proposals.items():
            for year, info in cells.items():
                values[row][year] = info["value"]
                states[row][year] = "assumed"
                notes[f"{row}|{year}"] = info["rationale"]

        # new_added / churned_out may have just been filled, so ending_base is stale.
        # The UI gets away without this because _bc_grid_ensure re-runs on the next
        # Streamlit rerun; an API request has no "next rerun".
        if business_model == "subscription":
            bc.recompute_subscription_waterfall(values, n_years)
            _mark_subscription_computed(states, n_years)

    return proposals, values, states, notes, _blank_cells(states)


def build_business_case(
    business_model: str,
    row_specs: list[dict],
    values: dict,
    n_years: int,
    license_model_type: str | None,
    tier_breaks: list[dict] | None,
    discount_rate: float,
    auto_recompute_waterfall: bool = True,
) -> dict[str, Any]:
    values = copy.deepcopy(values)

    if business_model == "subscription" and auto_recompute_waterfall:
        bc.recompute_subscription_waterfall(values, n_years)

    # build_pl_from_grid raises MissingInputError naming any blank cell it needs, which
    # the API surfaces as a 422 -- no pre-scan required, and no silently-zeroed P&L.
    warnings: list[str] = []

    pl = bc.build_pl_from_grid(
        business_model, row_specs, values, n_years,
        license_model_type=license_model_type,
        tier_breaks=tier_breaks or None,
    )
    npv_roi = bc.compute_npv_roi(pl, discount_rate=discount_rate)

    # Gross Profit and Cumulative Cash Flow exist ONLY as DataFrame rows -- they are
    # not in the returned dict -- so read them out before discarding the frame.
    df = pl["df"]
    table = {
        "revenue": [float(v) for v in pl["revenue_by_year"]],
        "capex": [float(v) for v in pl["capex_by_year"]],
        "opex": [float(v) for v in pl["opex_by_year"]],
        "gross_profit": [float(v) for v in df.loc["Gross Profit"].tolist()],
        "net_profit": [float(v) for v in pl["cash_flows"]],
        "cumulative_cash_flow": [float(v) for v in df.loc["Cumulative Cash Flow"].tolist()],
    }

    return {
        "business_model": pl["business_model"],
        "license_model_type": license_model_type,
        "projection_years": pl["projection_years"],
        "years": [str(c) for c in df.columns],
        "table": table,
        "cash_flows": [float(v) for v in pl["cash_flows"]],
        "total_capex": float(pl["total_capex"]),
        "total_opex": float(pl["total_opex"]),
        "final_cumulative": float(pl["final_cumulative"]),
        "npv": npv_roi["npv"],
        "roi": npv_roi["roi"],
        "discount_rate": npv_roi["discount_rate"],
        "investment_base": float(npv_roi["investment_base"]),
        "notes": list(pl["notes"]),
        "warnings": warnings,
    }
