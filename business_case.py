"""
Business-case flow helpers.

Non-UI logic for the conversational business-case builder:

1. classify_business_model()          - Groq picks one of the four models + why.
   extraction_signals_for_model()     - deterministic check that a chosen/overridden
                                        model is actually supported by the contract text.
2. MODEL_GRID_ROWS / prefill_grid_cells / propose_grid_assumptions
                                       - the Step 2 year-by-year input grid: which rows
                                        a model needs, deterministic pre-fill from the
                                        term sheet, and Groq-drafted (ramped, not flat)
                                        assumptions for whatever's still blank.
3. build_pl_from_grid()               - per-year Revenue/CAPEX/OPEX straight from the
                                        confirmed grid, run through calculate_profit.
4. compute_npv_roi()                  - NPV/ROI from a built P&L (build_pl_from_grid's
                                        output).

No Streamlit import. The Streamlit tab owns all session-state and widgets and calls
into these. All period conventions are annual, matching financial_engine.
"""

from __future__ import annotations

import json

import pandas as pd

from financial_engine import (
    MissingInputError,
    calculate_licensing_cost,
    calculate_npv,
    calculate_profit,
    calculate_roi,
    derive_licensing_inputs_from_extraction,
)

BUSINESS_MODELS = ("subscription", "adtech", "fee_based", "licensing")

MODEL_LABELS = {
    "subscription": "Subscription — recurring per-customer fee",
    "adtech": "Ad-tech — CPM / CPC media revenue",
    "fee_based": "Fee-based — flat fee per exposure unit",
    "licensing": "Licensing — software / IP licence paid to a provider",
}

DEFAULT_MODEL = "openai/gpt-oss-120b"

# --------------------------------------------------------------------------- #
# Currency
# --------------------------------------------------------------------------- #

CURRENCY_CHOICES = ("USD", "EUR", "GBP", "Other")

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}


def currency_symbol(code: str) -> str:
    """
    Display prefix for a currency code - concatenate directly before a number.

    Known codes (USD/EUR/GBP) return their symbol ("$", "€", "£"); anything
    else (a user-typed code like "INR" or "AUD") returns "<CODE> " so the
    amount reads e.g. "INR 1,234" rather than guessing a symbol.
    """
    if not code:
        return "$"
    code = code.strip().upper()
    return CURRENCY_SYMBOLS.get(code, f"{code} ")


class BusinessCaseError(Exception):
    """Raised when a business-case step cannot complete."""


# --------------------------------------------------------------------------- #
# 1. Classification
# --------------------------------------------------------------------------- #

_CLASSIFY_PROMPT = """You are a partnerships analyst. Classify the PRIMARY business model of this \
deal as exactly one of:

- "subscription": the partner earns a recurring per-customer/per-seat fee (often with activation and churn).
- "adtech": the partner earns media revenue priced on impressions (CPM) and/or clicks (CPC).
- "fee_based": the partner earns a flat fee per exposure unit (per unit, per transaction, per API call, per device, per shipment) with no CPM/CPC and no recurring-subscriber mechanic.
- "licensing": one party pays another a software / IP licence fee (flat, per-unit royalty, or tiered) - e.g. one company licensing another's software or hardware platform.

Also identify the deal's currency and whether it looks cross-border (the provider and the \
counterparty appear to sit in different currency regions - e.g. different currency symbols/codes \
used in the text, or party names/context implying different home countries).

Extracted term sheet (JSON):
{extraction_json}

Deterministically-flagged risks (names only):
{risk_names}

Return ONLY a JSON object with exactly these keys:
- "business_model": one of "subscription", "adtech", "fee_based", "licensing".
- "reasoning": 1-3 sentences citing the specific term-sheet language (revenue share, fees, volume basis, IP terms) that drove the choice.
- "signals": array of short strings - the concrete phrases/fields that point to this model.
- "confidence": "high", "medium", or "low".
- "currency": ISO 4217 code (e.g. "USD", "EUR", "GBP") the money amounts are stated in, or null if the \
term sheet never states or implies a currency.
- "cross_border": true or false - true only if you can point to actual evidence the two parties are in \
different currency regions.
- "counterparty_currency": if cross_border is true, the OTHER party's likely ISO 4217 currency code; \
else null.
- "currency_reasoning": one short sentence backing the currency/cross_border call, or "" if "currency" \
is null and cross_border is false.

No markdown, no code fences, no extra keys.
"""


def _groq_json(groq_client, prompt, model, *, attempts=2):
    """Call Groq for a JSON object response, tolerating one bad parse."""
    last = None
    for _ in range(attempts):
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        last = resp.choices[0].message.content
        try:
            parsed = json.loads(last)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    raise BusinessCaseError(f"Groq did not return a usable JSON object. Last response:\n{last}")


def classify_business_model(extraction: dict, risks: list, groq_client, model: str = DEFAULT_MODEL) -> dict:
    """
    Ask Groq to classify the deal as one of BUSINESS_MODELS, and - in the same
    call - detect the deal's stated currency and whether it looks cross-border.

    Returns {"business_model", "reasoning", "signals": [...], "confidence",
    "currency": str|None, "cross_border": bool, "counterparty_currency": str|None,
    "currency_reasoning": str}.
    Raises BusinessCaseError if the model returns an unusable answer.
    """
    risk_names = [r.get("risk_name", "") for r in (risks or [])]
    prompt = _CLASSIFY_PROMPT.format(
        extraction_json=json.dumps(extraction, indent=2),
        risk_names="\n".join(f"- {n}" for n in risk_names) or "(none)",
    )
    parsed = _groq_json(groq_client, prompt, model)

    bm = str(parsed.get("business_model", "")).strip().lower()
    if bm not in BUSINESS_MODELS:
        raise BusinessCaseError(f"Groq returned an unrecognised business_model: {bm!r}")

    signals = parsed.get("signals") or []
    if not isinstance(signals, list):
        signals = [str(signals)]

    currency = parsed.get("currency")
    currency = str(currency).strip().upper() if isinstance(currency, str) and currency.strip() else None

    counterparty_currency = parsed.get("counterparty_currency")
    counterparty_currency = (
        str(counterparty_currency).strip().upper()
        if isinstance(counterparty_currency, str) and counterparty_currency.strip() else None
    )

    return {
        "business_model": bm,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "signals": [str(s).strip() for s in signals if str(s).strip()],
        "confidence": str(parsed.get("confidence", "")).strip().lower() or "unknown",
        "currency": currency,
        "cross_border": bool(parsed.get("cross_border")),
        "counterparty_currency": counterparty_currency,
        "currency_reasoning": str(parsed.get("currency_reasoning", "")).strip(),
    }


# --------------------------------------------------------------------------- #
# 1b. Does the contract language actually support a given model? (deterministic)
# --------------------------------------------------------------------------- #
#
# Used to sanity-check a *user override*: if the user picks a model the term sheet
# has no language for, the resulting numbers are hypothetical, not grounded.
# Substring match against the lower-cased free-text fields of the extraction.

_MODEL_SIGNAL_KEYWORDS = {
    "subscription": [
        "subscription", "subscriber", "churn", "activation rate", "recurring",
        "monthly fee", "monthly subscription", "per active", "per seat", "per-seat",
        "per user", "per-user", "mrr", "arr", "retention",
    ],
    "adtech": [
        "cpm", "cpc", "impression", "click", "click-through", "clickthrough", "ctr",
        "ad inventory", "ad-inventory", "advertising", "advert", "programmatic",
        "fill rate", "per thousand", "per mille", "media revenue",
    ],
    "fee_based": [
        "flat fee", "fee per", "per-unit fee", "per unit fee", "per unit sold",
        "per transaction", "per device", "per api", "per call", "per connected",
        "per shipment", "per order", "per record", "per query", "per install",
        "usage fee", "usage-based", "metered", "per activation fee",
    ],
    "licensing": [
        "licens", "royalt", "intellectual property", "ip ownership", "sublicens",
        "oem", "platform fee", "platform license", "platform licence",
    ],
}

MODEL_SIGNAL_HINT = {
    "subscription": "subscriber / recurring-fee / churn / activation language",
    "adtech": "CPM / CPC / impressions / clicks / ad-inventory language",
    "fee_based": "flat fee-per-unit language (per unit / transaction / device / call)",
    "licensing": "licence / royalty / IP-ownership language",
}

_SIGNAL_TEXT_FIELDS = (
    "revenue_share", "revenue_share_details", "minimum_volume", "payment_terms",
    "renewal_terms", "termination_terms", "exclusivity", "IP_Ownership",
)


def extraction_signals_for_model(business_model: str, extraction: dict) -> dict:
    """
    Deterministic check for term-sheet language that supports ``business_model``.

    Scans the free-text fields of ``extraction`` (lower-cased) for the model's
    keyword list.

    Returns
    -------
    dict with:
        business_model : str
        supported      : bool          True if any keyword was found
        matched        : list[str]     the keywords/phrases found (sorted, unique)
        hint           : str           human description of what was looked for
    """
    haystack = " ".join(
        str(extraction.get(f) or "") for f in _SIGNAL_TEXT_FIELDS
    ).lower().replace("‑", "-").replace("–", "-").replace("—", "-")

    matched = sorted({
        kw for kw in _MODEL_SIGNAL_KEYWORDS.get(business_model, []) if kw in haystack
    })
    return {
        "business_model": business_model,
        "supported": bool(matched),
        "matched": matched,
        "hint": MODEL_SIGNAL_HINT.get(business_model, ""),
    }


# --------------------------------------------------------------------------- #
# 2. Year-by-year input grid (Step 2)
# --------------------------------------------------------------------------- #
#
# Step 2 of the Streamlit flow gathers inputs as an editable Year 0..N grid
# (st.data_editor) instead of one flat number per field. Each row is one driver;
# each cell tracks its own state: "prefilled" (from the term sheet), "user"
# (typed in), "assumed" (accepted industry-standard proposal), "computed"
# (rolled forward automatically - the subscription waterfall), or "blank".
#
# kind drives both the fallback ramp heuristic and (for subscription) whether the
# row is user-editable at all:
#   "volume" | "churn" | "rate" | "capex" | "opex"  - normal editable/assumable rows
#   "seed"                                           - editable ONLY at Year 0;
#                                                       later years roll forward
#   "computed"                                       - never editable, never assumed

MODEL_GRID_ROWS = {
    "subscription": [
        {"name": "beginning_base", "label": "Beginning subscriber base", "kind": "seed",
         "help": "Active subscribers at the start of the year. Year 0 is the deal's "
                 "starting point (0 for a brand-new partnership); later years roll "
                 "forward automatically from the prior year's ending base."},
        {"name": "new_added", "label": "New subscribers added", "kind": "volume",
         "prefill": "annual_volume", "help": "New subscribers who start in the year."},
        {"name": "churned_out", "label": "Subscribers churned out", "kind": "churn",
         "help": "Active subscribers lost during the year."},
        {"name": "ending_base", "label": "Ending subscriber base", "kind": "computed",
         "help": "= Beginning base + New subscribers - Churned out. Computed automatically."},
        {"name": "price", "label": "Annual fee per subscriber ({cur})", "kind": "rate",
         "help": "What the partner earns per active subscriber per year."},
    ],
    "adtech": [
        {"name": "annual_impressions", "label": "Ad impressions", "kind": "volume"},
        {"name": "cpm_rate", "label": "CPM rate ({cur} / 1,000 impressions)", "kind": "rate"},
        {"name": "annual_clicks", "label": "Clicks", "kind": "volume"},
        {"name": "cpc_rate", "label": "CPC rate ({cur} / click)", "kind": "rate"},
    ],
    "fee_based": [
        {"name": "annual_exposure_units", "label": "Annual unit volume (units / transactions / devices)",
         "kind": "volume", "prefill": "annual_volume",
         "help": "Billable units - transactions, devices, API calls, shipments - in the year."},
        {"name": "rate_per_unit", "label": "Fee per unit ({cur})", "kind": "rate"},
    ],
    "licensing": [
        # Always present, even for a flat-fee structure with no per-unit math -
        # a licence deal still has a volume of licensed units worth tracking.
        {"name": "volume", "label": "Licensed volume", "kind": "volume", "prefill": "annual_volume"},
        {"name": "rate_or_fee", "label": "Royalty per licensed unit ({cur})", "kind": "rate"},
    ],
}

GRID_CAPEX_ROW = {"name": "capex", "label": "Upfront / setup cost — CAPEX ({cur})", "kind": "capex",
                  "help": "One-time build / integration cost recognised in that year "
                          "(e.g. tooling, integration work, platform setup). This is the "
                          "deal's CAPEX line."}

DEFAULT_OPEX_LABEL = "Ongoing operating cost — OPEX ({cur})"


def apply_currency_label(label: str, symbol: str) -> str:
    """Substitute the "{cur}" placeholder in a default row label with the
    actual currency prefix (e.g. "$", "EUR "). Labels without the placeholder
    (counts, and any user-renamed label) pass through unchanged."""
    return label.replace("{cur}", symbol.strip()) if "{cur}" in label else label


def default_grid_years(extraction: dict) -> int:
    """Whole-year projection horizon default: the contract term, else 3 years."""
    derived = derive_licensing_inputs_from_extraction(extraction)
    dur = derived.get("duration_years")
    if dur:
        return max(1, int(round(dur)))
    return 3


def prefill_grid_cells(business_model: str, extraction: dict, row_specs: list, n_years: int) -> dict:
    """
    Deterministic pre-fill of grid cells from the term sheet extraction.

    Only the row(s) tagged ``prefill="annual_volume"`` get filled, spreading the
    term sheet's per-year minimum-volume figure evenly across Years 1..N (Year 0
    is left blank - it's the setup year, before the commitment starts). Nothing
    else is guessed here; that's what the assumption engine is for.

    Returns ``{row_name: {year_idx: {"value": float, "source": str}}}``.
    """
    derived = derive_licensing_inputs_from_extraction(extraction)
    per_year = derived.get("volume_per_year")
    src_text = derived.get("volume_source_text")

    out = {}
    if per_year is None:
        return out
    for spec in row_specs:
        if spec.get("prefill") == "annual_volume":
            out[spec["name"]] = {
                y: {"value": per_year,
                    "source": f"term sheet minimum_volume: {src_text!r} -> {per_year:g}/year"}
                for y in range(1, n_years + 1)
            }
    return out


def recompute_subscription_waterfall(values: dict, n_years: int) -> None:
    """
    In-place roll-forward of the subscription waterfall.

    ``beginning_base[0]`` is whatever the caller has (0 if unset); every later
    year's beginning base is the prior year's ending base. Ending base is always
    ``beginning + new_added - churned_out`` (missing new/churn treated as 0 for
    this arithmetic, independent of their blank/assumed state).
    """
    beg = values.setdefault("beginning_base", [None] * (n_years + 1))
    new = values.get("new_added") or [None] * (n_years + 1)
    churn = values.get("churned_out") or [None] * (n_years + 1)
    end = values.setdefault("ending_base", [None] * (n_years + 1))

    for y in range(n_years + 1):
        b = beg[y] if y == 0 else end[y - 1]
        b = b or 0.0
        beg[y] = b
        n = (new[y] if y < len(new) else None) or 0.0
        c = (churn[y] if y < len(churn) else None) or 0.0
        end[y] = b + n - c


# Annual ramp applied when projecting a known value into a blank year.
_GRID_RAMP_RATE = {"volume": 0.08, "rate": 0.03, "opex": 0.03, "churn": 0.0, "capex": 0.0, "seed": 0.0}

# Used only when a row has NO known value anywhere to anchor a projection from.
_GRID_BASELINE = {
    "volume": (10_000.0, "No volume signal available anywhere on this row - a generic starting "
                         "volume of 10,000/year is used, ramping ~8%/yr thereafter."),
    "churn": (0.0, "No churn signal available - assumed 0 for this year."),
    "rate": (75.0, "No pricing signal available - a generic $75/unit industry placeholder is used; "
                   "replace with the negotiated rate."),
    "capex": (200_000.0, "No CAPEX signal available - a generic $200k build/integration placeholder "
                         "is used, booked in Year 0."),
    "opex": (50_000.0, "No OPEX signal available - a generic $50k/yr placeholder is used."),
    "seed": (0.0, "New partnership - assumed no existing active base at Year 0."),
}


def _fallback_grid_cell(kind: str, row_label: str, year_idx: int, row_values: list):
    """Deterministic (no-Groq) ramped proposal for one blank cell. Returns (value, rationale)."""
    if kind == "capex" and year_idx != 0:
        return 0.0, "CAPEX is treated as a one-time Year 0 spend; no further CAPEX assumed."

    known = [(y, v) for y, v in enumerate(row_values) if v is not None]
    ramp = _GRID_RAMP_RATE.get(kind, 0.0)

    if known:
        nearest_y, nearest_v = min(known, key=lambda yv: abs(yv[0] - year_idx))
        value = nearest_v * ((1.0 + ramp) ** (year_idx - nearest_y))
        if ramp:
            direction = "growing" if year_idx >= nearest_y else "stepping back"
            rationale = (f"Projected from the known {row_label} value in Year {nearest_y} "
                        f"({nearest_v:,.2f}), {direction} {abs(ramp):.0%}/yr.")
        else:
            rationale = f"Held flat at the known {row_label} value from Year {nearest_y} ({nearest_v:,.2f})."
        return value, rationale

    base_val, base_reason = _GRID_BASELINE.get(kind, (0.0, "No standard default available."))
    if ramp and year_idx > 0:
        return base_val * ((1.0 + ramp) ** year_idx), base_reason
    return base_val, base_reason


_GRID_ASSUMPTIONS_PROMPT = """You are a partnerships financial analyst building a year-by-year plan for a \
{business_model} deal, covering Year 0 through Year {n_years}.

For each row below, some year values are already known (from the term sheet or the user); others are \
blank and need a proposed figure. Propose values that reasonably evolve year to year (e.g. volume \
ramping up, price stepping up modestly) rather than repeating one flat number, and stay consistent with \
any known values already on that row.

Rows (JSON):
{rows_json}

Commercial context from the term sheet:
{context_json}

Return ONLY a JSON object shaped like:
{{
  "<row_name>": {{
    "<year_index>": {{ "value": <number>, "rationale": "<one sentence>" }},
    ...
  }},
  ...
}}
Include an entry for every blank year index listed for every row. Values must be numbers >= 0. \
No markdown, no code fences, no extra keys.
"""


def propose_grid_assumptions(business_model: str, extraction: dict, row_specs: list, values: dict,
                             states: dict, n_years: int, groq_client, model: str = DEFAULT_MODEL) -> dict:
    """
    Propose a value for every still-blank cell in the grid, reasoning about how a
    value might reasonably change year to year rather than repeating one flat
    number. Nothing is left as ``None`` here - the caller shows every proposal to
    the user for Yes/Edit review, so a labelled industry-generic placeholder (with
    an honest rationale) is preferable to a gap.

    "computed" rows (the subscription waterfall's ending_base, and beginning_base
    beyond Year 0) are never targeted - they're always derived, never assumed.

    Returns ``{row_name: {year_idx: {"value": float, "rationale": str}}}`` - only
    for cells that were actually blank.
    """
    assumable = [r for r in row_specs if r["kind"] != "computed"]
    targets = {}
    for r in assumable:
        name = r["name"]
        row_states = states.get(name) or []
        blanks = [y for y in range(n_years + 1)
                  if y < len(row_states) and row_states[y] == "blank"
                  and not (r["kind"] == "seed" and y > 0)]
        if blanks:
            targets[name] = blanks
    if not targets:
        return {}

    rows_payload = []
    for r in assumable:
        name = r["name"]
        if name not in targets:
            continue
        row_vals = values.get(name) or []
        known = {y: row_vals[y] for y in range(n_years + 1)
                 if y < len(row_vals) and row_vals[y] is not None and y not in targets[name]}
        rows_payload.append({
            "name": name, "label": r["label"], "kind": r["kind"],
            "known": {str(y): v for y, v in known.items()},
            "blank_years": targets[name],
        })

    context = {
        k: extraction.get(k)
        for k in ("partner_name", "revenue_share", "revenue_share_details", "minimum_volume",
                  "payment_terms", "exclusivity", "contract_duration_months", "IP_Ownership")
    }
    prompt = _GRID_ASSUMPTIONS_PROMPT.format(
        business_model=business_model, n_years=n_years,
        rows_json=json.dumps(rows_payload, indent=2),
        context_json=json.dumps(context, indent=2),
    )

    parsed = {}
    try:
        parsed = _groq_json(groq_client, prompt, model)
    except BusinessCaseError:
        parsed = {}

    out = {}
    for r in assumable:
        name = r["name"]
        if name not in targets:
            continue
        row_out = {}
        groq_row = parsed.get(name) if isinstance(parsed, dict) else None
        row_vals = values.get(name) or []
        for y in targets[name]:
            entry = None
            if isinstance(groq_row, dict):
                entry = groq_row.get(str(y))
                if entry is None:
                    entry = groq_row.get(y)
            value, rationale = None, ""
            if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)) \
               and not isinstance(entry.get("value"), bool):
                value = float(entry["value"])
                rationale = str(entry.get("rationale", "")).strip()
            if value is None:
                value, rationale = _fallback_grid_cell(r["kind"], r["label"], y, row_vals)
            row_out[y] = {"value": max(0.0, value), "rationale": rationale or "Industry-standard estimate."}
        out[name] = row_out
    return out


# --------------------------------------------------------------------------- #
# 2b. Readable one-line-per-row summary of a row's assumed cells, for the
#     "Here are the assumptions I've made" review - detects whether the
#     assumed values across years form a flat / linear / percentage-growth
#     pattern before falling back to listing individual years.
# --------------------------------------------------------------------------- #

_MONEY_ROW_KINDS = {"rate", "capex", "opex"}

_FLAT_TOL = 0.02      # relative deviation from the mean still counts as "flat"
_SLOPE_TOL = 0.05     # relative-to-scale tolerance for a consistent $/yr step
_RATIO_TOL = 0.02     # absolute tolerance (e.g. 0.02 == 2 percentage points) on the growth ratio


def _detect_row_pattern(points: list):
    """
    points: [(year, value), ...] sorted by year, at least one point, values >= 0.

    Returns one of:
        {"kind": "single", "year": int, "value": float}
        {"kind": "flat", "value": float}
        {"kind": "linear", "start_value": float, "slope": float}
        {"kind": "percent", "start_value": float, "pct_per_year": float}
        {"kind": "irregular", "points": [(year, value), ...]}
    """
    if len(points) == 1:
        y, v = points[0]
        return {"kind": "single", "year": y, "value": v}

    values = [v for _, v in points]
    mean_v = sum(values) / len(values)

    if mean_v == 0:
        if all(v == 0 for v in values):
            return {"kind": "flat", "value": 0.0}
    elif max(abs(v - mean_v) for v in values) / mean_v <= _FLAT_TOL:
        return {"kind": "flat", "value": mean_v}

    scale = max(abs(v) for v in values) or 1.0
    slopes, ratios = [], []
    ratios_valid = True
    for (y1, v1), (y2, v2) in zip(points, points[1:]):
        dy = y2 - y1
        if dy <= 0:
            continue
        slopes.append((v2 - v1) / dy)
        if v1 > 0 and v2 > 0:
            ratios.append((v2 / v1) ** (1.0 / dy))
        else:
            ratios_valid = False

    # Score each candidate pattern as (how much of its own tolerance budget it
    # used, result) - 0.0 is a perfect fit, 1.0 is right at the edge of still
    # qualifying. A slow percentage-growth sequence (e.g. 5%/yr) has slopes
    # that only drift a little relative to EACH OTHER, which can still slip
    # inside a lenient linear tolerance - so rather than accepting the first
    # pattern that merely qualifies, both are scored and the tighter (lower
    # relative error) fit wins whenever both qualify.
    linear_fit = None
    if slopes:
        avg_slope = sum(slopes) / len(slopes)
        if abs(avg_slope) / scale > 1e-6:
            max_dev = max(abs(s - avg_slope) for s in slopes) / abs(avg_slope)
            if max_dev <= _SLOPE_TOL:
                linear_fit = (max_dev / _SLOPE_TOL,
                             {"kind": "linear", "start_value": values[0], "slope": avg_slope})

    percent_fit = None
    if ratios_valid and ratios:
        avg_ratio = sum(ratios) / len(ratios)
        if abs(avg_ratio - 1.0) > 1e-4:
            max_dev = max(abs(r - avg_ratio) for r in ratios)
            if max_dev <= _RATIO_TOL:
                percent_fit = (max_dev / _RATIO_TOL,
                               {"kind": "percent", "start_value": values[0],
                                "pct_per_year": (avg_ratio - 1.0) * 100.0})

    if linear_fit and (not percent_fit or linear_fit[0] <= percent_fit[0]):
        return linear_fit[1]
    if percent_fit:
        return percent_fit[1]

    return {"kind": "irregular", "points": list(points)}


def summarize_assumed_row(label: str, kind: str, points: list, rationale_by_year: dict,
                          currency_symbol: str = "$") -> str:
    """
    One short, readable line describing a row's assumed values across years -
    the pattern (flat / linear step / percentage growth / irregular), not a
    per-cell restatement. The grid itself remains the source of exact figures;
    this is only meant to be read alongside it, not instead of it.

    points : [(year, value), ...] - only the row's currently-ASSUMED cells,
        sorted by year (at least one).
    rationale_by_year : {year: str} - the per-cell rationale text (e.g. from
        propose_grid_assumptions), used only to append one short "why" clause,
        taken from the latest year in ``points`` (most relevant to the trend).
    """
    is_money = kind in _MONEY_ROW_KINDS
    fmt = (lambda v: f"{currency_symbol}{v:,.0f}") if is_money else (lambda v: f"{v:,.0f}")
    # A literal "$" (from a currency-suffixed label and/or a formatted amount)
    # reads fine as plain text, but if a line ends up with TWO of them,
    # Streamlit's markdown renderer treats the pair as inline LaTeX math
    # delimiters and mangles everything between them. Escape defensively so
    # this line always renders as plain text regardless of currency.
    esc = lambda s: s.replace("$", "\\$")
    label = esc(label)

    pattern = _detect_row_pattern(points)
    last_year = points[-1][0]
    why = esc((rationale_by_year.get(last_year) or "").strip())

    if pattern["kind"] == "single":
        text = f"Assumed **{label}** at {esc(fmt(pattern['value']))} for Year {pattern['year']}."
    elif pattern["kind"] == "flat":
        text = f"Assumed **{label}** at {esc(fmt(pattern['value']))}/year, flat across the term."
    elif pattern["kind"] == "linear":
        direction = "increasing" if pattern["slope"] >= 0 else "decreasing"
        text = (f"Assumed **{label}** starting at {esc(fmt(pattern['start_value']))}, "
                f"{direction} {esc(fmt(abs(pattern['slope'])))}/year.")
    elif pattern["kind"] == "percent":
        direction = "growing" if pattern["pct_per_year"] >= 0 else "declining"
        text = (f"Assumed **{label}** {direction} ~{abs(pattern['pct_per_year']):.0f}%/year "
                f"from {esc(fmt(pattern['start_value']))}.")
    else:
        parts = ", ".join(f"Year {y} = {esc(fmt(v))}" for y, v in pattern["points"])
        text = f"Assumed **{label}**: {parts}."

    if why:
        text += f" _{why}_"
    return text


# --------------------------------------------------------------------------- #
# 3. P&L directly from the confirmed grid - real per-year values into
#    calculate_profit / calculate_npv / calculate_roi. No re-asking for a flat
#    CAPEX/OPEX/duration - those come straight from the grid.
# --------------------------------------------------------------------------- #

def _grid_row(grid_values: dict, name: str, n_years: int) -> list:
    """A row's values as a plain float list, length n_years+1.

    Every cell must hold a real number. A missing row, a too-short row, or a blank
    (``None``) cell raises ``MissingInputError`` naming the exact gaps. Substituting
    0.0 would be indistinguishable from a genuine zero and would quietly skew revenue,
    profit, NPV and ROI with no indication anything was wrong.

    Only the rows a given model/licence structure actually consumes are read, so a
    row that plays no part in the calculation is never required.
    """
    vals = grid_values.get(name)
    if vals is None:
        raise MissingInputError(
            f"grid_values has no {name!r} row, which this business model requires for "
            f"Year 0 through Year {n_years}."
        )

    missing = [y for y in range(n_years + 1) if y >= len(vals) or vals[y] is None]
    if missing:
        cells = ", ".join(f"Year {y}" for y in missing)
        raise MissingInputError(
            f"{name!r} has no value for {cells}. Every cell must hold a real figure "
            "before the P&L can be built - use 0 for a genuinely-zero year rather than "
            "leaving it blank, or fill the gap with a clearly-labelled assumption."
        )

    return [float(vals[y]) for y in range(n_years + 1)]


def build_pl_from_grid(business_model: str, row_specs: list, grid_values: dict, n_years: int,
                       license_model_type: str = None, tier_breaks: list = None) -> dict:
    """
    Build a year-by-year P&L straight from the confirmed Step-2 grid - no separate
    scenario/driver re-entry. Per-year Revenue is derived from the grid's own
    volume/rate rows (model-specific formula below); CAPEX is the grid's CAPEX row;
    OPEX is the SUM of every row with kind "opex" (one or more line items). Each
    year's Revenue/CAPEX/OPEX is then run through ``calculate_profit`` (unchanged,
    reused as-is) to get Gross/Net Profit, and the Net Profit series is exactly
    what ``compute_npv_roi`` (also unchanged) expects for NPV/ROI.

    Revenue formula by model
    -------------------------
    subscription : ending_base[y] * price[y]
    adtech       : (annual_impressions[y] / 1000) * cpm_rate[y] + annual_clicks[y] * cpc_rate[y]
    fee_based    : annual_exposure_units[y] * rate_per_unit[y]
    licensing    : flat     -> rate_or_fee[y] (that year's flat licence fee, volume not multiplied)
                   per_unit -> volume[y] * rate_or_fee[y]
                   tiered   -> calculate_licensing_cost("tiered", volume=volume[y], tier_breaks=...)
                               run once per year against that year's volume (reuses the
                               verified marginal-bracket calculator unchanged)

    Parameters
    ----------
    business_model : one of BUSINESS_MODELS
    row_specs : list of the grid's row specs (model rows + GRID_CAPEX_ROW + any OPEX
        line items) - used only to find the OPEX row names (kind == "opex").
    grid_values : st.session_state.bc_grid_values - {row_name: [float|None, ...]}
    n_years : grid horizon (grid spans Year 0..n_years)
    license_model_type, tier_breaks : only used when business_model == "licensing"

    Returns
    -------
    dict with the same shape ``compute_npv_roi`` and the xlsx exporter expect:
        df               : pandas.DataFrame  rows = Revenue/CAPEX/OPEX/Gross Profit/
                           Net Profit/Cumulative Cash Flow, cols = Year 0..N
        business_model, projection_years
        cash_flows       : list[float]   Net Profit, Year 0..N
        revenue_by_year, capex_by_year, opex_by_year : list[float]
        total_capex, total_opex, final_cumulative     : float
        notes            : list[str]

    Raises
    ------
    BusinessCaseError  on an unrecognised business_model or license_model_type.
    MissingInputError  if any cell this model actually consumes is blank (None),
        absent, or beyond the end of a short row - naming the row and the exact
        years. Blanks are never silently read as 0, because a substituted zero is
        indistinguishable from a real one and would quietly skew the whole P&L.
    """
    if business_model == "subscription":
        ending = _grid_row(grid_values, "ending_base", n_years)
        price = _grid_row(grid_values, "price", n_years)
        revenue_by_year = [ending[y] * price[y] for y in range(n_years + 1)]

    elif business_model == "adtech":
        impressions = _grid_row(grid_values, "annual_impressions", n_years)
        cpm = _grid_row(grid_values, "cpm_rate", n_years)
        clicks = _grid_row(grid_values, "annual_clicks", n_years)
        cpc = _grid_row(grid_values, "cpc_rate", n_years)
        revenue_by_year = [
            (impressions[y] / 1000.0) * cpm[y] + clicks[y] * cpc[y] for y in range(n_years + 1)
        ]

    elif business_model == "fee_based":
        units = _grid_row(grid_values, "annual_exposure_units", n_years)
        rate = _grid_row(grid_values, "rate_per_unit", n_years)
        revenue_by_year = [units[y] * rate[y] for y in range(n_years + 1)]

    elif business_model == "licensing":
        # Each structure reads only the rows it actually uses: a flat licence fee
        # ignores volume, and a tiered one ignores the royalty row. Reading both up
        # front would demand figures that cannot affect the result.
        lmt = str(license_model_type or "flat").strip().lower()
        if lmt == "flat":
            revenue_by_year = list(_grid_row(grid_values, "rate_or_fee", n_years))
        elif lmt == "per_unit":
            volume = _grid_row(grid_values, "volume", n_years)
            rate = _grid_row(grid_values, "rate_or_fee", n_years)
            revenue_by_year = [volume[y] * rate[y] for y in range(n_years + 1)]
        elif lmt == "tiered":
            volume = _grid_row(grid_values, "volume", n_years)
            revenue_by_year = [0.0] * (n_years + 1)
            if tier_breaks:
                for y in range(n_years + 1):
                    if volume[y] > 0:
                        revenue_by_year[y] = calculate_licensing_cost(
                            model_type="tiered", flat_fee=None, per_unit_fee=None,
                            volume=volume[y], duration_years=1, tier_breaks=tier_breaks,
                        )["licensing_cost"]
        else:
            raise BusinessCaseError(f"Unknown license_model_type: {license_model_type!r}")
    else:
        raise BusinessCaseError(f"Unknown business_model: {business_model!r}")

    capex_by_year = _grid_row(grid_values, "capex", n_years)
    opex_row_names = [r["name"] for r in row_specs if r.get("kind") == "opex"]
    opex_by_year = [0.0] * (n_years + 1)
    for name in opex_row_names:
        row = _grid_row(grid_values, name, n_years)
        opex_by_year = [opex_by_year[y] + row[y] for y in range(n_years + 1)]

    notes = []
    if business_model == "licensing" and str(license_model_type or "flat").lower() == "tiered" \
       and not tier_breaks:
        notes.append(
            "Tiered licence structure has no tier breaks defined - revenue is shown as 0 "
            "until tier breaks are supplied."
        )

    gross_row, net_row = [], []
    for rev, cx, ox in zip(revenue_by_year, capex_by_year, opex_by_year):
        p = calculate_profit(revenue=rev, capex=cx, opex=ox)
        gross_row.append(p["gross_profit"])
        net_row.append(p["net_profit"])

    cumulative_row, running = [], 0.0
    for net in net_row:
        running += net
        cumulative_row.append(running)

    year_cols = [f"Year {y}" for y in range(n_years + 1)]
    rows = {
        "Revenue": revenue_by_year,
        "CAPEX": capex_by_year,
        "OPEX": opex_by_year,
        "Gross Profit": gross_row,
        "Net Profit": net_row,
        "Cumulative Cash Flow": cumulative_row,
    }
    df = pd.DataFrame.from_dict(rows, orient="index", columns=year_cols)
    df.index.name = "Attribute"

    return {
        "df": df,
        "business_model": business_model,
        "projection_years": n_years,
        "cash_flows": list(net_row),
        "revenue_by_year": revenue_by_year,
        "capex_by_year": capex_by_year,
        "opex_by_year": opex_by_year,
        "total_capex": sum(capex_by_year),
        "total_opex": sum(opex_by_year),
        "final_cumulative": cumulative_row[-1],
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# 4. NPV / ROI
# --------------------------------------------------------------------------- #


def compute_npv_roi(pl_model: dict, discount_rate: float = 0.10) -> dict:
    """
    NPV and ROI from a built P&L.

    - NPV: ``calculate_npv`` on the per-year Net Profit series (Year 0..N), which is
      the first difference of Cumulative Cash Flow.
    - ROI: ``calculate_roi`` with net_profit = final Cumulative Cash Flow and
      total_investment = total CAPEX + total OPEX over the horizon (total cash
      deployed). If that base is 0, ROI is reported as unavailable rather than raising.

    Returns {"npv": {...}, "roi": {...} | None, "discount_rate", "investment_base"}.
    """
    npv = calculate_npv(pl_model["cash_flows"], discount_rate=discount_rate)

    investment_base = pl_model["total_capex"] + pl_model["total_opex"]
    roi = None
    if investment_base > 0:
        roi = calculate_roi(
            net_profit=pl_model["final_cumulative"], total_investment=investment_base
        )

    return {
        "npv": npv,
        "roi": roi,
        "discount_rate": discount_rate,
        "investment_base": investment_base,
    }


# --------------------------------------------------------------------------- #
# Offline smoke test (no network): pre-fill + scenarios with hardcoded values.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    example_extraction = {
        "partner_name": "Meridian Motors Corporation",
        "contract_duration_months": 36,
        "revenue_share": "30% of Net Revenue",
        "minimum_volume": "50,000 units per 12-month period",
    }

    print("\n=== year-by-year grid (subscription): default years + prefill ===")
    n_years = default_grid_years(example_extraction)
    print("  default_grid_years:", n_years)
    row_specs = MODEL_GRID_ROWS["subscription"] + [GRID_CAPEX_ROW]
    prefill = prefill_grid_cells("subscription", example_extraction, row_specs, n_years)
    print("  prefill:", json.dumps(prefill, indent=2, default=str))

    print("\n=== subscription waterfall roll-forward ===")
    grid_values = {
        "beginning_base": [0.0] + [None] * n_years,
        "new_added": [None] + [(prefill.get("new_added") or {}).get(y, {}).get("value") for y in range(1, n_years + 1)],
        "churned_out": [None] + [5_000.0] * n_years,
    }
    recompute_subscription_waterfall(grid_values, n_years)
    for y in range(n_years + 1):
        print(f"  Year {y}: beginning={grid_values['beginning_base'][y]:,.0f}  "
              f"new={grid_values['new_added'][y] or 0:,.0f}  "
              f"churned={grid_values['churned_out'][y] or 0:,.0f}  "
              f"ending={grid_values['ending_base'][y]:,.0f}")

    print("\n=== fallback grid cell (no Groq) ===")
    v, why = _fallback_grid_cell("volume", "New customers added", 2, [None, 50_000.0, None, None])
    print(f"  volume Year 2 projected from Year 1: {v:,.0f}  -- {why}")
    v, why = _fallback_grid_cell("capex", "CAPEX", 1, [200_000.0, None, None])
    print(f"  capex Year 1 (no further CAPEX assumed): {v:,.0f}  -- {why}")
    v, why = _fallback_grid_cell("rate", "Annual revenue per active customer ($)", 0, [None, None])
    print(f"  rate Year 0 with no known values anywhere: {v:,.2f}  -- {why}")

    print("\n=== build_pl_from_grid (subscription, 3yr) ===")
    sub_row_specs = MODEL_GRID_ROWS["subscription"] + [GRID_CAPEX_ROW,
                                                        {"name": "opex__0__opex", "kind": "opex"}]
    sub_grid_values = {
        "beginning_base": [0.0, 0.0, 45000.0, 90000.0],
        "new_added": [0.0, 50000.0, 50000.0, 50000.0],
        "churned_out": [0.0, 5000.0, 5000.0, 5000.0],
        "ending_base": [0.0, 45000.0, 90000.0, 135000.0],
        "price": [0.0, 120.0, 126.0, 132.0],
        "capex": [2_000_000.0, 0.0, 0.0, 0.0],
        "opex__0__opex": [0.0, 3_500_000.0, 3_500_000.0, 3_500_000.0],
    }
    pl_grid = build_pl_from_grid("subscription", sub_row_specs, sub_grid_values, 3)
    print(pl_grid["df"].to_string(float_format=lambda x: f"{x:,.0f}"))
    print("  cash_flows:", [round(c) for c in pl_grid["cash_flows"]])
    nr_grid = compute_npv_roi(pl_grid, discount_rate=0.10)
    print(f"  NPV @10%: {nr_grid['npv']['npv']:,.0f}")
    print(f"  ROI: {nr_grid['roi']['roi_pct']:.1f}%" if nr_grid["roi"] else "  ROI: n/a")

    print("\n=== build_pl_from_grid (licensing, per_unit, 3yr) ===")
    lic_row_specs = [{"name": "volume", "kind": "volume"}, {"name": "rate_or_fee", "kind": "rate"},
                     GRID_CAPEX_ROW, {"name": "opex__0__opex", "kind": "opex"}]
    lic_grid_values = {
        "volume": [0.0, 50000.0, 50000.0, 50000.0],
        "rate_or_fee": [0.0, 75.0, 78.0, 81.0],
        "capex": [500_000.0, 0.0, 0.0, 0.0],
        "opex__0__opex": [0.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
    }
    pl_lic = build_pl_from_grid("licensing", lic_row_specs, lic_grid_values, 3,
                                license_model_type="per_unit")
    print(pl_lic["df"].to_string(float_format=lambda x: f"{x:,.0f}"))

    print("\n=== build_pl_from_grid (licensing, tiered, 3yr) ===")
    pl_tier = build_pl_from_grid(
        "licensing", lic_row_specs, lic_grid_values, 3, license_model_type="tiered",
        tier_breaks=[{"min_units": 0, "rate_per_unit": 100}, {"min_units": 30_000, "rate_per_unit": 80}],
    )
    print(pl_tier["df"].loc["Revenue"].to_string())
    for n in pl_tier["notes"]:
        print("  note:", n)
