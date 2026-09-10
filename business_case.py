"""
Business-case flow helpers.

Non-UI logic for the conversational business-case builder:

1. classify_business_model()          - Groq picks one of the four models + why.
2. MODEL_INPUT_SPECS / required_fields - which financial_engine inputs a model needs.
3. prefill_inputs_from_extraction()   - deterministic pre-fill from the term sheet.
4. propose_assumptions()              - Groq drafts clearly-labelled defaults for the
                                        fields the term sheet doesn't cover (fallback
                                        table if Groq is unavailable).
5. run_scenarios()                    - Low / Medium / High by multiplying the model's
                                        driver(s), calling the matched financial_engine
                                        function each time.

No Streamlit import. The Streamlit tab owns all session-state and widgets and calls
into these. All period conventions are annual, matching financial_engine.
"""

from __future__ import annotations

import json

from financial_engine import (
    FinancialEngineError,
    calculate_adtech_revenue,
    calculate_fee_based_revenue,
    calculate_licensing_cost,
    calculate_subscription_revenue,
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


class BusinessCaseError(Exception):
    """Raised when a business-case step cannot complete."""


# --------------------------------------------------------------------------- #
# 1. Classification
# --------------------------------------------------------------------------- #

_CLASSIFY_PROMPT = """You are a partnerships analyst. Classify the PRIMARY business model of this \
deal as exactly one of:

- "subscription": the partner earns a recurring per-customer/per-seat fee (often with activation and churn).
- "adtech": the partner earns media revenue priced on impressions (CPM) and/or clicks (CPC).
- "fee_based": the partner earns a flat fee per exposure unit (per vehicle, per transaction, per API call, per device) with no CPM/CPC and no recurring-subscriber mechanic.
- "licensing": one party pays another a software / IP licence fee (flat, per-unit royalty, or tiered) - e.g. an OEM licensing a chip/software platform.

Extracted term sheet (JSON):
{extraction_json}

Deterministically-flagged risks (names only):
{risk_names}

Return ONLY a JSON object with exactly these keys:
- "business_model": one of "subscription", "adtech", "fee_based", "licensing".
- "reasoning": 1-3 sentences citing the specific term-sheet language (revenue share, fees, volume basis, IP terms) that drove the choice.
- "signals": array of short strings - the concrete phrases/fields that point to this model.
- "confidence": "high", "medium", or "low".

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
    Ask Groq to classify the deal as one of BUSINESS_MODELS.

    Returns {"business_model", "reasoning", "signals": [...], "confidence"}.
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

    return {
        "business_model": bm,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "signals": [str(s).strip() for s in signals if str(s).strip()],
        "confidence": str(parsed.get("confidence", "")).strip().lower() or "unknown",
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
        "subscription", "subscriber", "churn", "activation", "recurring",
        "monthly fee", "monthly subscription", "per active", "per seat", "per-seat",
        "per user", "per-user", "mrr", "arr", "retention",
    ],
    "adtech": [
        "cpm", "cpc", "impression", "click", "click-through", "clickthrough", "ctr",
        "ad inventory", "ad-inventory", "advertising", "advert", "programmatic",
        "fill rate", "per thousand", "per mille", "media revenue",
    ],
    "fee_based": [
        "flat fee", "fee per", "per-unit fee", "per unit fee", "per vehicle",
        "per transaction", "per device", "per api", "per call", "per connected",
        "usage fee", "usage-based", "per activation fee", "per-unit license fee",
        "per-unit licence fee",
    ],
    "licensing": [
        "licens", "royalt", "intellectual property", "ip ownership", "sublicens",
        "oem", "platform fee", "platform license", "platform licence",
    ],
}

MODEL_SIGNAL_HINT = {
    "subscription": "subscriber / recurring-fee / churn / activation language",
    "adtech": "CPM / CPC / impressions / clicks / ad-inventory language",
    "fee_based": "flat fee-per-unit language (per vehicle / transaction / device / call)",
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
# 2. Input specs per model
# --------------------------------------------------------------------------- #
#
# kind:  "count" | "money" | "rate" (0-1) | "years" | "choice" | "tiers"
# prefill: key in prefill_inputs_from_extraction()'s output this field maps to.

MODEL_INPUT_SPECS = {
    "subscription": [
        {"name": "new_customers_per_year", "label": "New customers acquired per year",
         "kind": "count", "prefill": "annual_volume",
         "help": "Prospective customers reached per year, before activation."},
        {"name": "price", "label": "Annual revenue per active customer ($)", "kind": "money",
         "help": "What the partner earns per active customer per year."},
        {"name": "activation_rate", "label": "Activation rate (0-1)", "kind": "rate",
         "help": "Fraction of new customers who actually start a subscription."},
        {"name": "annual_churn_rate", "label": "Annual churn rate (0-1)", "kind": "rate",
         "help": "Fraction of the active base lost per year."},
        {"name": "duration_years", "label": "Term (years)", "kind": "years", "prefill": "duration_years"},
    ],
    "adtech": [
        {"name": "annual_impressions", "label": "Ad impressions per year", "kind": "count"},
        {"name": "cpm_rate", "label": "Revenue per 1,000 impressions — CPM ($)", "kind": "money"},
        {"name": "annual_clicks", "label": "Clicks per year", "kind": "count"},
        {"name": "cpc_rate", "label": "Revenue per click — CPC ($)", "kind": "money"},
        {"name": "duration_years", "label": "Term (years)", "kind": "years", "prefill": "duration_years"},
    ],
    "fee_based": [
        {"name": "annual_exposure_units", "label": "Billable exposure units per year",
         "kind": "count", "prefill": "annual_volume",
         "help": "Vehicles / transactions / devices / API calls billed per year."},
        {"name": "rate_per_unit", "label": "Fee per unit ($)", "kind": "money"},
        {"name": "duration_years", "label": "Term (years)", "kind": "years", "prefill": "duration_years"},
    ],
    "licensing": [
        {"name": "license_model_type", "label": "Licence structure", "kind": "choice",
         "choices": ["flat", "per_unit", "tiered"]},
        {"name": "flat_fee", "label": "Annual flat licence fee ($)", "kind": "money",
         "needs_type": ["flat"]},
        {"name": "per_unit_fee", "label": "Royalty per licensed unit ($)", "kind": "money",
         "needs_type": ["per_unit"]},
        {"name": "tier_breaks", "label": "Tier breaks (units → rate/unit)", "kind": "tiers",
         "needs_type": ["tiered"]},
        {"name": "volume", "label": "Total licensed units over the term", "kind": "count",
         "prefill": "total_volume", "needs_type": ["per_unit", "tiered"]},
        {"name": "duration_years", "label": "Term (years)", "kind": "years", "prefill": "duration_years"},
    ],
}


def required_fields(business_model: str, inputs: dict) -> list:
    """
    Field names the matched financial_engine function needs, given the current inputs.

    For licensing this depends on the chosen ``license_model_type`` (flat needs
    flat_fee only; per_unit needs per_unit_fee + volume; tiered needs tier_breaks +
    volume).
    """
    specs = MODEL_INPUT_SPECS[business_model]
    lm_type = inputs.get("license_model_type")
    out = []
    for spec in specs:
        needs = spec.get("needs_type")
        if needs is not None and lm_type not in needs:
            continue
        out.append(spec["name"])
    return out


def missing_fields(business_model: str, inputs: dict) -> list:
    """Required field names whose value is currently None / empty."""
    out = []
    for name in required_fields(business_model, inputs):
        val = inputs.get(name)
        if val is None or (name == "tier_breaks" and not val):
            out.append(name)
    return out


# --------------------------------------------------------------------------- #
# 3. Deterministic pre-fill from the term sheet extraction
# --------------------------------------------------------------------------- #

def prefill_inputs_from_extraction(business_model: str, extraction: dict) -> dict:
    """
    Deterministic pre-fill of engine inputs from an already-extracted term sheet.

    Returns ``{field_name: {"value": <number>, "source": "<human note>"}}`` for the
    fields the term sheet actually addresses (duration, volume / minimum commitment).
    Anything not derivable is simply absent - the caller then asks the user or offers
    a labelled assumption. Never guesses.
    """
    derived = derive_licensing_inputs_from_extraction(extraction)
    out = {}

    dur = derived.get("duration_years")
    if dur is not None:
        months = extraction.get("contract_duration_months")
        out["duration_years"] = {
            "value": dur,
            "source": f"term sheet: contract_duration_months = {months} → {dur:g} years",
        }

    per_year = derived.get("volume_per_year")
    total = derived.get("volume")
    src_text = derived.get("volume_source_text")

    specs = MODEL_INPUT_SPECS[business_model]
    for spec in specs:
        pf = spec.get("prefill")
        if pf == "annual_volume" and per_year is not None:
            out[spec["name"]] = {
                "value": per_year,
                "source": f"term sheet minimum_volume: {src_text!r} → {per_year:g}/year "
                          f"(verify this maps to '{spec['name']}')",
            }
        elif pf == "total_volume" and total is not None:
            out[spec["name"]] = {
                "value": total,
                "source": f"term sheet minimum_volume: {src_text!r} → {total:g} over the term "
                          "(MINIMUM commitment — actual may be higher)",
            }

    return out


# --------------------------------------------------------------------------- #
# 4. Proposed assumptions for the remaining fields
# --------------------------------------------------------------------------- #

# Used only if Groq is unavailable / unusable. Value is None where a number simply
# cannot be guessed from industry standards (e.g. a price, a raw volume) — those
# stay missing and the user must supply them.
_FALLBACK_ASSUMPTIONS = {
    "activation_rate": (0.30, "Industry default: a newly launched paid feature typically activates ~25-35% of reached customers."),
    "annual_churn_rate": (0.15, "Industry default: consumer subscription annual churn commonly runs 15-20%; 15% used as a mid-point."),
    "cpm_rate": (3.0, "Industry default: blended display CPM commonly $2-5."),
    "cpc_rate": (0.60, "Industry default: blended display CPC commonly $0.40-0.80."),
    "duration_years": (3.0, "Industry default: initial partnership terms are commonly 3 years."),
    "price": (None, "A per-customer price cannot be assumed from industry standards — needs a real figure."),
    "rate_per_unit": (None, "A per-unit fee cannot be assumed — needs a real figure or the negotiated rate."),
    "per_unit_fee": (None, "A per-unit royalty cannot be assumed — needs the negotiated rate."),
    "flat_fee": (None, "A flat licence fee cannot be assumed — needs the negotiated figure."),
    "new_customers_per_year": (None, "Reach/volume cannot be assumed — needs a real figure or the term-sheet minimum."),
    "annual_exposure_units": (None, "Exposure volume cannot be assumed — needs a real figure or the term-sheet minimum."),
    "annual_impressions": (None, "Impression volume cannot be assumed — needs a real figure or a media plan."),
    "annual_clicks": (None, "Click volume cannot be assumed — needs a real figure or an assumed CTR on impressions."),
}

_ASSUMPTIONS_PROMPT = """You are a partnerships financial analyst. For a {business_model} deal we still \
need values for these inputs (financial_engine, annual periods):

{missing_block}

Commercial context from the term sheet:
{context_json}

Propose a reasonable INDUSTRY-STANDARD default for each field you can. If a field cannot \
responsibly be assumed from industry norms (e.g. a specific price or a raw traffic volume), \
return its value as null and say so in the rationale.

Return ONLY a JSON object mapping each field name to an object:
  "<field>": {{ "value": <number or null>, "rationale": "<one sentence: the standard range or the term it is based on>" }}

Rates (activation_rate, annual_churn_rate) must be between 0 and 1. No markdown, no code fences, no extra keys.
"""

_RATE_FIELDS = {"activation_rate", "annual_churn_rate"}


def propose_assumptions(business_model: str, extraction: dict, missing: list, groq_client,
                        model: str = DEFAULT_MODEL) -> dict:
    """
    Draft clearly-labelled default assumptions for the still-missing fields.

    Returns ``{field: {"value": <number|None>, "rationale": str, "source": "groq"|"fallback"}}``
    for every field in ``missing``. The caller MUST show each to the user and only
    apply the ones the user individually accepts — nothing here is auto-applied.
    """
    if not missing:
        return {}

    labels = {s["name"]: s["label"] for s in MODEL_INPUT_SPECS[business_model]}
    context = {
        k: extraction.get(k)
        for k in ("partner_name", "revenue_share", "revenue_share_details", "minimum_volume",
                  "payment_terms", "exclusivity", "contract_duration_months", "IP_Ownership")
    }
    prompt = _ASSUMPTIONS_PROMPT.format(
        business_model=business_model,
        missing_block="\n".join(f"- {m} ({labels.get(m, m)})" for m in missing),
        context_json=json.dumps(context, indent=2),
    )

    parsed = {}
    try:
        parsed = _groq_json(groq_client, prompt, model)
    except BusinessCaseError:
        parsed = {}

    out = {}
    for field in missing:
        entry = parsed.get(field) if isinstance(parsed, dict) else None
        value, rationale, source = None, "", "fallback"

        if isinstance(entry, dict) and "value" in entry:
            value = entry.get("value")
            rationale = str(entry.get("rationale", "")).strip()
            source = "groq"
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                value = None

        if value is None and source != "groq":
            fb = _FALLBACK_ASSUMPTIONS.get(field)
            if fb is not None:
                value, rationale = fb
                rationale = rationale or ""

        if value is not None:
            value = float(value)
            if field in _RATE_FIELDS:
                value = min(1.0, max(0.0, value))

        out[field] = {
            "value": value,
            "rationale": rationale or "No industry-standard default available — please supply a real figure.",
            "source": source,
        }
    return out


# --------------------------------------------------------------------------- #
# 5. Low / Medium / High scenarios
# --------------------------------------------------------------------------- #

SCENARIO_NAMES = ("low", "medium", "high")

# field -> (low_multiplier, medium_multiplier, high_multiplier), tuned so "low" is
# the pessimistic revenue/most-costly case.
SCENARIO_MULTIPLIERS = {
    "subscription": {
        "activation_rate": (0.8, 1.0, 1.2),
        "annual_churn_rate": (1.25, 1.0, 0.8),
    },
    "adtech": {
        "annual_impressions": (0.8, 1.0, 1.2),
        "annual_clicks": (0.8, 1.0, 1.2),
    },
    "fee_based": {
        "annual_exposure_units": (0.8, 1.0, 1.2),
    },
    "licensing": {
        "volume": (0.8, 1.0, 1.2),
    },
}

_HEADLINE_KEY = {
    "subscription": "total_revenue",
    "adtech": "total_revenue",
    "fee_based": "total_revenue",
    "licensing": "licensing_cost",
}

_HEADLINE_LABEL = {
    "subscription": "Total revenue over the term",
    "adtech": "Total revenue over the term",
    "fee_based": "Total revenue over the term",
    "licensing": "Total licensing cost over the term (paid by the OEM)",
}


def _call_engine(business_model: str, inputs: dict):
    if business_model == "subscription":
        return calculate_subscription_revenue(
            new_customers_per_year=inputs.get("new_customers_per_year"),
            price=inputs.get("price"),
            activation_rate=inputs.get("activation_rate"),
            annual_churn_rate=inputs.get("annual_churn_rate"),
            duration_years=inputs.get("duration_years"),
        )
    if business_model == "adtech":
        return calculate_adtech_revenue(
            annual_impressions=inputs.get("annual_impressions"),
            cpm_rate=inputs.get("cpm_rate"),
            annual_clicks=inputs.get("annual_clicks"),
            cpc_rate=inputs.get("cpc_rate"),
            duration_years=inputs.get("duration_years"),
        )
    if business_model == "fee_based":
        return calculate_fee_based_revenue(
            annual_exposure_units=inputs.get("annual_exposure_units"),
            rate_per_unit=inputs.get("rate_per_unit"),
            duration_years=inputs.get("duration_years"),
        )
    if business_model == "licensing":
        return calculate_licensing_cost(
            model_type=inputs.get("license_model_type"),
            flat_fee=inputs.get("flat_fee"),
            per_unit_fee=inputs.get("per_unit_fee"),
            volume=inputs.get("volume"),
            duration_years=inputs.get("duration_years"),
            tier_breaks=inputs.get("tier_breaks"),
        )
    raise BusinessCaseError(f"Unknown business_model: {business_model!r}")


def run_scenarios(business_model: str, inputs: dict) -> dict:
    """
    Run the matched financial_engine function three times — Low / Medium / High —
    by multiplying the model's driver field(s).

    Drivers:
      - subscription : activation_rate (x0.8/1.0/1.2) and annual_churn_rate (x1.25/1.0/0.8)
      - adtech       : annual_impressions and annual_clicks (x0.8/1.0/1.2)
      - fee_based    : annual_exposure_units (x0.8/1.0/1.2)
      - licensing    : volume (x0.8/1.0/1.2); a flat-fee licence has no volume driver,
                       so Low/High there reflect +/-10% fee-negotiation uncertainty.

    Returns
    -------
    dict with:
        business_model : str
        driver_fields  : list[str]
        headline_key   : str        key into each scenario's ``result``
        headline_label : str
        note           : str | None
        scenarios      : {name: {"multipliers": {...}, "adjusted": {field: value},
                                 "headline": float, "result": <engine return dict>}}

    Raises
    ------
    BusinessCaseError if a required input is still missing (wraps MissingInputError).
    """
    still_missing = missing_fields(business_model, inputs)
    if still_missing:
        raise BusinessCaseError(
            "Cannot run scenarios — still missing: " + ", ".join(still_missing)
        )

    mult_map = dict(SCENARIO_MULTIPLIERS[business_model])
    note = None
    if business_model == "licensing" and not inputs.get("volume"):
        mult_map = {"flat_fee": (0.9, 1.0, 1.1)}
        note = ("Flat-fee licence has no volume driver — Low/High shown here are "
                "+/-10% fee-negotiation uncertainty, not demand sensitivity.")

    scenarios = {}
    for i, scen in enumerate(SCENARIO_NAMES):
        scen_inputs = dict(inputs)
        adjusted = {}
        multipliers = {}
        for field, triple in mult_map.items():
            base = inputs.get(field)
            if base is None:
                continue
            val = base * triple[i]
            if field in _RATE_FIELDS:
                val = min(1.0, max(0.0, val))
            scen_inputs[field] = val
            adjusted[field] = val
            multipliers[field] = triple[i]

        try:
            result = _call_engine(business_model, scen_inputs)
        except FinancialEngineError as e:
            raise BusinessCaseError(f"{scen.title()} scenario failed: {e}") from e

        scenarios[scen] = {
            "multipliers": multipliers,
            "adjusted": adjusted,
            "headline": result[_HEADLINE_KEY[business_model]],
            "result": result,
        }

    return {
        "business_model": business_model,
        "driver_fields": list(mult_map),
        "headline_key": _HEADLINE_KEY[business_model],
        "headline_label": _HEADLINE_LABEL[business_model],
        "note": note,
        "scenarios": scenarios,
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

    print("=== prefill (subscription) ===")
    print(json.dumps(prefill_inputs_from_extraction("subscription", example_extraction), indent=2, default=str))

    print("\n=== required / missing (subscription) ===")
    inputs = {"duration_years": 3.0, "new_customers_per_year": 50_000}
    print("required:", required_fields("subscription", inputs))
    print("missing :", missing_fields("subscription", inputs))

    print("\n=== scenarios (subscription, all inputs hardcoded) ===")
    inputs = {
        "new_customers_per_year": 50_000, "price": 120.0,
        "activation_rate": 0.4, "annual_churn_rate": 0.15, "duration_years": 3,
    }
    out = run_scenarios("subscription", inputs)
    for name, s in out["scenarios"].items():
        print(f"  {name:<7} adjusted={ {k: round(v, 4) for k, v in s['adjusted'].items()} }  "
              f"{out['headline_label']}: {s['headline']:,.0f}")

    print("\n=== scenarios (licensing per_unit) ===")
    lic = {"license_model_type": "per_unit", "per_unit_fee": 75.0, "volume": 150_000, "duration_years": 3}
    out = run_scenarios("licensing", lic)
    for name, s in out["scenarios"].items():
        print(f"  {name:<7} adjusted={s['adjusted']}  {out['headline_label']}: {s['headline']:,.0f}")

    print("\n=== scenarios (licensing flat — no volume driver) ===")
    lic = {"license_model_type": "flat", "flat_fee": 1_200_000.0, "duration_years": 3}
    out = run_scenarios("licensing", lic)
    print("  note:", out["note"])
    for name, s in out["scenarios"].items():
        print(f"  {name:<7} adjusted={s['adjusted']}  {out['headline_label']}: {s['headline']:,.0f}")
