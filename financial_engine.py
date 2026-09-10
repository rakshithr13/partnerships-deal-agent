"""
Financial modelling engine for a partnership / term sheet.

Period convention
-----------------
**Every function in this module works in annual periods.** All durations are given
as ``duration_years``; all rates, volumes and prices are annual unless a parameter
name says otherwise; all traces returned are per-year.

- The three revenue models and the flat-fee branch of the licensing model take a
  **whole number of years** for ``duration_years`` (they model discrete annual
  periods). ``calculate_cloud_hosting_cost`` and the flat licensing branch accept a
  **fractional** ``duration_years`` (they are linear in time), and their per-year
  trace ends with a partial-year row when the term isn't a whole number of years.
- ``calculate_cloud_hosting_cost`` is the one place a monthly figure survives, and
  only internally: cloud providers publish list prices per month (per GB-month, per
  hour, ...), so that function keeps the fetched rate in its native published
  (typically monthly) unit and annualises it by x12. Its docstring spells this out.
- ``derive_licensing_inputs_from_extraction`` converts the term sheet's
  ``contract_duration_months`` to years (``/ 12``) and reports the result.

Two kinds of function live here:

1. Deterministic calculators - pure Python, no network, no LLM. Given the same
   inputs they always return the same numbers. These are the revenue models, the
   licensing-cost model, and the universal tools (profit / NPV / ROI / chart).
   Every one is independently testable with hardcoded values (see __main__).

2. One hybrid function - ``calculate_cloud_hosting_cost`` - which fetches a
   current per-unit price from the public web (Tavily) and extracts a number from
   it (Groq), then does a deterministic ``usage x rate`` calculation. Its result
   is explicitly *directional*: it returns the fetched rate, the source URL, and
   the raw quoted text next to the number so a human can sanity-check it.
   ``calculate_licensing_cost`` can *optionally* do a similar rough web lookup,
   but only as a labelled sanity-check - never as a verification.

Missing-input policy
--------------------
None of these functions silently substitutes a default for a missing *required*
input. If a required value is ``None`` they raise ``MissingInputError``. It is the
calling code's job to catch that, ask the user, and only then pass in either the
real figure or an explicitly-labelled industry-standard assumption. (``0`` is a
real value and is accepted; only ``None`` counts as "missing".)

No Streamlit dependency. ``generate_pl_chart`` returns an Altair chart object that
Streamlit can render later via ``st.altair_chart(...)``, but importing this module
never imports Streamlit.
"""

from __future__ import annotations

import json
import math
import os
import re
from numbers import Real

# Altair is already a dependency (pulled in by Streamlit) and does not import
# Streamlit itself, so this keeps the module UI-framework-free.
import altair as alt
import pandas as pd


class FinancialEngineError(Exception):
    """Raised when a calculation cannot be completed (e.g. a web rate could not be extracted)."""


class MissingInputError(FinancialEngineError):
    """
    Raised when a *required* input is ``None``.

    The caller is expected to catch this, ask the user for the figure, and offer a
    clearly-labelled industry-standard assumption only if the user says they don't
    have the real number.
    """


# --------------------------------------------------------------------------- #
# Shared validation helpers
# --------------------------------------------------------------------------- #

def _require(**named_values):
    """
    Raise ``MissingInputError`` if any supplied value is ``None``.

    ``_require(volume=volume, price=price)`` -> raises listing every name whose
    value is ``None``. ``0``/``0.0``/``""`` are NOT treated as missing.
    """
    missing = [name for name, value in named_values.items() if value is None]
    if missing:
        raise MissingInputError(
            "Missing required input(s): "
            + ", ".join(sorted(missing))
            + ". The caller must obtain these (real figure, or a clearly-labelled "
            "industry-standard assumption) before calling this function."
        )


def _number(name, value, *, allow_negative=False):
    """Validate that ``value`` is a real number and return it as ``float``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FinancialEngineError(f"{name!r} must be a number, got {type(value).__name__}")
    value = float(value)
    if not allow_negative and value < 0:
        raise FinancialEngineError(f"{name!r} must not be negative, got {value}")
    return value


def _rate(name, value):
    """Validate a proportion in the closed interval [0, 1] and return it as ``float``."""
    value = _number(name, value)
    if value > 1:
        raise FinancialEngineError(
            f"{name!r} must be a proportion between 0 and 1 (e.g. 0.05 for 5%), got {value}"
        )
    return value


def _whole_years(name, value):
    """
    Validate a duration as a positive WHOLE number of years and return it as ``int``.

    Used by the models that step through discrete annual periods (the revenue
    models and the flat-fee licensing branch's trace). A fractional term must be
    resolved by the caller (round, or split the analysis) before calling these.
    """
    value = _number(name, value)
    if value <= 0 or value != int(value):
        raise FinancialEngineError(
            f"{name!r} must be a positive whole number of years, got {value}. "
            "Round or split the term before calling this function."
        )
    return int(value)


def _positive_years(name, value):
    """Validate a duration as a positive number of years (fractional allowed); return ``float``."""
    value = _number(name, value)
    if value <= 0:
        raise FinancialEngineError(f"{name!r} must be greater than 0 years, got {value}")
    return value


def _year_slices(duration_years):
    """
    Break a (possibly fractional) term into per-year slices for a linear trace.

    ``2.5`` -> ``[(1, 1.0), (2, 1.0), (3, 0.5)]`` - the ``float`` is that year's
    portion of a full year. ``Σ portions == duration_years``.
    """
    full = int(math.floor(duration_years))
    frac = duration_years - full
    slices = [(y, 1.0) for y in range(1, full + 1)]
    if frac > 1e-9:
        slices.append((full + 1, frac))
    return slices


# --------------------------------------------------------------------------- #
# Revenue models (deterministic) - annual periods
# --------------------------------------------------------------------------- #

def calculate_subscription_revenue(new_customers_per_year, price, activation_rate,
                                   annual_churn_rate, duration_years):
    """
    Recurring subscription revenue with a continuously-growing base: new customers
    arrive every YEAR, activate at some rate, and the standing base churns annually.

    Model (annual periods)
    ----------------------
    Let ``new_subscribers_per_year = new_customers_per_year * activation_rate``
    (constant each year). Starting from ``active_base = 0``, for each year y = 1..N:

    - ``active_base = active_base * (1 - annual_churn_rate) + new_subscribers_per_year``
    - ``revenue_y  = active_base * price``

    ``total_revenue = Σ_{y=1..N} revenue_y``. The base grows toward a steady state
    of ``new_subscribers_per_year / annual_churn_rate`` (when the churn rate > 0);
    with ``annual_churn_rate == 0`` it grows by ``new_subscribers_per_year`` a year.

    Parameters
    ----------
    new_customers_per_year : number
        Prospective customers acquired per year (before activation), a count.
    price : number
        Annual revenue per active customer (currency / customer / year).
    activation_rate : number in [0, 1]
        Fraction of ``new_customers_per_year`` that actually starts a subscription.
    annual_churn_rate : number in [0, 1]
        Fraction of the active base lost per year.
    duration_years : positive whole int
        Number of years modelled.

    Returns
    -------
    dict with:
        total_revenue : float                over the whole term (currency)
        new_subscribers_per_year : float     activated arrivals added each year
        total_new_subscribers : float        new_subscribers_per_year * duration_years
        ending_active_base : float           active base after the final year
        yearly : list[dict]                  per-year {year, new_subscribers,
                                             active_base, revenue} - the full trace,
                                             so the recurrence can be checked by hand
        assumptions : dict                   echo of the inputs used
    """
    _require(
        new_customers_per_year=new_customers_per_year, price=price,
        activation_rate=activation_rate, annual_churn_rate=annual_churn_rate,
        duration_years=duration_years,
    )
    new_customers_per_year = _number("new_customers_per_year", new_customers_per_year)
    price = _number("price", price)
    activation_rate = _rate("activation_rate", activation_rate)
    annual_churn_rate = _rate("annual_churn_rate", annual_churn_rate)
    duration_years = _whole_years("duration_years", duration_years)

    new_subscribers_per_year = new_customers_per_year * activation_rate

    yearly = []
    total_revenue = 0.0
    active_base = 0.0
    for year in range(1, duration_years + 1):
        active_base = active_base * (1 - annual_churn_rate) + new_subscribers_per_year
        revenue = active_base * price
        total_revenue += revenue
        yearly.append({
            "year": year,
            "new_subscribers": new_subscribers_per_year,
            "active_base": active_base,
            "revenue": revenue,
        })

    return {
        "total_revenue": total_revenue,
        "new_subscribers_per_year": new_subscribers_per_year,
        "total_new_subscribers": new_subscribers_per_year * duration_years,
        "ending_active_base": active_base,
        "yearly": yearly,
        "assumptions": {
            "new_customers_per_year": new_customers_per_year,
            "price": price,
            "activation_rate": activation_rate,
            "annual_churn_rate": annual_churn_rate,
            "duration_years": duration_years,
        },
    }


def calculate_adtech_revenue(annual_impressions, cpm_rate, annual_clicks, cpc_rate, duration_years):
    """
    Ad-tech revenue from an annual CPM (per-mille impression) + CPC (per-click) mix.

    Model (annual periods)
    ----------------------
    - ``annual_cpm_revenue = (annual_impressions / 1000) * cpm_rate``
    - ``annual_cpc_revenue = annual_clicks * cpc_rate``
    - ``annual_revenue     = annual_cpm_revenue + annual_cpc_revenue``
    - ``total_revenue      = annual_revenue * duration_years``

    ``annual_impressions`` and ``annual_clicks`` are per-year volumes, held flat
    across the term.

    Parameters
    ----------
    annual_impressions : number   ad impressions served per year
    cpm_rate : number             revenue per 1,000 impressions (currency)
    annual_clicks : number        clicks per year
    cpc_rate : number             revenue per click (currency)
    duration_years : positive whole int

    Returns
    -------
    dict with total_revenue, annual_revenue, annual_cpm_revenue, annual_cpc_revenue,
    total_cpm_revenue, total_cpc_revenue, yearly (per-year trace), assumptions.
    """
    _require(
        annual_impressions=annual_impressions, cpm_rate=cpm_rate, annual_clicks=annual_clicks,
        cpc_rate=cpc_rate, duration_years=duration_years,
    )
    annual_impressions = _number("annual_impressions", annual_impressions)
    cpm_rate = _number("cpm_rate", cpm_rate)
    annual_clicks = _number("annual_clicks", annual_clicks)
    cpc_rate = _number("cpc_rate", cpc_rate)
    duration_years = _whole_years("duration_years", duration_years)

    annual_cpm_revenue = (annual_impressions / 1000.0) * cpm_rate
    annual_cpc_revenue = annual_clicks * cpc_rate
    annual_revenue = annual_cpm_revenue + annual_cpc_revenue

    yearly = [
        {
            "year": year,
            "cpm_revenue": annual_cpm_revenue,
            "cpc_revenue": annual_cpc_revenue,
            "revenue": annual_revenue,
        }
        for year in range(1, duration_years + 1)
    ]

    return {
        "total_revenue": annual_revenue * duration_years,
        "annual_revenue": annual_revenue,
        "annual_cpm_revenue": annual_cpm_revenue,
        "annual_cpc_revenue": annual_cpc_revenue,
        "total_cpm_revenue": annual_cpm_revenue * duration_years,
        "total_cpc_revenue": annual_cpc_revenue * duration_years,
        "yearly": yearly,
        "assumptions": {
            "annual_impressions": annual_impressions,
            "cpm_rate": cpm_rate,
            "annual_clicks": annual_clicks,
            "cpc_rate": cpc_rate,
            "duration_years": duration_years,
        },
    }


def calculate_fee_based_revenue(annual_exposure_units, rate_per_unit, duration_years):
    """
    Flat fee-per-exposure-unit revenue (e.g. per active user, per API call, per seat).

    Model (annual periods)
    ----------------------
    - ``annual_revenue = annual_exposure_units * rate_per_unit``
    - ``total_revenue  = annual_revenue * duration_years``

    ``annual_exposure_units`` is the number of billable units per year, held flat.
    ``rate_per_unit`` is the fee per unit and is period-independent.

    Parameters
    ----------
    annual_exposure_units : number   billable units per year
    rate_per_unit : number           fee charged per unit (currency)
    duration_years : positive whole int

    Returns
    -------
    dict with total_revenue, annual_revenue, total_units, yearly (per-year trace),
    assumptions.
    """
    _require(
        annual_exposure_units=annual_exposure_units, rate_per_unit=rate_per_unit,
        duration_years=duration_years,
    )
    annual_exposure_units = _number("annual_exposure_units", annual_exposure_units)
    rate_per_unit = _number("rate_per_unit", rate_per_unit)
    duration_years = _whole_years("duration_years", duration_years)

    annual_revenue = annual_exposure_units * rate_per_unit
    yearly = [
        {"year": year, "exposure_units": annual_exposure_units, "revenue": annual_revenue}
        for year in range(1, duration_years + 1)
    ]

    return {
        "total_revenue": annual_revenue * duration_years,
        "annual_revenue": annual_revenue,
        "total_units": annual_exposure_units * duration_years,
        "yearly": yearly,
        "assumptions": {
            "annual_exposure_units": annual_exposure_units,
            "rate_per_unit": rate_per_unit,
            "duration_years": duration_years,
        },
    }


# --------------------------------------------------------------------------- #
# Cost model: cloud hosting (HYBRID - fetch + calculate)
# --------------------------------------------------------------------------- #

_CLOUD_RATE_PROMPT = """You are a cloud pricing analyst. Below are web search results about \
{provider} pricing. Extract ONE current published per-unit price that matches this usage:

Usage unit wanted: {unit_description}

Search results:
{results_block}

Return ONLY a JSON object with exactly these keys:
- "rate_usd": number or null - the price in USD for ONE unit of "{unit_description}". null if the results do not clearly state a matching price.
- "rate_unit": string - the exact unit the rate is per, INCLUDING its time basis (e.g. "per GB-month", "per vCPU-hour", "per million queries"). "" if unknown.
- "quoted_text": string - the sentence or price line from the results you took the number from, quoted verbatim. "" if none.
- "source_result_number": integer or null - which numbered search result above the quote came from.
- "notes": string - one short line on any caveat (region, tier, commitment) affecting this rate.

No markdown, no code fences, no extra keys.
"""


def _build_cloud_query(provider, usage_assumptions):
    explicit = usage_assumptions.get("pricing_query")
    if explicit:
        return str(explicit)
    unit = usage_assumptions.get("unit_description", "")
    return f"{provider} {unit} current published price per unit official pricing"


def calculate_cloud_hosting_cost(provider, usage_assumptions, *, tavily_client=None, groq_client=None,
                                 model="openai/gpt-oss-120b", max_results=5):
    """
    HYBRID: fetch a current cloud per-unit price from the web, then compute usage x rate.

    This is the one non-deterministic function in the module. The *arithmetic* is
    deterministic, but ``fetched_rate`` comes from a live Tavily search + Groq
    extraction, so the result will vary as public pricing pages and search rankings
    change. Treat the number as **directionally useful, not precise**. The return
    value carries ``fetched_rate``, ``source_url`` and ``raw_quoted_text``
    specifically so a human can open the page and verify it by hand.

    Period handling (why a month appears here)
    ------------------------------------------
    Cloud providers publish list prices *per month* (per GB-month, per hour billed
    monthly, per million requests per month, ...). This function keeps the fetched
    rate in that native published unit and annualises it:

        monthly_cost  = monthly_quantity * fetched_rate
        annual_cost   = monthly_cost * 12
        estimated_cost = annual_cost * duration_years

    So you give it your *steady-state monthly consumption* as ``quantity`` and the
    term in *years* as ``duration_years``; everything returned above the raw rate
    is annual or whole-term.

    Parameters
    ----------
    provider : str
        e.g. "AWS", "Snowflake", "GCP". Used only to steer the search.
    usage_assumptions : dict
        Required keys:
          - "quantity" : number       - steady-state units consumed PER MONTH
                                         (matches the monthly unit in unit_description)
          - "unit_description" : str   - what one unit is, including its monthly time
                                         basis, e.g. "GB of Amazon S3 Standard storage
                                         held for a month" or "million BigQuery
                                         on-demand query bytes per month". Drives the search.
        Optional keys:
          - "duration_years" : number  - term length in years, fractional allowed
                                         (default 1). A partial final year is fine.
          - "pricing_query" : str      - override the auto-generated search query
    tavily_client, groq_client :
        Pre-built clients. If omitted, they are constructed from ``TAVILY_API_KEY`` /
        ``GROQ_API_KEY`` in the environment.
    model : str
        Groq model for the extraction step.
    max_results : int
        Number of Tavily results to feed the extractor.

    Returns
    -------
    dict with:
        provider               : str
        estimated_cost         : float   monthly_quantity * fetched_rate * 12 * duration_years
        estimated_annual_cost  : float   monthly_quantity * fetched_rate * 12
        estimated_monthly_cost : float   monthly_quantity * fetched_rate
        fetched_rate           : float   per-unit price extracted from the web (USD, native unit)
        rate_unit              : str     unit the rate is per (incl. time basis), per the extractor
        monthly_quantity       : float   the "quantity" you passed in
        duration_years         : float
        yearly                 : list[dict]  per-year {year, portion_of_year, cost}
        source_url             : str | None  page the quote came from
        raw_quoted_text        : str         verbatim price line for manual verification
        search_query           : str
        extractor_notes        : str
        all_sources            : list[dict]  every {title, url} Tavily returned
        confidence             : "directional"
        disclaimer             : str

    Raises
    ------
    MissingInputError   if provider or a required usage_assumptions key is None/absent.
    FinancialEngineError if no matching rate could be extracted from the search results.
    """
    _require(provider=provider, usage_assumptions=usage_assumptions)
    if not isinstance(usage_assumptions, dict):
        raise FinancialEngineError("usage_assumptions must be a dict")

    quantity = usage_assumptions.get("quantity")
    unit_description = usage_assumptions.get("unit_description")
    _require(quantity=quantity, unit_description=unit_description)
    quantity = _number("usage_assumptions['quantity']", quantity)
    duration_years = _positive_years(
        "usage_assumptions['duration_years']",
        usage_assumptions.get("duration_years", 1),
    )

    if tavily_client is None:
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise FinancialEngineError("TAVILY_API_KEY is not set and no tavily_client was provided")
        tavily_client = TavilyClient(api_key=api_key)

    if groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise FinancialEngineError("GROQ_API_KEY is not set and no groq_client was provided")
        groq_client = Groq(api_key=api_key)

    query = _build_cloud_query(provider, usage_assumptions)
    search = tavily_client.search(query=query, max_results=max_results)
    results = search.get("results", []) if isinstance(search, dict) else []
    if not results:
        raise FinancialEngineError(f"Tavily returned no results for query: {query!r}")

    results_block = "\n\n".join(
        f"[{i}] {r.get('title', '(no title)')}\n"
        f"URL: {r.get('url', '')}\n"
        f"{(r.get('content') or '').strip()}"
        for i, r in enumerate(results, start=1)
    )
    all_sources = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results]

    prompt = _CLOUD_RATE_PROMPT.format(
        provider=provider,
        unit_description=unit_description,
        results_block=results_block,
    )

    parsed = None
    for _attempt in range(2):
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        try:
            parsed = json.loads(response.choices[0].message.content)
            break
        except (json.JSONDecodeError, TypeError):
            continue

    if not isinstance(parsed, dict) or parsed.get("rate_usd") is None:
        raise FinancialEngineError(
            "Could not extract a matching per-unit rate from the search results for "
            f"{unit_description!r} ({provider}). The caller should ask the user to supply "
            "the rate manually or point at a specific pricing page."
        )

    fetched_rate = _number("fetched_rate", parsed["rate_usd"])
    src_num = parsed.get("source_result_number")
    source_url = None
    if isinstance(src_num, int) and 1 <= src_num <= len(results):
        source_url = results[src_num - 1].get("url")

    estimated_monthly_cost = quantity * fetched_rate
    estimated_annual_cost = estimated_monthly_cost * 12
    estimated_cost = estimated_annual_cost * duration_years
    yearly = [
        {"year": year, "portion_of_year": frac, "cost": estimated_annual_cost * frac}
        for year, frac in _year_slices(duration_years)
    ]

    return {
        "provider": provider,
        "estimated_cost": estimated_cost,
        "estimated_annual_cost": estimated_annual_cost,
        "estimated_monthly_cost": estimated_monthly_cost,
        "fetched_rate": fetched_rate,
        "rate_unit": str(parsed.get("rate_unit", "")).strip(),
        "monthly_quantity": quantity,
        "duration_years": duration_years,
        "yearly": yearly,
        "source_url": source_url,
        "raw_quoted_text": str(parsed.get("quoted_text", "")).strip(),
        "search_query": query,
        "extractor_notes": str(parsed.get("notes", "")).strip(),
        "all_sources": all_sources,
        "confidence": "directional",
        "disclaimer": (
            "Rate was fetched from a public web search and extracted by an LLM. "
            "It is directionally useful only - verify 'fetched_rate' (native published "
            "unit, see 'rate_unit') against the provider's official pricing page "
            "(source_url / raw_quoted_text) before using this number in a real model."
        ),
    }


# --------------------------------------------------------------------------- #
# Cost model: software / IP licensing (deterministic; optional rough web check)
# --------------------------------------------------------------------------- #

_VOLUME_LEADING_NUMBER_RE = re.compile(r"([\d][\d,\.]*)")
_VOLUME_PERIOD_RE = re.compile(
    r"per\s+(?:(\d+)\s*[- ])?\s*(month|months|year|years|12-month|12\s*month|annum|quarter|quarters)",
    re.IGNORECASE,
)


def derive_licensing_inputs_from_extraction(extracted):
    """
    Pull ``duration_years`` and ``volume`` out of a termsheet_extractor result so
    the caller doesn't re-ask for fields the term sheet already contains.

    Reads:
      - ``extracted["contract_duration_months"]`` -> ``duration_years`` (``/ 12``;
        may be fractional, e.g. an 18-month term -> 1.5). ``None`` if absent.
      - ``extracted["minimum_volume"]``           -> parsed into a total unit count
        over the contract term, plus the per-year rate, when the free text parses.

    ``minimum_volume`` is free text (e.g. "50,000 units per 12-month period"). This
    does a best-effort parse:
      - leading number                    -> volume for one stated period
      - "per ... month/quarter/year/annum" -> length of that period, in years
      - normalised to a per-year figure, then scaled by ``duration_years``

    Anything it cannot parse comes back as ``None`` with a note - it never guesses.
    The caller decides whether to ask the user or accept a labelled assumption.

    Returns
    -------
    dict with:
        duration_years      : float | None   contract term in years
        volume              : float | None   total units over the whole term
        volume_per_year     : float | None   the stated commitment normalised to per-year
        volume_source_text  : str | None     the raw minimum_volume string
        volume_parse_note   : str            what happened during parsing
    """
    if not isinstance(extracted, dict):
        raise FinancialEngineError("extracted must be a dict from termsheet_extractor")

    months = extracted.get("contract_duration_months")
    if isinstance(months, bool) or not isinstance(months, (int, float)):
        duration_years = None
    else:
        duration_years = months / 12.0
        if duration_years == int(duration_years):
            duration_years = float(int(duration_years))

    raw = extracted.get("minimum_volume")
    out = {
        "duration_years": duration_years,
        "volume": None,
        "volume_per_year": None,
        "volume_source_text": raw if isinstance(raw, str) else None,
        "volume_parse_note": "",
    }

    if not isinstance(raw, str) or not raw.strip():
        out["volume_parse_note"] = "No minimum_volume text in the extraction to parse."
        return out

    num_match = _VOLUME_LEADING_NUMBER_RE.search(raw)
    if not num_match:
        out["volume_parse_note"] = f"Could not find a numeric quantity in {raw!r}."
        return out

    try:
        per_period = float(num_match.group(1).replace(",", ""))
    except ValueError:
        out["volume_parse_note"] = f"Could not parse the quantity {num_match.group(1)!r} as a number."
        return out

    period_match = _VOLUME_PERIOD_RE.search(raw)
    if period_match:
        count = int(period_match.group(1)) if period_match.group(1) else 1
        unit = period_match.group(2).lower()
        if "year" in unit or "annum" in unit or "12" in unit:
            period_years = 1.0 * count
        elif "quarter" in unit:
            period_years = 0.25 * count
        else:  # month(s)
            period_years = (1.0 / 12.0) * count
    else:
        period_years = None

    if period_years is None:
        out["volume_parse_note"] = (
            f"Parsed a quantity of {per_period:g} from {raw!r} but not the period it "
            "applies to. Caller must confirm whether this is per-year, per-term, or "
            "per-month before it can be used as a volume."
        )
        return out

    volume_per_year = per_period / period_years
    out["volume_per_year"] = volume_per_year

    if duration_years is None:
        out["volume_parse_note"] = (
            f"Parsed {per_period:g} units per {period_years:g} year(s) "
            f"(= {volume_per_year:g}/year) from {raw!r}, but contract_duration_months "
            "is missing so the total term volume cannot be computed."
        )
        return out

    out["volume"] = volume_per_year * duration_years
    out["volume_parse_note"] = (
        f"Parsed {per_period:g} units per {period_years:g} year(s) (= {volume_per_year:g}/year) "
        f"from {raw!r}; scaled to {out['volume']:g} over the {duration_years:g}-year term. "
        "This is a MINIMUM commitment - actual volume may be higher."
    )
    return out


def _tiered_cost(volume, tier_breaks):
    """
    Marginal (bracketed) tier pricing.

    ``tier_breaks`` is a list of ``{"min_units": <int>, "rate_per_unit": <number>}``.
    Units from ``min_units`` up to the next tier's ``min_units`` are billed at that
    tier's rate; the last tier runs to infinity. The list is sorted here; a tier
    starting at 0 is required. ``volume`` is the total licensed units over the term.

    Returns ``(total_cost, breakdown)`` where breakdown is a list of
    ``{from_unit, to_unit, units_in_tier, rate_per_unit, tier_cost}``.
    """
    tiers = sorted(
        ({"min_units": _number("tier min_units", t["min_units"]),
          "rate_per_unit": _number("tier rate_per_unit", t["rate_per_unit"])}
         for t in tier_breaks),
        key=lambda t: t["min_units"],
    )
    if not tiers:
        raise FinancialEngineError("tier_breaks is empty")
    if tiers[0]["min_units"] > 0:
        raise FinancialEngineError("tier_breaks must include a tier starting at min_units=0")

    total = 0.0
    breakdown = []
    for idx, tier in enumerate(tiers):
        start = tier["min_units"]
        end = tiers[idx + 1]["min_units"] if idx + 1 < len(tiers) else float("inf")
        if volume <= start:
            break
        units_in_tier = min(volume, end) - start
        tier_cost = units_in_tier * tier["rate_per_unit"]
        total += tier_cost
        breakdown.append({
            "from_unit": start,
            "to_unit": None if end == float("inf") else end,
            "units_in_tier": units_in_tier,
            "rate_per_unit": tier["rate_per_unit"],
            "tier_cost": tier_cost,
        })
    return total, breakdown


_LICENSE_SANITY_PROMPT = """You are a technology-licensing analyst. Below are web search results \
about public / list pricing for: {product_description}

Search results:
{results_block}

Return ONLY a JSON object with exactly these keys:
- "typical_low_usd": number or null - low end of the typical PUBLIC per-unit license price, USD.
- "typical_high_usd": number or null - high end of the typical PUBLIC per-unit license price, USD.
- "basis": string - what one unit is (e.g. "per GPU", "per vehicle", "per seat/year"). "" if unclear.
- "quoted_text": string - the price line you used, quoted verbatim. "" if none.
- "source_result_number": integer or null - which numbered result the quote came from.

No markdown, no code fences, no extra keys.
"""


def _licensing_sanity_check(effective_per_unit, product_description, tavily_client, groq_client,
                            model, max_results):
    """Rough, clearly-labelled public-pricing comparison. NOT a verification. See caller."""
    search = tavily_client.search(
        query=f"{product_description} license fee per unit typical list price",
        max_results=max_results,
    )
    results = search.get("results", []) if isinstance(search, dict) else []
    if not results:
        return {
            "status": "inconclusive",
            "note": "No public pricing results found for a comparison.",
            "label": "ROUGH PUBLIC-PRICING COMPARISON - NOT A VERIFICATION",
        }

    results_block = "\n\n".join(
        f"[{i}] {r.get('title', '')}\nURL: {r.get('url', '')}\n{(r.get('content') or '').strip()}"
        for i, r in enumerate(results, start=1)
    )
    parsed = None
    for _attempt in range(2):
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _LICENSE_SANITY_PROMPT.format(
                product_description=product_description, results_block=results_block)}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        try:
            parsed = json.loads(response.choices[0].message.content)
            break
        except (json.JSONDecodeError, TypeError):
            continue

    label = "ROUGH PUBLIC-PRICING COMPARISON - NOT A VERIFICATION"
    if not isinstance(parsed, dict) or parsed.get("typical_low_usd") is None:
        return {
            "status": "inconclusive",
            "note": "Could not extract a comparable public price from the results.",
            "label": label,
        }

    low = float(parsed["typical_low_usd"])
    high = float(parsed.get("typical_high_usd") or low)
    src_num = parsed.get("source_result_number")
    source_url = (
        results[src_num - 1].get("url")
        if isinstance(src_num, int) and 1 <= src_num <= len(results) else None
    )

    if effective_per_unit < 0.5 * low:
        verdict = "negotiated rate looks LOW vs public list pricing"
    elif effective_per_unit > 2.0 * high:
        verdict = "negotiated rate looks HIGH vs public list pricing"
    else:
        verdict = "negotiated rate is roughly in line with public list pricing"

    return {
        "status": "unusual" if verdict.startswith("negotiated rate looks") else "in_line",
        "verdict": verdict,
        "negotiated_effective_per_unit": effective_per_unit,
        "public_typical_low_usd": low,
        "public_typical_high_usd": high,
        "public_basis": str(parsed.get("basis", "")).strip(),
        "quoted_text": str(parsed.get("quoted_text", "")).strip(),
        "source_url": source_url,
        "label": label,
        "note": (
            "Public list pricing rarely matches a negotiated OEM deal (volume discounts, "
            "bundling, strategic terms). Use this only to decide whether to double-check "
            "the negotiated number, not as evidence it is right or wrong."
        ),
    }


def calculate_licensing_cost(model_type, flat_fee, per_unit_fee, volume, duration_years,
                             tier_breaks=None, *, run_sanity_check=False, product_description=None,
                             tavily_client=None, groq_client=None, model="openai/gpt-oss-120b",
                             max_results=5):
    """
    Cost of a software / IP license an OEM pays to a provider (e.g. an NVIDIA-style
    per-unit chip/software license, or a flat platform license).

    Deterministic by default. If ``run_sanity_check=True`` it *also* does a rough,
    clearly-labelled Tavily/Groq comparison against public list pricing - that
    comparison is advisory only and never changes the computed cost.

    Models (annual periods)
    -----------------------
    model_type == "flat":
        ``licensing_cost = flat_fee * duration_years``
        ``flat_fee`` is the ANNUAL license fee (currency / year). ``duration_years``
        may be fractional; the ``yearly`` trace then ends with a partial-year row.
        Requires: flat_fee, duration_years.

    model_type == "per_unit":
        ``licensing_cost = per_unit_fee * volume``
        ``volume`` is the total licensed units over the WHOLE term (use
        ``derive_licensing_inputs_from_extraction`` to get it from the term sheet).
        Duration does not enter the arithmetic. Requires: per_unit_fee, volume.

    model_type == "tiered":
        Marginal bracket pricing over ``tier_breaks`` (see ``_tiered_cost``): the
        first N units at tier-1 rate, the next block at tier-2 rate, etc.
        ``licensing_cost = Σ (units_in_bracket * bracket_rate)`` over total-term
        ``volume``. Duration does not enter the arithmetic.
        Requires: volume, tier_breaks.

    Parameters
    ----------
    model_type : {"flat", "per_unit", "tiered"}
    flat_fee : number | None            annual flat license fee (currency / year)
    per_unit_fee : number | None        royalty per licensed unit (currency / unit)
    volume : number | None              total licensed units over the whole term
    duration_years : number | None      contract term in years (fractional allowed;
                                        for "flat" it must be > 0, for "per_unit"/
                                        "tiered" it is echoed but unused)
    tier_breaks : list[dict] | None     [{"min_units": int, "rate_per_unit": number}, ...]
    run_sanity_check : bool             do the rough public-pricing comparison
    product_description : str | None    required if run_sanity_check (what is being licensed)
    tavily_client, groq_client, model, max_results
                                        only used when run_sanity_check is True

    Returns
    -------
    dict with:
        model_type          : str
        licensing_cost      : float          over the whole term
        annual_cost         : float | None   licensing_cost / duration_years when duration known
        effective_per_unit  : float | None   licensing_cost / volume when volume known
        duration_years      : number | None
        volume              : number | None
        yearly              : list | None    per-year {year, portion_of_year, cost} for "flat"
        breakdown           : list | None    per-tier breakdown for "tiered"
        assumptions         : dict
        sanity_check        : dict | None    present only when run_sanity_check=True

    Raises
    ------
    MissingInputError    if an input required by the chosen model_type is None.
    FinancialEngineError on an unknown model_type or malformed tier_breaks.
    """
    _require(model_type=model_type)
    model_type = str(model_type).strip().lower()

    breakdown = None
    yearly = None
    effective_per_unit = None
    annual_cost = None
    duration_years_out = duration_years

    if model_type == "flat":
        _require(flat_fee=flat_fee, duration_years=duration_years)
        flat_fee_v = _number("flat_fee", flat_fee)
        years = _positive_years("duration_years", duration_years)
        duration_years_out = years
        licensing_cost = flat_fee_v * years
        annual_cost = flat_fee_v
        yearly = [
            {"year": year, "portion_of_year": frac, "cost": flat_fee_v * frac}
            for year, frac in _year_slices(years)
        ]

    elif model_type == "per_unit":
        _require(per_unit_fee=per_unit_fee, volume=volume)
        per_unit_v = _number("per_unit_fee", per_unit_fee)
        volume_v = _number("volume", volume)
        licensing_cost = per_unit_v * volume_v
        effective_per_unit = per_unit_v
        if duration_years is not None:
            years = _positive_years("duration_years", duration_years)
            duration_years_out = years
            annual_cost = licensing_cost / years

    elif model_type == "tiered":
        _require(volume=volume, tier_breaks=tier_breaks)
        volume_v = _number("volume", volume)
        if not isinstance(tier_breaks, list) or not tier_breaks:
            raise FinancialEngineError("tier_breaks must be a non-empty list for model_type='tiered'")
        licensing_cost, breakdown = _tiered_cost(volume_v, tier_breaks)
        effective_per_unit = licensing_cost / volume_v if volume_v else None
        if duration_years is not None:
            years = _positive_years("duration_years", duration_years)
            duration_years_out = years
            annual_cost = licensing_cost / years

    else:
        raise FinancialEngineError(
            f"Unknown model_type {model_type!r}; expected 'flat', 'per_unit', or 'tiered'"
        )

    result = {
        "model_type": model_type,
        "licensing_cost": licensing_cost,
        "annual_cost": annual_cost,
        "effective_per_unit": effective_per_unit,
        "duration_years": duration_years_out,
        "volume": volume,
        "yearly": yearly,
        "breakdown": breakdown,
        "assumptions": {
            "flat_fee_annual": flat_fee,
            "per_unit_fee": per_unit_fee,
            "volume": volume,
            "duration_years": duration_years,
            "tier_breaks": tier_breaks,
        },
        "sanity_check": None,
    }

    if run_sanity_check:
        _require(product_description=product_description)
        if effective_per_unit is None:
            result["sanity_check"] = {
                "status": "not_applicable",
                "note": "A flat-fee license has no per-unit rate to compare against public pricing.",
                "label": "ROUGH PUBLIC-PRICING COMPARISON - NOT A VERIFICATION",
            }
        else:
            if tavily_client is None:
                from tavily import TavilyClient
                api_key = os.getenv("TAVILY_API_KEY")
                if not api_key:
                    raise FinancialEngineError("TAVILY_API_KEY is not set and no tavily_client was provided")
                tavily_client = TavilyClient(api_key=api_key)
            if groq_client is None:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise FinancialEngineError("GROQ_API_KEY is not set and no groq_client was provided")
                groq_client = Groq(api_key=api_key)
            result["sanity_check"] = _licensing_sanity_check(
                effective_per_unit, product_description, tavily_client, groq_client, model, max_results
            )

    return result


# --------------------------------------------------------------------------- #
# Universal tools (deterministic)
# --------------------------------------------------------------------------- #

def calculate_profit(revenue, capex, opex):
    """
    Profit summary with CAPEX and OPEX tracked separately.

    All three inputs cover the SAME period (by this module's convention, either a
    single year or the whole term - just be consistent).

    Definitions
    -----------
    - ``gross_profit  = revenue - opex``                  (operating result before capital costs)
    - ``net_profit    = revenue - opex - capex``
    - ``gross_margin_pct = gross_profit / revenue * 100`` (None if revenue == 0)
    - ``net_margin_pct   = net_profit  / revenue * 100``  (None if revenue == 0)

    Parameters
    ----------
    revenue : number   total revenue over the period (currency)
    capex : number     capital expenditure over the period (currency)
    opex : number      operating expenditure over the period (currency)

    Returns
    -------
    dict with revenue, capex, opex, gross_profit, net_profit,
    gross_margin_pct, net_margin_pct.
    """
    _require(revenue=revenue, capex=capex, opex=opex)
    revenue = _number("revenue", revenue, allow_negative=True)
    capex = _number("capex", capex, allow_negative=True)
    opex = _number("opex", opex, allow_negative=True)

    gross_profit = revenue - opex
    net_profit = revenue - opex - capex

    gross_margin_pct = (gross_profit / revenue * 100) if revenue else None
    net_margin_pct = (net_profit / revenue * 100) if revenue else None

    return {
        "revenue": revenue,
        "capex": capex,
        "opex": opex,
        "gross_profit": gross_profit,
        "net_profit": net_profit,
        "gross_margin_pct": gross_margin_pct,
        "net_margin_pct": net_margin_pct,
    }


def calculate_npv(cash_flows, discount_rate=0.10):
    """
    Net present value of a series of ANNUAL cash flows.

    Formula
    -------
    ``NPV = Σ_{t=0}^{n} CF_t / (1 + discount_rate) ** t``   (t in years)

    ``cash_flows[0]`` is the cash flow at t=0 (typically the negative initial
    investment, undiscounted). ``cash_flows[t]`` is the net cash flow at the end of
    year ``t``. ``discount_rate`` is the ANNUAL discount rate.

    Parameters
    ----------
    cash_flows : list[number]   non-empty; one entry per year (t=0 first). Every
                                element must be a real number (0 for a zero year).
    discount_rate : number      annual discount rate (default 0.10 = 10%). Must be > -1.

    Returns
    -------
    dict with npv, discount_rate, discounted_by_year (list of per-year present values).

    Raises
    ------
    MissingInputError    if cash_flows is None, or any element is None.
    FinancialEngineError if cash_flows is empty or discount_rate <= -1.
    """
    _require(cash_flows=cash_flows, discount_rate=discount_rate)
    if not isinstance(cash_flows, (list, tuple)) or len(cash_flows) == 0:
        raise FinancialEngineError("cash_flows must be a non-empty list of numbers")
    if any(cf is None for cf in cash_flows):
        raise MissingInputError(
            "cash_flows contains a None entry - every year's cash flow must be provided "
            "(use 0 for a genuinely-zero year, not None)."
        )
    discount_rate = _number("discount_rate", discount_rate, allow_negative=True)
    if discount_rate <= -1:
        raise FinancialEngineError(f"discount_rate must be > -1, got {discount_rate}")

    discounted = []
    for t, cf in enumerate(cash_flows):
        cf = _number(f"cash_flows[{t}]", cf, allow_negative=True)
        discounted.append(cf / ((1 + discount_rate) ** t))

    return {
        "npv": sum(discounted),
        "discount_rate": discount_rate,
        "discounted_by_year": discounted,
    }


def calculate_roi(net_profit, total_investment):
    """
    Return on investment, as a percentage.

    Formula
    -------
    ``roi_pct = net_profit / total_investment * 100``

    ``net_profit`` is normally the whole-term net profit; ``total_investment`` the
    whole-term capital deployed. (Divide net_profit by the term in years first if
    you want an annualised figure.)

    Parameters
    ----------
    net_profit : number          net profit (currency); may be negative.
    total_investment : number    total capital deployed (currency); must be > 0.

    Returns
    -------
    dict with roi_pct, net_profit, total_investment.

    Raises
    ------
    MissingInputError    if either input is None.
    FinancialEngineError if total_investment <= 0 (ROI is undefined).
    """
    _require(net_profit=net_profit, total_investment=total_investment)
    net_profit = _number("net_profit", net_profit, allow_negative=True)
    total_investment = _number("total_investment", total_investment, allow_negative=True)
    if total_investment <= 0:
        raise FinancialEngineError(
            f"total_investment must be greater than 0 to compute ROI, got {total_investment}"
        )

    return {
        "roi_pct": net_profit / total_investment * 100,
        "net_profit": net_profit,
        "total_investment": total_investment,
    }


def _pl_records(pl_data):
    """Normalise ``pl_data`` into a list of {period, series, amount} records for plotting."""
    if isinstance(pl_data, dict):
        # Either {"Y1": {...}, "Y2": {...}} or a single flat {"revenue": ..., ...}
        if pl_data and all(isinstance(v, dict) for v in pl_data.values()):
            rows = [{"period": str(k), **v} for k, v in pl_data.items()]
        else:
            rows = [{"period": "Total", **pl_data}]
    elif isinstance(pl_data, (list, tuple)):
        rows = []
        for i, item in enumerate(pl_data, start=1):
            if not isinstance(item, dict):
                raise FinancialEngineError("each pl_data row must be a dict")
            label = item.get("period") or item.get("year") or item.get("label") \
                or item.get("name") or f"Y{i}"
            rows.append({"period": str(label), **{k: v for k, v in item.items()
                                                  if k not in ("period", "year", "label", "name")}})
    else:
        raise FinancialEngineError("pl_data must be a dict or a list of dicts")

    records = []
    for row in rows:
        period = row["period"]
        for series, amount in row.items():
            if series == "period":
                continue
            if isinstance(amount, bool) or not isinstance(amount, Real):
                continue
            records.append({"period": period, "series": series, "amount": float(amount)})
    if not records:
        raise FinancialEngineError("pl_data contained no numeric series to plot")
    return records


def generate_pl_chart(pl_data, *, title="Partnership P&L (annual)"):
    """
    Build a grouped-bar P&L chart as an Altair chart object.

    Returns the chart only - it does NOT render. Wire it into Streamlit later with
    ``st.altair_chart(generate_pl_chart(pl_data), use_container_width=True)``.

    Accepted ``pl_data`` shapes
    ---------------------------
    - list of period dicts (typically one per year):
        ``[{"year": 1, "revenue": 100, "opex": 60, "capex": 20, "net_profit": 20}, ...]``
      (the label key may be "year", "period", "label", or "name"; any other numeric
      key becomes a bar series)
    - dict of period dicts:
        ``{"Y1": {"revenue": 100, ...}, "Y2": {...}}``
    - a single flat dict:
        ``{"revenue": 100, "opex": 60, "capex": 20, "net_profit": 20}``
      (rendered as one period labelled "Total")

    Returns
    -------
    altair.Chart

    Raises
    ------
    FinancialEngineError if pl_data has no numeric series or a bad shape.
    """
    _require(pl_data=pl_data)
    records = _pl_records(pl_data)
    df = pd.DataFrame.from_records(records)

    period_order = list(dict.fromkeys(df["period"]))
    series_order = list(dict.fromkeys(df["series"]))

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("period:N", sort=period_order, title=None),
            xOffset=alt.XOffset("series:N", sort=series_order),
            y=alt.Y("amount:Q", title="Amount"),
            color=alt.Color("series:N", sort=series_order, title="Line item"),
            tooltip=[
                alt.Tooltip("period:N", title="Period"),
                alt.Tooltip("series:N", title="Line item"),
                alt.Tooltip("amount:Q", title="Amount", format=",.2f"),
            ],
        )
        .properties(title=title, height=360)
    )


# --------------------------------------------------------------------------- #
# Direct-run self test: every deterministic function with hardcoded values.
# Run:  python financial_engine.py
# The hybrid cloud function is only exercised with --live (needs API keys).
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-test the financial engine with hardcoded values")
    parser.add_argument("--live", action="store_true",
                        help="also call calculate_cloud_hosting_cost (needs TAVILY_API_KEY + GROQ_API_KEY)")
    args = parser.parse_args()

    def show(label, value):
        print(f"\n=== {label} ===")
        print(json.dumps(value, indent=2, default=str))

    # --- Revenue models (annual) ------------------------------------------
    sub = calculate_subscription_revenue(
        new_customers_per_year=60_000, price=120.0, activation_rate=0.4,
        annual_churn_rate=0.30, duration_years=3,
    )
    show("subscription_revenue (60k new/yr, 40% activate, 30%/yr churn, $120/yr, 3yr)",
         {k: v for k, v in sub.items() if k != "yearly"})
    for row in sub["yearly"]:
        print(f"  year {row['year']}: base {row['active_base']:.1f}  revenue {row['revenue']:.2f}")

    ad = calculate_adtech_revenue(
        annual_impressions=60_000_000, cpm_rate=4.5, annual_clicks=300_000,
        cpc_rate=1.2, duration_years=3,
    )
    show("adtech_revenue (60M impr/yr @ $4.50 CPM + 300k clicks/yr @ $1.20 CPC, 3yr)", ad)

    fee = calculate_fee_based_revenue(annual_exposure_units=600_000, rate_per_unit=2.5, duration_years=3)
    show("fee_based_revenue (600k units/yr @ $2.50, 3yr)", fee)

    # --- Licensing cost -------------------------------------------------
    show("licensing_cost flat ($1.2M/yr, 3yr)",
         calculate_licensing_cost("flat", flat_fee=1_200_000, per_unit_fee=None,
                                  volume=None, duration_years=3))
    show("licensing_cost flat, partial term ($1.2M/yr, 1.5yr)",
         calculate_licensing_cost("flat", flat_fee=1_200_000, per_unit_fee=None,
                                  volume=None, duration_years=1.5))
    show("licensing_cost per_unit ($75/unit x 150k units over 3yr)",
         calculate_licensing_cost("per_unit", flat_fee=None, per_unit_fee=75,
                                  volume=150_000, duration_years=3))
    show("licensing_cost tiered (150k units: 0-50k@100, 50k-100k@80, 100k+@60)",
         calculate_licensing_cost("tiered", flat_fee=None, per_unit_fee=None,
                                  volume=150_000, duration_years=3,
                                  tier_breaks=[
                                      {"min_units": 0, "rate_per_unit": 100},
                                      {"min_units": 50_000, "rate_per_unit": 80},
                                      {"min_units": 100_000, "rate_per_unit": 60},
                                  ]))

    # --- Derive licensing inputs from a term sheet extraction ----------
    example_extraction = {
        "contract_duration_months": 36,
        "minimum_volume": "50,000 units per 12-month period",
    }
    show("derive_licensing_inputs_from_extraction (36-month term, 50k/yr minimum)",
         derive_licensing_inputs_from_extraction(example_extraction))

    # --- Universal tools ----------------------------------------------
    profit = calculate_profit(revenue=fee["total_revenue"], capex=400_000, opex=2_700_000)
    show("profit (3yr fee revenue, $400k capex, $2.7M opex)", profit)

    show("npv (-1.0M now, then +400k/yr for 4 yrs @ 10%)",
         calculate_npv([-1_000_000, 400_000, 400_000, 400_000, 400_000], discount_rate=0.10))

    show("roi ($600k net profit on $1.3M investment)",
         calculate_roi(net_profit=600_000, total_investment=1_300_000))

    # --- Missing-input behaviour ----------------------------------------
    try:
        calculate_subscription_revenue(new_customers_per_year=None, price=120.0,
                                       activation_rate=0.4, annual_churn_rate=0.3, duration_years=3)
    except MissingInputError as e:
        print(f"\n=== missing-input guard works ===\n{e}")

    # --- Chart --------------------------------------------------------
    chart = generate_pl_chart([
        {"year": 1, "revenue": 1_800_000, "opex": 900_000, "capex": 400_000, "net_profit": 500_000},
        {"year": 2, "revenue": 2_400_000, "opex": 1_000_000, "capex": 100_000, "net_profit": 1_300_000},
        {"year": 3, "revenue": 3_000_000, "opex": 1_100_000, "capex": 100_000, "net_profit": 1_800_000},
    ])
    print(f"\n=== generate_pl_chart -> {type(chart).__name__} "
          f"({len(chart.data)} rows, mark={chart.mark}) ===")

    # --- Hybrid (opt-in) --------------------------------------------
    if args.live:
        from dotenv import load_dotenv
        load_dotenv()
        show("calculate_cloud_hosting_cost (LIVE: AWS S3 Standard, 10 TB stored/mo, 3yr)",
             calculate_cloud_hosting_cost(
                 provider="AWS",
                 usage_assumptions={
                     "quantity": 10 * 1024,  # GB held per month
                     "unit_description": "Amazon S3 Standard storage per GB-month (US East)",
                     "duration_years": 3,
                 },
             ))
    else:
        print("\n(skip calculate_cloud_hosting_cost - pass --live with API keys to exercise it)")
