"""
Eval harness for business_case.py — exercised directly, NOT through Streamlit.

Runs the current grid-based flow end to end for five hand-built deals:

    classify_business_model         ->  must return the expected model
    prefill_grid_cells              ->  term-sheet volume lands in the right cells
    recompute_subscription_waterfall->  ending = beginning + new - churned, rolled forward
    propose_grid_assumptions        ->  fills ONLY blank cells; never overwrites a
                                        pre-filled or user-entered value
    (assembled grid)                ->  no cell reaching financial_engine may be None
                                        (build_pl_from_grid silently treats None as 0)
    build_pl_from_grid              ->  per-year revenue must vary where the deal's
    + compute_npv_roi                   economics say it should, and the P&L identities
                                        must hold

Prints a pass/fail line per check, a verdict per case, and a summary count.
Exit code is 0 only if every case passes.

Needs GROQ_API_KEY in the environment / .env (classify + propose call Groq).
No new dependencies — plain script, stdlib + business_case + financial_engine.

    python eval_business_case_flow.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from groq import Groq

import business_case as bc
from financial_engine import FinancialEngineError


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #
# Each case: a term-sheet extraction + risk register, the expected classification,
# and `user_cells` = the real figures a user would type into the grid (the ones that
# can be neither pre-filled from the term sheet nor responsibly assumed). The harness
# then lets prefill + propose_grid_assumptions fill the rest, exactly like the UI flow.
#
# user_cells values: a scalar applies to every year 0..N; a dict applies per year index.

def _extraction(**overrides):
    base = {
        "partner_name": None,
        "contract_duration_months": None,
        "revenue_share": None,
        "revenue_share_details": None,
        "minimum_volume": None,
        "exclusivity": None,
        "payment_terms": None,
        "termination_terms": None,
        "renewal_terms": None,
        "IP_Ownership": None,
    }
    base.update(overrides)
    return base


CASES = [
    {
        "name": "subscription_streaming",
        "expected_model": "subscription",
        "extraction": _extraction(
            partner_name="Northwind Media Group",
            contract_duration_months=24,
            revenue_share="Provider earns a recurring monthly subscription fee of $12.99 per active subscriber",
            revenue_share_details=(
                "Fee is billed monthly per active subscriber. Provider bears activation and "
                "retention; revenue is recognised monthly, net of churn. No per-unit licensing "
                "and no CPM/CPC component."
            ),
            minimum_volume="150,000 new subscribers per 12-month period",
            exclusivity="Non-exclusive",
            payment_terms="Net 30 from monthly invoice",
            termination_terms="Either party on 90 days' notice",
            renewal_terms="Auto-renews for 12-month terms",
        ),
        "risks": [
            {"risk_name": "Missing contract term: IP_Ownership", "severity": "high"},
            {"risk_name": "Minimum volume commitment without stated penalty/waiver terms", "severity": "medium"},
        ],
        # price per active subscriber per YEAR (12.99 * 12); can't be assumed from norms.
        # beginning_base year 0 = 0 -> brand-new partnership, no existing base.
        "user_cells": {"price": 155.88, "beginning_base": {0: 0.0}},
        "expect_revenue_variation": True,
    },
    {
        "name": "adtech_display_network",
        "expected_model": "adtech",
        "extraction": _extraction(
            partner_name="Beacon Ad Exchange",
            contract_duration_months=36,
            revenue_share=(
                "Partner is compensated on a CPM basis (dollars per thousand impressions) for "
                "display ad inventory, plus a CPC performance component for clicks. No subscription "
                "or per-unit licensing."
            ),
            revenue_share_details="Monthly settlement of served impressions and attributed clicks.",
            minimum_volume=None,
            exclusivity="Non-exclusive",
            payment_terms="Net 45",
            termination_terms="Either party on 60 days' notice",
        ),
        "risks": [
            {"risk_name": "Missing contract term: minimum_volume", "severity": "low"},
        ],
        # raw traffic volumes can't be assumed; CPM/CPC are left for the assumption pass.
        "user_cells": {"annual_impressions": 800_000_000, "annual_clicks": 4_000_000},
        "expect_revenue_variation": True,
    },
    {
        "name": "fee_based_metered_api",
        "expected_model": "fee_based",
        "extraction": _extraction(
            partner_name="Halcyon Data Exchange",
            contract_duration_months=None,  # deliberately absent -> default_grid_years falls back to 3
            revenue_share="Provider receives a flat fee of $0.004 per API call served",
            revenue_share_details=(
                "A single flat per-call fee, metered monthly. No revenue share, no CPM/CPC, "
                "no per-seat or subscriber activation/churn mechanic, no software licence."
            ),
            # Deliberately absent: with no volume commitment in the term sheet there is
            # nothing to pre-fill, so call volume must come from the assumption engine --
            # which is what makes the variation check below actually exercise it.
            minimum_volume=None,
            exclusivity="Exclusive within North America",
            payment_terms="Monthly invoicing in arrears on metered call volume",
        ),
        "risks": [
            {"risk_name": "Exclusivity commitment", "severity": "high"},
            {"risk_name": "Missing contract term: contract_duration_months", "severity": "medium"},
            {"risk_name": "Missing contract term: minimum_volume", "severity": "low"},
        ],
        # per-call fee is negotiated, not assumable; call volume is left to the engine.
        "user_cells": {"rate_per_unit": 0.004},
        # Revenue = assumed call volume x pinned rate, so a flat result here would mean
        # the assumption engine returned a lazy constant series instead of a ramp.
        "expect_revenue_variation": True,
    },
    {
        "name": "licensing_per_unit_embedded_sdk",
        "expected_model": "licensing",
        "license_model_type": "per_unit",
        "extraction": _extraction(
            partner_name="Vanta Software",
            contract_duration_months=None,
            revenue_share=(
                "Licensee pays Provider a per-unit software licence royalty of $60 for each "
                "unit shipped with the Provider embedded SDK."
            ),
            revenue_share_details="Royalty per unit shipped. Provider retains all platform IP.",
            minimum_volume=None,             # -> volume must come from user_cells
            exclusivity=None,
            payment_terms="Royalty reported and paid quarterly",
            IP_Ownership="Provider retains all IP; Licensee receives a non-transferable licence.",
        ),
        "risks": [
            {"risk_name": "Missing contract term: exclusivity", "severity": "high"},
        ],
        "user_cells": {"rate_or_fee": 60.0, "volume": 400_000},
        "expect_revenue_variation": False,  # user pins both drivers flat across years
    },
    {
        "name": "licensing_flat_platform_fee",
        "expected_model": "licensing",
        "license_model_type": "flat",
        "extraction": _extraction(
            partner_name="Helix Platform",
            contract_duration_months=36,
            revenue_share=(
                "Licensee pays Provider a fixed annual platform licence fee of $4,000,000, invoiced "
                "quarterly. There is no per-unit or revenue-share component."
            ),
            revenue_share_details="Flat annual fee, independent of volume.",
            minimum_volume=None,
            exclusivity=None,
            payment_terms="Quarterly instalments",
            IP_Ownership="Provider owns platform IP; Licensee granted a non-transferable licence.",
        ),
        "risks": [
            {"risk_name": "Missing contract term: termination_terms", "severity": "high"},
        ],
        # A flat annual fee is flat by definition, so revenue SHOULD be constant here.
        "user_cells": {"rate_or_fee": 4_000_000.0},
        "expect_revenue_variation": False,
    },
]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

class CheckLog:
    """Collects pass/fail lines for one case."""

    def __init__(self):
        self.lines = []
        self.ok = True

    def check(self, label, passed, detail=""):
        self.ok = self.ok and passed
        status = "PASS" if passed else "FAIL"
        dots = "." * max(3, 34 - len(label))
        self.lines.append(f"      {label} {dots} {status}" + (f"  ({detail})" if detail else ""))

    def dump(self):
        for line in self.lines:
            print(line)


def _row_specs(model, license_model_type=None):
    """Model rows + CAPEX + a single OPEX line item, as the UI assembles them."""
    rows = [dict(r) for r in bc.MODEL_GRID_ROWS[model]]
    if model == "licensing":
        for r in rows:
            if r["name"] == "rate_or_fee":
                r["label"] = ("Annual flat licence fee ({cur})"
                              if license_model_type == "flat"
                              else "Royalty per licensed unit ({cur})")
    rows.append(dict(bc.GRID_CAPEX_ROW))
    rows.append({"name": "opex__0__opex", "label": bc.DEFAULT_OPEX_LABEL, "kind": "opex"})
    return rows


def _blank_grid(row_specs, n_years):
    values = {r["name"]: [None] * (n_years + 1) for r in row_specs}
    states = {r["name"]: ["blank"] * (n_years + 1) for r in row_specs}
    return values, states


def _mark_computed(states, n_years):
    for y in range(1, n_years + 1):
        states["beginning_base"][y] = "computed"
    for y in range(n_years + 1):
        states["ending_base"][y] = "computed"


def _editable_cells(row_specs, states, n_years):
    """Every (row, year) a user or the assumption pass is allowed to fill."""
    return [
        (r["name"], y)
        for r in row_specs
        for y in range(n_years + 1)
        if states[r["name"]][y] != "computed"
    ]


def run_case(case, groq_client):
    model = case["expected_model"]
    license_type = case.get("license_model_type")
    print(f"\n[{case['name']}]  expecting model = {model!r}")
    log = CheckLog()

    extraction = case["extraction"]
    n_years = bc.default_grid_years(extraction)
    row_specs = _row_specs(model, license_type)

    # --- CHECK 1: classification ---
    try:
        cls = bc.classify_business_model(extraction, case["risks"], groq_client)
        got = cls["business_model"]
        log.check("classify_business_model", got == model,
                  f"got {got!r}, confidence {cls['confidence']!r}")
    except bc.BusinessCaseError as e:
        log.check("classify_business_model", False, f"error: {e}")

    values, states = _blank_grid(row_specs, n_years)

    # --- CHECK 2: deterministic pre-fill from the term sheet ---
    prefill = bc.prefill_grid_cells(model, extraction, row_specs, n_years)
    for row, cells in prefill.items():
        for year, info in cells.items():
            values[row][year] = info["value"]
            states[row][year] = "prefilled"
    expect_prefill = extraction.get("minimum_volume") is not None
    got_prefill = bool(prefill)
    log.check(
        "prefill_grid_cells",
        got_prefill == expect_prefill,
        (f"{sum(len(c) for c in prefill.values())} cell(s) from minimum_volume"
         if got_prefill else "nothing to pre-fill (no minimum_volume)"),
    )

    # the user's own authoritative figures
    for row, spec in case["user_cells"].items():
        cells = spec if isinstance(spec, dict) else {y: spec for y in range(n_years + 1)}
        for year, val in cells.items():
            values[row][year] = float(val)
            states[row][year] = "user"

    if model == "subscription":
        bc.recompute_subscription_waterfall(values, n_years)
        _mark_computed(states, n_years)

    # snapshot everything already known, to prove the assumption pass leaves it alone
    known = {(r, y): values[r][y]
             for r in values for y in range(n_years + 1)
             if states[r][y] in ("prefilled", "user")}

    # --- CHECK 3: assumptions fill ONLY blanks, never overwrite ---
    proposals = bc.propose_grid_assumptions(
        model, extraction, row_specs, values, states, n_years, groq_client
    )
    targeted_known = [
        f"{row}[Y{year}]" for row, cells in proposals.items() for year in cells
        if (row, year) in known
    ]
    log.check(
        "assumptions target only blanks",
        not targeted_known,
        f"{sum(len(c) for c in proposals.values())} proposed, {len(known)} known cells untouched"
        if not targeted_known else f"would overwrite: {targeted_known}",
    )

    for row, cells in proposals.items():
        for year, info in cells.items():
            values[row][year] = info["value"]
            states[row][year] = "assumed"
    if model == "subscription":
        bc.recompute_subscription_waterfall(values, n_years)
        _mark_computed(states, n_years)

    # --- CHECK 4: known values survived byte-for-byte ---
    clobbered = [f"{r}[Y{y}] {v} -> {values[r][y]}"
                 for (r, y), v in known.items() if values[r][y] != v]
    log.check("no silent overwrites", not clobbered,
              f"{len(known)} pre-filled/user cells intact" if not clobbered
              else f"changed: {clobbered}")

    # --- CHECK 5: subscription waterfall arithmetic ---
    if model == "subscription":
        beg, new, churn, end = (values["beginning_base"], values["new_added"],
                                values["churned_out"], values["ending_base"])
        bad = []
        for y in range(n_years + 1):
            expected_end = (beg[y] or 0) + (new[y] or 0) - (churn[y] or 0)
            if abs((end[y] or 0) - expected_end) > 1e-6:
                bad.append(f"Y{y}: {end[y]} != {expected_end}")
            if y > 0 and abs((beg[y] or 0) - (end[y - 1] or 0)) > 1e-6:
                bad.append(f"Y{y}: beginning {beg[y]} != prior ending {end[y-1]}")
        log.check("waterfall rolls forward", not bad,
                  f"ending base {[round(v) for v in end]}" if not bad else "; ".join(bad))

    # --- CHECK 6: nothing reaching financial_engine is None ---
    # build_pl_from_grid's _grid_row turns a None into 0.0 without complaint, so a gap
    # here becomes a quietly-wrong P&L rather than an error.
    nones = [f"{r}[Y{y}]" for r, y in _editable_cells(row_specs, states, n_years)
             if values[r][y] is None]
    log.check("no None reaches financial_engine", not nones,
              f"{len(_editable_cells(row_specs, states, n_years))} cells all populated"
              if not nones else f"still None: {nones}")

    # --- CHECK 7: P&L + NPV/ROI ---
    if nones:
        log.check("build_pl_from_grid + compute_npv_roi", False, "skipped — grid incomplete")
        log.check("revenue varies across years", False, "skipped — grid incomplete")
    else:
        try:
            pl = bc.build_pl_from_grid(model, row_specs, values, n_years,
                                       license_model_type=license_type)
            nr = bc.compute_npv_roi(pl, discount_rate=0.10)

            rev, cap, opx = pl["revenue_by_year"], pl["capex_by_year"], pl["opex_by_year"]
            identities_ok = (
                len(pl["cash_flows"]) == n_years + 1
                and all(abs(pl["cash_flows"][y] - (rev[y] - opx[y] - cap[y])) < 1e-6
                        for y in range(n_years + 1))
                and abs(nr["investment_base"] - (pl["total_capex"] + pl["total_opex"])) < 1e-6
                and len(nr["npv"]["discounted_by_year"]) == n_years + 1
            )
            roi_txt = f"{nr['roi']['roi_pct']:,.1f}%" if nr["roi"] else "n/a (zero investment)"
            log.check("build_pl_from_grid + compute_npv_roi", identities_ok,
                      f"NPV {nr['npv']['npv']:,.0f}, ROI {roi_txt}")

            # --- CHECK 8: variation where the deal's economics imply it ---
            operating = [round(v, 6) for v in rev[1:]] or [0.0]
            varies = len(set(operating)) > 1
            expected = case["expect_revenue_variation"]
            log.check(
                "revenue varies across years", varies == expected,
                f"{'varies' if varies else 'flat'} as expected: "
                f"{[round(v) for v in rev]}"
                if varies == expected else
                f"expected {'variation' if expected else 'flat'}, got {[round(v) for v in rev]}",
            )
        except (bc.BusinessCaseError, FinancialEngineError) as e:
            log.check("build_pl_from_grid + compute_npv_roi", False, f"error: {e}")
            log.check("revenue varies across years", False, "skipped — P&L failed")

    log.dump()
    verdict = "CASE PASS" if log.ok else "CASE FAIL"
    print(f"  => {verdict}")
    return log.ok


def main():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set (needed for classify + propose). Aborting.")
        return 2

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print(f"Running {len(CASES)} business-case evals against business_case.py\n" + "=" * 68)
    results = [run_case(case, groq_client) for case in CASES]

    passed = sum(results)
    print("\n" + "=" * 68)
    print(f"SUMMARY: {passed}/{len(CASES)} cases passed")
    for case, ok in zip(CASES, results):
        print(f"  {'PASS' if ok else 'FAIL'}  {case['name']}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
