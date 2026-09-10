"""
Eval harness for business_case.py — exercised directly, NOT through Streamlit.

Runs the three Groq-and-engine functions end to end for five hand-built deals:

    classify_business_model  ->  must return the expected model
    propose_assumptions      ->  every generated value must carry source: "groq" | "fallback"
    (assemble inputs)        ->  no required financial_engine field may be silently None
    run_scenarios            ->  low / medium / high headline values must NOT all be equal

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


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #
# Each case: a term-sheet extraction + risk register, the expected classification,
# and `user_inputs` = the real figures a user would confirm (the ones that can be
# neither pre-filled from the term sheet nor responsibly assumed). The harness then
# lets prefill + propose_assumptions fill the rest, exactly like the UI flow.

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
        # price per active customer per YEAR (12.99 * 12); can't be assumed from norms.
        "user_inputs": {"price": 155.88},
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
        # raw traffic volumes can't be assumed; CPM/CPC are left for propose_assumptions.
        "user_inputs": {"annual_impressions": 800_000_000, "annual_clicks": 4_000_000},
    },
    {
        "name": "fee_based_telematics",
        "expected_model": "fee_based",
        "extraction": _extraction(
            partner_name="Orbit Telematics",
            contract_duration_months=None,  # deliberately absent -> propose_assumptions fills duration
            revenue_share="Provider receives a flat fee of $3.50 per connected vehicle per year",
            revenue_share_details=(
                "A single flat per-vehicle fee. No revenue share, no CPM/CPC, no subscriber "
                "activation/churn mechanic, no software licence."
            ),
            minimum_volume="900,000 connected vehicles per year",
            exclusivity="Exclusive within North America",
            payment_terms="Annual fee invoiced quarterly",
        ),
        "risks": [
            {"risk_name": "Exclusivity commitment", "severity": "high"},
            {"risk_name": "Missing contract term: contract_duration_months", "severity": "medium"},
        ],
        # per-unit fee is negotiated, not assumable.
        "user_inputs": {"rate_per_unit": 3.50},
    },
    {
        "name": "licensing_per_unit_autonomy_stack",
        "expected_model": "licensing",
        "extraction": _extraction(
            partner_name="Vanta Silicon",
            contract_duration_months=None,   # -> propose_assumptions fills duration
            revenue_share=(
                "OEM pays Provider a per-unit software licence royalty of $60 for each vehicle "
                "equipped with the Provider autonomy stack."
            ),
            revenue_share_details="Royalty per equipped vehicle. Provider retains all platform IP.",
            minimum_volume=None,             # -> volume must come from user_inputs
            exclusivity=None,
            payment_terms="Royalty reported and paid quarterly",
            IP_Ownership="Provider retains all IP; OEM receives a non-transferable licence.",
        ),
        "risks": [
            {"risk_name": "Missing contract term: exclusivity", "severity": "high"},
        ],
        "user_inputs": {
            "license_model_type": "per_unit",
            "per_unit_fee": 60.0,
            "volume": 1_200_000,  # total equipped vehicles over the term
        },
    },
    {
        "name": "licensing_flat_platform_fee",
        "expected_model": "licensing",
        "extraction": _extraction(
            partner_name="Helix Platform",
            contract_duration_months=36,
            revenue_share=(
                "OEM pays Provider a fixed annual platform licence fee of $4,000,000, invoiced "
                "quarterly. There is no per-unit or revenue-share component."
            ),
            revenue_share_details="Flat annual fee, independent of volume.",
            minimum_volume=None,
            exclusivity=None,
            payment_terms="Quarterly instalments",
            IP_Ownership="Provider owns platform IP; OEM granted a non-transferable licence.",
        ),
        "risks": [
            {"risk_name": "Missing contract term: termination_terms", "severity": "high"},
        ],
        "user_inputs": {
            "license_model_type": "flat",
            "flat_fee": 4_000_000.0,
        },
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


def _assemble_inputs(model, extraction, user_inputs, groq_client, log):
    """Reproduce the UI flow: prefill -> user figures -> propose -> apply -> validate."""
    inputs = {}

    # 1. deterministic pre-fill from the term sheet
    prefill = bc.prefill_inputs_from_extraction(model, extraction)
    for field, info in prefill.items():
        inputs[field] = info["value"]

    # 2. the real figures the user confirms (authoritative)
    inputs.update(user_inputs)

    # 3. propose assumptions for whatever is still missing
    missing = bc.missing_fields(model, inputs)
    proposed = bc.propose_assumptions(model, extraction, missing, groq_client)

    # --- CHECK: every proposed value is labelled groq | fallback, never bare ---
    unlabelled = [
        f for f, info in proposed.items()
        if not isinstance(info, dict) or info.get("source") not in ("groq", "fallback")
    ]
    if proposed:
        tag_summary = ", ".join(
            f"{f}[{proposed[f]['source']}"
            + ("/none]" if proposed[f]["value"] is None else "]")
            for f in proposed
        )
    else:
        tag_summary = "none proposed"
    log.check(
        "propose_assumptions labelling",
        not unlabelled,
        f"{len(proposed)} proposed: {tag_summary}" if not unlabelled
        else f"unlabelled: {unlabelled}",
    )

    # 4. apply proposed values (only where we don't already have one)
    for field, info in proposed.items():
        if info["value"] is not None and inputs.get(field) is None:
            inputs[field] = info["value"]

    # --- CHECK: no required financial_engine field is silently None ---
    required = bc.required_fields(model, inputs)
    none_fields = [f for f in required if inputs.get(f) is None
                   or (f == "tier_breaks" and not inputs.get(f))]
    log.check(
        "no required field is None",
        not none_fields,
        f"{len(required)} required: {', '.join(required)}" if not none_fields
        else f"still None: {none_fields}",
    )

    return inputs, none_fields


def run_case(case, groq_client):
    print(f"\n[{case['name']}]  expecting model = {case['expected_model']!r}")
    log = CheckLog()
    model_for_flow = case["expected_model"]

    # --- CHECK 1: classification ---
    try:
        cls = bc.classify_business_model(case["extraction"], case["risks"], groq_client)
        got = cls["business_model"]
        log.check(
            "classify_business_model",
            got == case["expected_model"],
            f"got {got!r}, confidence {cls['confidence']!r}",
        )
    except bc.BusinessCaseError as e:
        log.check("classify_business_model", False, f"error: {e}")

    # --- CHECK 2 + 3: assemble inputs (prefill / propose labelling / no None) ---
    inputs, none_fields = _assemble_inputs(
        model_for_flow, case["extraction"], case["user_inputs"], groq_client, log
    )

    # --- CHECK 4: three distinct scenario headlines ---
    if none_fields:
        log.check("run_scenarios headlines not all equal", False,
                  "skipped — inputs incomplete")
    else:
        try:
            out = bc.run_scenarios(model_for_flow, inputs)
            heads = {name: out["scenarios"][name]["headline"] for name in bc.SCENARIO_NAMES}
            distinct = len(set(round(v, 6) for v in heads.values())) >= 2
            log.check(
                "run_scenarios headlines not all equal",
                distinct,
                "low={low:,.0f}  med={medium:,.0f}  high={high:,.0f}".format(**heads),
            )
        except bc.BusinessCaseError as e:
            log.check("run_scenarios headlines not all equal", False, f"error: {e}")

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
