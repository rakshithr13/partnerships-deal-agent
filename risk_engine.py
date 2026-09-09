"""
Deterministic risk engine for extracted term sheet data.

Pipeline: extracted JSON (from termsheet_extractor) -> deterministic Python
rules flag triggers -> Groq writes a contract-specific explanation for each ->
risk register (JSON list).

The *detection* is 100% deterministic Python. Groq is only used to turn an
already-fired trigger into human-readable prose that cites this contract's
actual language — never to decide whether something is a risk.

Usage:
    python risk_engine.py test_agreement_extracted.json

Or as a library:
    from risk_engine import build_risk_register
    risks = build_risk_register(extracted_data)
"""

import json
import os
import re
import sys

from groq import Groq

from termsheet_extractor import DEFAULT_MODEL, REQUIRED_FIELDS


class RiskEngineError(Exception):
    """Raised when the risk register can't be built."""


# How bad it is to go into a deal with a given field unknown (rule 4).
_MISSING_FIELD_SEVERITY = {
    "partner_name": "medium",
    "contract_duration_months": "medium",
    "revenue_share": "medium",
    "revenue_share_details": "low",
    "minimum_volume": "low",
    "exclusivity": "high",
    "payment_terms": "medium",
    "termination_terms": "high",
    "renewal_terms": "medium",
    "IP_Ownership": "high",
}

_NON_EXCLUSIVE_VALUES = {"non-exclusive", "nonexclusive", "non exclusive"}

_TERMINATION_NOTICE_THRESHOLD_DAYS = 90

_UNIT_TO_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}

# "90 days' notice", "90-day written notice", "90 days prior written notice"
_NOTICE_BEFORE_RE = re.compile(
    r"(\d+)\s*[-\s]?\s*(day|days|week|weeks|month|months|year|years)\b"
    r"[^.;,]{0,40}?\bnotice",
    re.IGNORECASE,
)
# "notice period of 90 days", "on notice of not less than 90 days"
_NOTICE_AFTER_RE = re.compile(
    r"\bnotice\b[^.;,]{0,40}?(\d+)\s*[-\s]?\s*"
    r"(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)

# Language that would signal a consequence/relief mechanism attached to a
# minimum-volume commitment (rule 3). Deliberately volume-specific so an
# unrelated "cure period" in the termination clause doesn't suppress the flag.
_PENALTY_KEYWORDS = (
    "penalt", "shortfall", "short fall", "liquidated damages",
    "make-up", "make up", "make-good", "make good", "makegood",
    "true-up", "true up", "take-or-pay", "take or pay",
    "waiv", "buy-out", "buyout", "deficiency", "compensat",
    "minimum fee", "minimum payment", "minimum purchase fee", "underperformance",
)

_CONVENIENCE_KEYWORDS = (
    "for convenience", "without cause", "without reason",
    "at will", "for any reason", "for no reason",
)


def _normalize(text: str) -> str:
    """Fold the unicode dashes/quotes the extractor sometimes preserves down to ASCII."""
    return (
        text.replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("’", "'")
        .replace("‘", "'")
    )


def _clauses(text: str):
    """Split a terms string into rough clauses for locality-aware matching."""
    return [c.strip() for c in re.split(r"[.;]", text) if c.strip()]


def _notice_days_in(text: str):
    """Smallest notice period (in days) mentioned in `text`, or None if none found."""
    found = []
    for rx in (_NOTICE_BEFORE_RE, _NOTICE_AFTER_RE):
        for m in rx.finditer(text):
            found.append(int(m.group(1)) * _UNIT_TO_DAYS[m.group(2).lower()])
    return min(found) if found else None


def _string_fields(data: dict):
    return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}


def detect_triggers(data: dict):
    """
    Apply the deterministic rules to an extracted term sheet dict.

    Returns a list of trigger dicts, each carrying enough context
    (`evidence`, `explain_focus`) for the explanation step.
    """
    triggers = []
    haystack = _normalize(" ".join(_string_fields(data).values())).lower()

    # Rule 1: exclusivity present.
    excl = data.get("exclusivity")
    if isinstance(excl, str) and excl.strip().lower() not in _NON_EXCLUSIVE_VALUES:
        norm = _normalize(excl)
        fully_exclusive = "non-exclusive" not in norm.lower()
        triggers.append({
            "rule": "exclusivity_present",
            "risk_name": "Exclusivity commitment",
            "severity": "high" if fully_exclusive else "medium",
            "source_field": "exclusivity",
            "evidence": norm,
            "explain_focus": (
                "The agreement grants exclusivity. Explain the commercial and legal "
                "downside given this specific exclusivity's scope, territory, and any carve-outs."
            ),
        })

    # Rule 2: termination-for-convenience notice period under 90 days.
    term = data.get("termination_terms")
    if isinstance(term, str) and term.strip():
        norm = _normalize(term)
        conv_clauses = [
            c for c in _clauses(norm)
            if any(k in c.lower() for k in _CONVENIENCE_KEYWORDS)
        ]
        if conv_clauses:
            clause_text = "; ".join(conv_clauses)
            days = _notice_days_in(clause_text)
            if days is None:
                days = _notice_days_in(norm)  # fall back to the whole termination text
            if days is None:
                triggers.append({
                    "rule": "termination_for_convenience_notice_unspecified",
                    "risk_name": "Termination-for-convenience notice period not specified",
                    "severity": "medium",
                    "source_field": "termination_terms",
                    "evidence": clause_text,
                    "explain_focus": (
                        "The contract permits termination for convenience but the extracted "
                        "language states no notice period. Explain why an unspecified / "
                        "potentially very short notice period is a risk here."
                    ),
                })
            elif days < _TERMINATION_NOTICE_THRESHOLD_DAYS:
                triggers.append({
                    "rule": "short_termination_for_convenience_notice",
                    "risk_name": "Short termination-for-convenience notice period",
                    "severity": "high",
                    "source_field": "termination_terms",
                    "evidence": clause_text,
                    "explain_focus": (
                        f"A party may terminate for convenience on roughly {days} days' notice, "
                        f"below the {_TERMINATION_NOTICE_THRESHOLD_DAYS}-day threshold. Explain "
                        "the exposure this short runway creates given this deal's other terms."
                    ),
                })

    # Rule 3: minimum volume commitment present with no penalty/waiver clause mentioned.
    mv = data.get("minimum_volume")
    if isinstance(mv, str) and mv.strip():
        if not any(kw in haystack for kw in _PENALTY_KEYWORDS):
            triggers.append({
                "rule": "minimum_volume_without_penalty_terms",
                "risk_name": "Minimum volume commitment without stated penalty/waiver terms",
                "severity": "medium",
                "source_field": "minimum_volume",
                "evidence": _normalize(mv),
                "explain_focus": (
                    "There is a minimum volume / spend commitment, but nothing in the extracted "
                    "term sheet describes a shortfall penalty, make-good, true-up, or waiver "
                    "mechanism. Explain the two-sided risk of that silence for this commitment."
                ),
            })

    # Rule 4: any null / missing field from the extraction.
    for field in REQUIRED_FIELDS:
        if data.get(field) is None:
            triggers.append({
                "rule": "missing_extracted_field",
                "risk_name": f"Missing contract term: {field}",
                "severity": _MISSING_FIELD_SEVERITY.get(field, "low"),
                "source_field": field,
                "evidence": None,
                "explain_focus": (
                    f"The extractor found no value for '{field}' (it is null / absent). "
                    "Explain the risk of negotiating or analyzing this partnership with that "
                    "term unknown, given what the rest of the term sheet does say."
                ),
            })

    return triggers


_EXPLAIN_PROMPT = """You are a contracts risk analyst. A deterministic rule has flagged a potential risk in a partnership term sheet.

Full extracted term sheet (JSON):
{data_json}

Flagged risk:
- rule: {rule}
- risk_name: {risk_name}
- source field: {source_field}
- relevant contract language: {evidence}
- what to explain: {explain_focus}

Write an analysis grounded in THIS contract's actual language, numbers, parties, and territory. Quote or cite the specific terms. Do NOT produce generic boilerplate that could apply to any contract. If the relevant language is null/absent, reason concretely about the consequence of that specific gap.

Return ONLY a JSON object with exactly these keys:
- "reason": string - why this specific clause (or gap) is a risk, citing the actual terms.
- "potential_impact": string - the concrete commercial, legal, or operational consequence if this risk materializes for this deal.
- "recommended_action": string - a specific negotiation ask or diligence step to mitigate it.

No markdown, no code fences, no extra keys, no commentary.
"""


def _explain(trigger: dict, data: dict, groq_client: Groq, model: str) -> dict:
    """Ask Groq for a contract-specific explanation of one fired trigger."""
    evidence = trigger.get("evidence")
    prompt = _EXPLAIN_PROMPT.format(
        data_json=json.dumps(data, indent=2),
        rule=trigger["rule"],
        risk_name=trigger["risk_name"],
        source_field=trigger["source_field"],
        evidence=evidence if evidence else "(field is null / not present in the term sheet)",
        explain_focus=trigger["explain_focus"],
    )

    required = ("reason", "potential_impact", "recommended_action")
    for _attempt in range(2):
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and all(parsed.get(k) for k in required):
            return {k: str(parsed[k]).strip() for k in required}

    raise RiskEngineError(
        f"Groq did not return a usable explanation for rule '{trigger['rule']}'"
    )


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def build_risk_register(
    data: dict,
    groq_client: Groq = None,
    model: str = DEFAULT_MODEL,
    strict: bool = False,
) -> list:
    """
    Build the risk register from an extracted term sheet dict.

    Returns a list of risk dicts, each with keys: risk_name, severity, reason,
    source_field, potential_impact, recommended_action. Sorted high -> low severity.

    If `strict` is True, a failed Groq explanation raises RiskEngineError.
    Otherwise that one risk gets a clearly-marked fallback explanation and the
    rest of the register is still produced.
    """
    if not isinstance(data, dict):
        raise RiskEngineError(f"Expected the extracted term sheet as a dict, got {type(data).__name__}")

    if groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RiskEngineError("GROQ_API_KEY is not set and no groq_client was provided")
        groq_client = Groq(api_key=api_key)

    register = []
    for trigger in detect_triggers(data):
        try:
            explanation = _explain(trigger, data, groq_client, model)
        except RiskEngineError:
            if strict:
                raise
            explanation = {
                "reason": (
                    f"Deterministic rule '{trigger['rule']}' fired on field "
                    f"'{trigger['source_field']}'. Relevant language: "
                    f"{trigger.get('evidence') or '(field is null / absent)'}. "
                    "Automated explanation was unavailable."
                ),
                "potential_impact": "Not assessed - LLM explanation unavailable.",
                "recommended_action": "Have a contracts reviewer assess this flag manually.",
            }

        register.append({
            "risk_name": trigger["risk_name"],
            "severity": trigger["severity"],
            "reason": explanation["reason"],
            "source_field": trigger["source_field"],
            "potential_impact": explanation["potential_impact"],
            "recommended_action": explanation["recommended_action"],
        })

    register.sort(key=lambda r: _SEVERITY_ORDER.get(r["severity"], 3))
    return register


def _write_json(payload: str, out_path: str) -> None:
    """Write JSON as UTF-8 (no BOM). Use this instead of shell `>` redirection,
    which on PowerShell produces UTF-16-with-BOM that won't round-trip as JSON."""
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
        if not payload.endswith("\n"):
            fh.write("\n")


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Build a risk register from an extracted term sheet JSON"
    )
    parser.add_argument("json_path", help="Path to the extracted term sheet JSON")
    parser.add_argument(
        "-o", "--out",
        help="Path to write the risk register JSON to (UTF-8). "
             "Default: <name>_risks.json next to the input file.",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print JSON to stdout only; do not write a file.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail if any Groq explanation can't be generated (default: fall back).",
    )
    args = parser.parse_args()

    try:
        with open(args.json_path, "r", encoding="utf-8-sig") as fh:
            extracted = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read extracted JSON: {e}")
        sys.exit(1)

    try:
        risks = build_risk_register(extracted, strict=args.strict)
    except RiskEngineError as e:
        print(f"Risk engine failed: {e}")
        sys.exit(1)

    payload = json.dumps(risks, indent=2)

    if args.stdout:
        print(payload)
    else:
        base = os.path.splitext(args.json_path)[0]
        if base.endswith("_extracted"):
            base = base[: -len("_extracted")]
        out_path = args.out or f"{base}_risks.json"
        _write_json(payload, out_path)
        print(f"Wrote {out_path}  ({len(risks)} risk(s))")
