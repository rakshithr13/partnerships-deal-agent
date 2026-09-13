"""Pydantic models for every request and response.

Field names and types mirror the backend's actual return shapes exactly — see
``termsheet_extractor.REQUIRED_FIELDS``, ``risk_engine.build_risk_register``,
``business_case.classify_business_model`` / ``build_pl_from_grid`` /
``compute_npv_roi``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BusinessModelLiteral = Literal["subscription", "adtech", "fee_based", "licensing"]
LicenseTypeLiteral = Literal["flat", "per_unit", "tiered"]
SeverityLiteral = Literal["high", "medium", "low"]
KindLiteral = Literal["volume", "churn", "rate", "capex", "opex", "seed", "computed"]
StateLiteral = Literal["prefilled", "user", "assumed", "computed", "blank"]

YearSeries = list[float | None]
YearIndex = Annotated[int, Field(ge=0)]


class ApiModel(BaseModel):
    # extra="forbid" so a client sending e.g. "ip_ownership" gets a 422 rather than a
    # silently-null field and a quietly-wrong risk register.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------- #
# Shared domain models
# --------------------------------------------------------------------------- #

class ExtractionModel(ApiModel):
    """The 10 REQUIRED_FIELDS. Every one is nullable — the extractor returns None
    for anything the document doesn't address."""

    partner_name: str | None = None
    contract_duration_months: int | None = Field(default=None, ge=0)
    revenue_share: str | None = None
    revenue_share_details: str | None = None
    minimum_volume: str | None = None
    exclusivity: str | None = None
    payment_terms: str | None = None
    termination_terms: str | None = None
    renewal_terms: str | None = None
    IP_Ownership: str | None = None  # noqa: N815 - matches the extractor's key exactly


class RiskModel(ApiModel):
    risk_name: str
    severity: SeverityLiteral
    reason: str
    source_field: str
    potential_impact: str
    recommended_action: str


class RiskInput(ApiModel):
    """Request-side risk. ``classify_business_model`` reads only ``risk_name``, so
    everything else is optional and a full ``RiskModel`` validates unchanged."""

    risk_name: str
    severity: SeverityLiteral | None = None
    reason: str | None = None
    source_field: str | None = None
    potential_impact: str | None = None
    recommended_action: str | None = None


class TriggerModel(ApiModel):
    rule: str
    risk_name: str
    severity: SeverityLiteral
    source_field: str
    evidence: str | None
    explain_focus: str


class RowSpecModel(ApiModel):
    name: str = Field(min_length=1)
    label: str
    kind: KindLiteral
    help: str | None = None
    prefill: Literal["annual_volume"] | None = None


class OpexItem(ApiModel):
    key: str | None = Field(default=None, pattern=r"^opex__\d+__[a-z0-9_]+$")
    label: str = Field(min_length=1)


class TierBreak(ApiModel):
    min_units: float = Field(ge=0)
    rate_per_unit: float = Field(ge=0)


class CellRef(ApiModel):
    row: str
    year: int


class PrefillCell(ApiModel):
    value: float
    source: str


class AssumptionCell(ApiModel):
    value: float
    rationale: str


# prefill_grid_cells / propose_grid_assumptions key the inner dict by INT year.
# JSON object keys are strings, so these serialize as {"1": {...}} and validate
# back from "1" -> 1 on the way in.
PrefillMap = dict[str, dict[YearIndex, PrefillCell]]
AssumptionMap = dict[str, dict[YearIndex, AssumptionCell]]


def validate_grid_shape(
    values: dict[str, list[Any]],
    states: dict[str, list[Any]] | None,
    n_years: int,
) -> None:
    expected = n_years + 1
    for row, series in values.items():
        if len(series) != expected:
            raise ValueError(
                f"values[{row!r}] has {len(series)} entries; expected {expected} "
                f"(Year 0 through Year {n_years})"
            )
    if states is None:
        return
    for row, series in states.items():
        if len(series) != expected:
            raise ValueError(
                f"states[{row!r}] has {len(series)} entries; expected {expected}"
            )
    unknown = set(states) - set(values)
    if unknown:
        raise ValueError(f"states has rows absent from values: {sorted(unknown)}")


# --------------------------------------------------------------------------- #
# /v1/extract
# --------------------------------------------------------------------------- #

class ExtractResponse(ApiModel):
    extraction: ExtractionModel
    filename: str
    size_bytes: int
    llm_model: str


# --------------------------------------------------------------------------- #
# /v1/risks
# --------------------------------------------------------------------------- #

class TriggersRequest(ApiModel):
    extraction: ExtractionModel


class TriggersResponse(ApiModel):
    triggers: list[TriggerModel]
    count: int


class AssessRiskRequest(ApiModel):
    extraction: ExtractionModel
    strict: bool = False
    llm_model: str | None = None


class AssessRiskResponse(ApiModel):
    risks: list[RiskModel]
    count: int
    severity_counts: dict[str, int]
    llm_model: str


# --------------------------------------------------------------------------- #
# /v1/business-model
# --------------------------------------------------------------------------- #

class ClassifyRequest(ApiModel):
    extraction: ExtractionModel
    risks: list[RiskInput] = Field(default_factory=list)
    llm_model: str | None = None


class ClassificationModel(ApiModel):
    business_model: BusinessModelLiteral
    reasoning: str
    signals: list[str]
    # Plain str, not a Literal: the backend lowercases whatever the model returned and
    # only substitutes "unknown" when it's falsy, so "fairly high" is reachable.
    confidence: str
    currency: str | None
    cross_border: bool
    counterparty_currency: str | None
    currency_reasoning: str


class ClassifyResponse(ApiModel):
    classification: ClassificationModel
    llm_model: str


class SignalsRequest(ApiModel):
    business_model: BusinessModelLiteral
    extraction: ExtractionModel


class SignalsModel(ApiModel):
    business_model: BusinessModelLiteral
    supported: bool
    matched: list[str]
    hint: str


# --------------------------------------------------------------------------- #
# /v1/grid
# --------------------------------------------------------------------------- #

class GridTemplateRequest(ApiModel):
    business_model: BusinessModelLiteral
    extraction: ExtractionModel
    n_years: int | None = Field(default=None, ge=1, le=40)
    currency: str = "USD"
    license_model_type: LicenseTypeLiteral | None = None
    opex_items: list[OpexItem] | None = None
    row_label_overrides: dict[str, str] = Field(default_factory=dict)
    include_prefill: bool = True


class GridTemplateResponse(ApiModel):
    business_model: BusinessModelLiteral
    license_model_type: LicenseTypeLiteral | None
    n_years: int
    years: list[str]
    currency: str
    currency_symbol: str
    row_specs: list[RowSpecModel]
    values: dict[str, YearSeries]
    states: dict[str, list[StateLiteral]]
    prefill: PrefillMap
    blank_cells: list[CellRef]


class GridPrefillRequest(ApiModel):
    business_model: BusinessModelLiteral
    extraction: ExtractionModel
    row_specs: list[RowSpecModel]
    n_years: int = Field(ge=1, le=40)


class GridPrefillResponse(ApiModel):
    prefill: PrefillMap


class GridRecomputeRequest(ApiModel):
    n_years: int = Field(ge=1, le=40)
    values: dict[str, YearSeries]
    states: dict[str, list[StateLiteral]] | None = None

    @model_validator(mode="after")
    def _shape(self):
        validate_grid_shape(self.values, self.states, self.n_years)
        return self


class GridRecomputeResponse(ApiModel):
    values: dict[str, YearSeries]
    states: dict[str, list[StateLiteral]]
    computed_rows: list[str]


class GridAssumptionsRequest(ApiModel):
    business_model: BusinessModelLiteral
    extraction: ExtractionModel
    row_specs: list[RowSpecModel]
    n_years: int = Field(ge=1, le=40)
    values: dict[str, YearSeries]
    states: dict[str, list[StateLiteral]]
    apply: bool = True
    llm_model: str | None = None

    @model_validator(mode="after")
    def _shape(self):
        validate_grid_shape(self.values, self.states, self.n_years)
        return self


class GridAssumptionsResponse(ApiModel):
    assumptions: AssumptionMap
    values: dict[str, YearSeries]
    states: dict[str, list[StateLiteral]]
    assumption_notes: dict[str, str]
    blank_cells: list[CellRef]
    llm_model: str


# --------------------------------------------------------------------------- #
# /v1/business-case
# --------------------------------------------------------------------------- #

class BusinessCaseRequest(ApiModel):
    business_model: BusinessModelLiteral
    row_specs: list[RowSpecModel]
    n_years: int = Field(ge=1, le=40)
    values: dict[str, YearSeries]
    license_model_type: LicenseTypeLiteral | None = None
    tier_breaks: list[TierBreak] | None = None
    discount_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    auto_recompute_waterfall: bool = True

    @model_validator(mode="after")
    def _check(self):
        validate_grid_shape(self.values, None, self.n_years)

        if self.business_model != "licensing":
            if self.license_model_type is not None or self.tier_breaks is not None:
                raise ValueError(
                    "license_model_type and tier_breaks apply only to the "
                    "'licensing' business model"
                )
            return self

        if self.license_model_type == "tiered":
            if not self.tier_breaks:
                raise ValueError("tier_breaks is required when license_model_type='tiered'")
            ordered = sorted(self.tier_breaks, key=lambda t: t.min_units)
            if ordered[0].min_units != 0:
                raise ValueError("tier_breaks must include a tier starting at min_units=0")
            mins = [t.min_units for t in ordered]
            if len(set(mins)) != len(mins):
                raise ValueError("tier_breaks must have distinct min_units")
        return self


class PLTable(ApiModel):
    revenue: list[float]
    capex: list[float]
    opex: list[float]
    gross_profit: list[float]
    net_profit: list[float]
    cumulative_cash_flow: list[float]


class NpvModel(ApiModel):
    npv: float
    discount_rate: float
    discounted_by_year: list[float]


class RoiModel(ApiModel):
    roi_pct: float
    net_profit: float
    total_investment: float


class BusinessCaseResponse(ApiModel):
    business_model: BusinessModelLiteral
    license_model_type: LicenseTypeLiteral | None
    projection_years: int
    years: list[str]
    table: PLTable
    cash_flows: list[float]
    total_capex: float
    total_opex: float
    final_cumulative: float
    npv: NpvModel
    roi: RoiModel | None
    discount_rate: float
    investment_base: float
    notes: list[str]
    warnings: list[str]


# --------------------------------------------------------------------------- #
# meta / health
# --------------------------------------------------------------------------- #

class HealthResponse(ApiModel):
    status: str
    groq_configured: bool


class MetaResponse(ApiModel):
    business_models: list[str]
    model_labels: dict[str, str]
    currency_choices: list[str]
    currency_symbols: dict[str, str]
    license_model_types: list[str]
    cell_states: list[str]
    row_kinds: list[str]
    grid_rows: dict[str, list[RowSpecModel]]
    default_opex_label: str
    default_llm_model: str
    max_grid_years: int
