import streamlit as st
import chromadb
from tavily import TavilyClient
from groq import Groq
from pypdf import PdfReader
import docx
import io
import json
import os
import re
import copy
from dotenv import load_dotenv

import pandas as pd
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode, JsCode

import business_case as bc
from financial_engine import generate_pl_chart, fetch_fx_rate, FinancialEngineError
from termsheet_extractor import extract_termsheet, ExtractionError
from risk_engine import build_risk_register, RiskEngineError

# Load variables from .env into the environment BEFORE reading them
load_dotenv()

# Initialize clients (put your keys here or in environment variables)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") #Using API keys from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY") #Using API keys from .env

# Fail early with a clear message instead of a confusing SDK error later
missing = [
    name
    for name, value in (("TAVILY_API_KEY", TAVILY_API_KEY), ("GROQ_API_KEY", GROQ_API_KEY))
    if not value
]
if missing:
    st.error(f"Missing required environment variable(s): {', '.join(missing)}")
    st.stop()

tavily = TavilyClient(api_key=TAVILY_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)



# Local Vector Database Setup for MSAs
chroma_client = chromadb.PersistentClient(path="./msa_vector_db")
collection = chroma_client.get_or_create_collection(name="partner_msas")

st.set_page_config(page_title="Partnership Intelligence & MSA Portfolio Agent", layout="wide")
st.title("Unified Partnership Agent")

tab1, tab2, tab3 = st.tabs([
    "🌐 Partner & Competitor Intel",
    "📑 Multi-MSA Contract Portfolio",
    "📊 Business Case",
])

# ==========================================
# TAB 1: MARKET INTELLIGENCE & COMPETITOR ANALYSIS
# ==========================================
with tab1:
    target_company = st.text_input("Target Partner Company:", "Valeo")
    if st.button("Run Market & Competitor Scan"):
        with st.spinner("Analyzing partner strategy and identifying competitors..."):
            # Pass 1: Partner Scan
            partner_res = tavily.search(query=f"{target_company} strategic priorities partnerships automotive",
                                        max_results=3)
            partner_text = "\n".join([r['content'] for r in partner_res['results']])

            # Pass 2: Competitor Scan
            comp_res = tavily.search(query=f"top direct competitors of {target_company} automotive news", max_results=3)
            comp_text = "\n".join([r['content'] for r in comp_res['results']])

            intel_prompt = f"""
            You are an Executive BD Strategist. Analyze this data:
            PARTNER DATA ({target_company}):
            {partner_text}

            COMPETITOR DATA:
            {comp_text}

            Provide:
            1. **Target Partner Overview & Strategic Priorities**
            2. **Key Competitors & Market Landscape**
            3. **Comparative Threat Analysis** (Where competitors are winning vs {target_company})
            4. **Specific Partnership BD Strategy & Recommendations**
            """

            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": intel_prompt}]
            )
            st.markdown(response.choices[0].message.content)

# ==========================================
# TAB 2: MULTI-MSA PORTFOLIO RAG
# ==========================================
with tab2:
    st.subheader("1. Ingest New MSA into Portfolio")
    col1, col2 = st.columns(2)
    with col1:
        partner_name = st.text_input("Partner Name for this MSA:", "Bosch")
    with col2:
        uploaded_file = st.file_uploader("Upload MSA (PDF/DOCX)", type=["pdf", "docx"])

    if st.button("Index MSA to Vector Database") and uploaded_file and partner_name:
        extracted_text = ""
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            extracted_text = "\n".join([p.text for p in doc.paragraphs])

        # Store in Chroma Vector DB with Partner Metadata
        collection.add(
            documents=[extracted_text[:10000]],  # Store text chunk
            metadatas=[{"partner": partner_name, "filename": uploaded_file.name}],
            ids=[f"msa_{partner_name.lower()}"]
        )
        st.success(f"Indexed MSA for {partner_name} into permanent memory!")

    st.divider()
    st.subheader("2. Query Across All MSAs")
    query = st.text_input("Ask a portfolio-wide legal/commercial question:",
                          "Compare payment terms and liability caps across all agreements")

    if st.button("Run Portfolio Audit"):
        with st.spinner("Searching indexed MSA database..."):
            # RAG Retrieval from ChromaDB
            results = collection.query(query_texts=[query], n_results=3)
            retrieved_docs = "\n\n---\n\n".join(results['documents'][0]) if results[
                'documents'] else "No contracts found."

            rag_prompt = f"""
            You are a Senior Legal Counsel and BD Operations Director.
            Analyze these retrieved excerpts from our stored Master Services Agreements:

            {retrieved_docs}

            User Question: {query}

            Provide:
            1. **Direct Answer & Comparison Table**
            2. **Commercial Risk Audit**
            3. **Actionable Recommendations for BD Negotiations**
            """

            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": rag_prompt}]
            )
            st.markdown(response.choices[0].message.content)

# ==========================================
# TAB 3: CONVERSATIONAL BUSINESS CASE
# ==========================================
with tab3:
    st.subheader("Partnership Business Case")
    st.caption(
        "Step 1: classify the deal's business model. Step 2: fill in a year-by-year "
        "input grid (pre-filled from the term sheet, assumption-fillable). Step 3: "
        "P&L, NPV/ROI, and a chart computed straight from the grid."
    )

    # --- session state --------------------------------------------------------
    _BC_DEFAULTS = {
        "bc_extraction": None,       # dict from termsheet_extractor
        "bc_risks": None,            # list from risk_engine
        "bc_contract_name": None,
        "bc_classification": None,   # {business_model, reasoning, signals, confidence}
        "bc_model": None,            # confirmed model
        "bc_model_confirmed": False,
        "bc_inputs": {},             # side-channel: license_model_type / tier_breaks (licensing only)
        "bc_input_meta": {},         # field -> "real" | "user" | "assumed"
        "bc_override_ack": False,    # user acknowledged an unsupported model override
        # --- currency ---
        "bc_currency": None,         # "USD" | "EUR" | "GBP" | "Other" (the selectbox choice)
        "bc_currency_other": "",     # free-text ISO code when "Other" is chosen
        "bc_fx_result": None,        # cached fetch_fx_rate() output, None until fetched
        # --- step 2 (year-by-year input grid) ---
        "bc_grid_years": None,           # int N -> grid spans Year 0..N
        "bc_grid_opex_items": None,      # [{"key","label"}, ...] dynamic OPEX rows
        "bc_grid_row_labels": {},        # row_name -> user-renamed label (overrides the auto default)
        "bc_grid_values": None,          # row_name -> [float|None] length N+1
        "bc_grid_states": None,          # row_name -> ["prefilled"|"user"|"assumed"|"computed"|"blank"]
        "bc_grid_assumption_notes": {},  # "row_name|year" -> rationale, for every currently-assumed cell
        "bc_grid_review_dismissed": False, # user clicked "Edit values" - hide the review box until the next save
        "bc_grid_assumptions_ack": False,# user said Yes to the standing assumptions
        "bc_grid_confirmed": False,
        "bc_grid_save_flash": False,     # show a one-shot "Grid saved" confirmation after the next rerun
    }
    _BC_GRID_KEYS = [
        "bc_grid_years", "bc_grid_opex_items", "bc_grid_row_labels", "bc_grid_values",
        "bc_grid_states", "bc_grid_assumption_notes", "bc_grid_review_dismissed",
        "bc_grid_assumptions_ack", "bc_grid_confirmed", "bc_grid_save_flash",
    ]
    for _k, _v in _BC_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = copy.deepcopy(_v)

    def _bc_reset(from_step):
        steps = {
            "contract": ["bc_extraction", "bc_risks", "bc_contract_name", "bc_classification",
                         "bc_model", "bc_model_confirmed", "bc_inputs", "bc_input_meta",
                         "bc_override_ack", "bc_currency", "bc_currency_other", "bc_fx_result",
                         *_BC_GRID_KEYS],
            "model": ["bc_classification", "bc_model", "bc_model_confirmed", "bc_inputs",
                      "bc_input_meta", "bc_override_ack", *_BC_GRID_KEYS],
        }
        for key in steps[from_step]:
            st.session_state[key] = copy.deepcopy(_BC_DEFAULTS[key])
        st.session_state.pop("bc_grid_editor", None)

    # --- Step 2 helpers: the year-by-year input grid --------------------------
    _BC_STATE_COLORS = {
        "prefilled": "#cfe3ff",  # blue  - from the term sheet
        "user": "#ffffff",       # none  - typed in, nothing to flag
        "assumed": "#fff3b0",    # amber - accepted industry-standard proposal
        "computed": "#e6e6e6",   # grey  - rolled forward automatically
        "blank": "#ffd6d6",      # pink  - still needs a value
    }
    _BC_STATE_CHIPS = [
        ("Pre-filled from term sheet", "prefilled"),
        ("You entered", "user"),
        ("Assumed (industry default)", "assumed"),
        ("Computed automatically", "computed"),
        ("Still blank", "blank"),
    ]

    def _bc_grid_legend():
        chips = "".join(
            f'<span style="background:{_BC_STATE_COLORS[key]};border:1px solid #999;'
            f'border-radius:4px;padding:1px 8px;margin-right:8px;font-size:0.85em;">{label}</span>'
            for label, key in _BC_STATE_CHIPS
        )
        st.markdown(chips, unsafe_allow_html=True)

    def _bc_grid_row_specs(model, license_type=None, currency_symbol="$"):
        rows = [dict(r) for r in bc.MODEL_GRID_ROWS[model]]
        if model == "licensing":
            for r in rows:
                if r["name"] == "rate_or_fee":
                    r["label"] = ("Annual flat licence fee ({cur})" if license_type == "flat"
                                 else "Royalty per licensed unit ({cur})")
        rows.append(dict(bc.GRID_CAPEX_ROW))
        for item in (st.session_state.bc_grid_opex_items or []):
            rows.append({"name": item["key"], "label": item["label"], "kind": "opex"})
        for r in rows:
            r["label"] = bc.apply_currency_label(r["label"], currency_symbol)
        # A user-renamed label (via the grid's editable "Row label" column) always
        # wins over the auto-generated default above.
        custom = st.session_state.bc_grid_row_labels or {}
        for r in rows:
            if r["name"] in custom:
                r["label"] = custom[r["name"]]
        return rows

    def _bc_values_close(a, b):
        if (a is None) != (b is None):
            return False
        if a is None:
            return True
        return abs(a - b) < 1e-9

    def _bc_grid_editor_reset():
        """Drop the data_editor widget's cached state.

        Streamlit's ``st.data_editor`` remembers its own copy of the grid under
        its widget key and keeps returning/re-applying that cached copy across
        reruns, even when we overwrite ``bc_grid_values`` programmatically
        (Fill assumptions, add/remove an OPEX row, change the year count, ...).
        Left alone, that stale cache is what "Save grid edits" reads back on
        the next click - silently reverting a just-filled cell to its old
        value, or making an edit look like it "didn't take". Popping the key
        forces the widget to rebuild fresh from whatever is in session state.
        """
        st.session_state.pop("bc_grid_editor", None)

    def _bc_grid_ensure(model, extraction, row_specs, n_years):
        """Idempotent: makes sure every row has a values/states array of length n_years+1,
        pre-filling new rows and re-deriving the subscription waterfall."""
        if st.session_state.bc_grid_values is None:
            st.session_state.bc_grid_values = {}
        if st.session_state.bc_grid_states is None:
            st.session_state.bc_grid_states = {}
        prefill = bc.prefill_grid_cells(model, extraction, row_specs, n_years)
        for r in row_specs:
            name = r["name"]
            vals = st.session_state.bc_grid_values.get(name)
            sts = st.session_state.bc_grid_states.get(name)
            if vals is None:
                vals = [None] * (n_years + 1)
                sts = ["blank"] * (n_years + 1)
                for y, info in (prefill.get(name) or {}).items():
                    vals[y] = info["value"]
                    sts[y] = "prefilled"
            elif len(vals) != n_years + 1:
                if len(vals) < n_years + 1:
                    pad = n_years + 1 - len(vals)
                    vals = vals + [None] * pad
                    sts = sts + ["blank"] * pad
                else:
                    vals = vals[:n_years + 1]
                    sts = sts[:n_years + 1]
            st.session_state.bc_grid_values[name] = vals
            st.session_state.bc_grid_states[name] = sts
        if model == "subscription":
            bc.recompute_subscription_waterfall(st.session_state.bc_grid_values, n_years)
            for y in range(1, n_years + 1):
                st.session_state.bc_grid_states["beginning_base"][y] = "computed"
            for y in range(n_years + 1):
                st.session_state.bc_grid_states["ending_base"][y] = "computed"

    _BC_MONEY_KINDS = {"rate", "capex", "opex"}

    def _bc_grid_df(row_specs, n_years):
        # Index = the stable technical row name (never shown); "Row label" is the
        # user-editable display text, kept separate so renaming a row can never
        # break the name-based lookups the P&L math depends on. Alongside each
        # "Year N" value column sits a hidden "Year N__state" column (that cell's
        # prefilled/user/assumed/computed/blank state) and one "__is_money" column
        # per row - AgGrid's cellStyle/valueFormatter/editable JS callbacks read
        # these straight off the row data, so a single grid can be colored,
        # currency-formatted, and edit-locked without a separate read-only table.
        year_cols = [f"Year {y}" for y in range(n_years + 1)]
        idx = [r["name"] for r in row_specs]
        data = {"Row label": [r["label"] for r in row_specs]}
        for col in year_cols:
            data[col] = []
        for r in row_specs:
            vals = st.session_state.bc_grid_values[r["name"]]
            for y, col in enumerate(year_cols):
                v = vals[y]
                data[col].append(float("nan") if v is None else float(v))
        for y, col in enumerate(year_cols):
            data[f"{col}__state"] = [st.session_state.bc_grid_states[r["name"]][y] for r in row_specs]
        data["__is_money"] = [r.get("kind") in _BC_MONEY_KINDS for r in row_specs]
        data["__row_key"] = [r["name"] for r in row_specs]
        return pd.DataFrame(data, index=idx), year_cols

    def _bc_grid_options(df, year_cols, currency_symbol):
        """GridOptions for the single editable+colored AgGrid grid.

        Cell background, currency formatting, and which cells are locked
        (state == "computed") are all driven by the hidden "{col}__state" and
        "__is_money" columns baked into ``df`` by ``_bc_grid_df`` - the JS
        callbacks below read them straight off ``params.data``, so no
        parallel read-only table is needed to show the coloring.
        """
        state_colors_json = json.dumps(_BC_STATE_COLORS)
        currency_symbol_json = json.dumps(currency_symbol)

        cell_style = JsCode(f"""
            function(params) {{
                var colors = {state_colors_json};
                var state = params.data ? params.data[params.colDef.field + "__state"] : null;
                var bg = colors[state];
                return bg ? {{backgroundColor: bg, color: "#111"}} : {{}};
            }}
        """)
        is_editable = JsCode("""
            function(params) {
                var state = params.data ? params.data[params.colDef.field + "__state"] : null;
                return state !== "computed";
            }
        """)
        value_formatter = JsCode(f"""
            function(params) {{
                if (params.value === null || params.value === undefined || isNaN(params.value)) {{
                    return "\\u2014";
                }}
                var isMoney = params.data ? params.data["__is_money"] : false;
                var prefix = isMoney ? {currency_symbol_json} : "";
                return prefix + Number(params.value).toLocaleString(
                    undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}
                );
            }}
        """)
        value_parser = JsCode("""
            function(params) {
                if (params.newValue === null || params.newValue === undefined) return null;
                var s = String(params.newValue).trim();
                if (s === "" || s === "\\u2014") return null;
                var cleaned = s.replace(/[^0-9.\\-]/g, "");
                if (cleaned === "" || cleaned === "-" || cleaned === ".") return null;
                var v = parseFloat(cleaned);
                return isNaN(v) ? params.oldValue : v;
            }
        """)

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(resizable=True, sortable=False, filter=False, minWidth=130)
        gb.configure_column("Row label", editable=True, pinned="left", minWidth=240)
        for col in year_cols:
            gb.configure_column(
                col, type=["numericColumn"], editable=is_editable, cellStyle=cell_style,
                valueFormatter=value_formatter, valueParser=value_parser,
            )
            gb.configure_column(f"{col}__state", hide=True)
        gb.configure_column("__is_money", hide=True)
        gb.configure_column("__row_key", hide=True)
        # Key rows by their stable technical name rather than the wrapper's
        # default positional id, so row identity survives a full data refresh.
        gb.configure_grid_options(getRowId=JsCode("function(params) { return params.data['__row_key']; }"))
        # AgGrid updates a cell's displayed VALUE as soon as new row data arrives,
        # but doesn't always re-invoke that cell's cellStyle in the same pass -
        # an edited cell's background color (prefilled/assumed -> user) would lag
        # one Save/Fill cycle behind the value itself. Forcing a full cell
        # refresh whenever row data updates keeps color and value in sync.
        gb.configure_grid_options(onRowDataUpdated=JsCode(
            "function(e) { e.api.refreshCells({force: true}); }"
        ))
        return gb.build()

    def _bc_grid_blanks(row_specs, n_years):
        return [
            (r, y) for r in row_specs for y in range(n_years + 1)
            if st.session_state.bc_grid_states[r["name"]][y] == "blank"
        ]

    def _bc_grid_has_assumed(row_specs, n_years):
        return any(
            st.session_state.bc_grid_states[r["name"]][y] == "assumed"
            for r in row_specs for y in range(n_years + 1)
        )

    # --- readable extraction / risk views (replaces raw st.json dumps) --------
    _EXTRACTION_FIELD_LABELS = {
        "partner_name": "Partner / Counterparty",
        "contract_duration_months": "Contract Duration",
        "revenue_share": "Revenue Share",
        "revenue_share_details": "Revenue Share — Details",
        "minimum_volume": "Minimum Volume Commitment",
        "exclusivity": "Exclusivity",
        "payment_terms": "Payment Terms",
        "termination_terms": "Termination Terms",
        "renewal_terms": "Renewal Terms",
        "IP_Ownership": "IP Ownership",
    }

    def _bc_render_extraction(extraction):
        rows = []
        for field, label in _EXTRACTION_FIELD_LABELS.items():
            val = extraction.get(field)
            if field == "contract_duration_months" and val is not None:
                display = f"{val} months (~{val / 12:.1f} yrs)"
            elif val:
                display = str(val)
            else:
                display = "— not stated —"
            rows.append({"Field": label, "Value": display})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    _SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟢"}

    def _bc_render_risks(risks):
        if not risks:
            st.caption("No risks flagged.")
            return
        for r in risks:
            sev = str(r.get("severity") or "low").lower()
            icon = _SEVERITY_ICON.get(sev, "⚪")
            with st.container(border=True):
                st.markdown(f"{icon} **{r.get('risk_name', '(unnamed risk)')}** — _{sev.title()} severity_")
                if r.get("reason"):
                    st.write(r["reason"])
                if r.get("potential_impact"):
                    st.caption(f"**Potential impact:** {r['potential_impact']}")
                if r.get("recommended_action"):
                    st.caption(f"**Recommended action:** {r['recommended_action']}")

    # --- currency helper --------------------------------------------------
    def _bc_effective_currency():
        """(code, symbol) - the code actually in effect right now, from live
        session state, so every downstream render picks up an edit immediately."""
        choice = st.session_state.bc_currency or "USD"
        if choice == "Other":
            code = (st.session_state.bc_currency_other or "").strip().upper() or "USD"
        else:
            code = choice
        return code, bc.currency_symbol(code)

    def _bc_format_money_df(df, symbol):
        """Display-only copy of an all-monetary DataFrame with the currency prefix baked in."""
        out = df.copy().astype(object)
        for col in df.columns:
            out[col] = [f"{symbol}{v:,.0f}" for v in df[col]]
        return out

    # --- 0. Load the contract (extraction + risk register) -------------------
    if st.session_state.bc_extraction is None:
        st.markdown("#### Load a term sheet")
        st.write("Runs `termsheet_extractor` then `risk_engine`, then keeps the results for this session.")
        up = st.file_uploader("Term sheet (.docx)", type=["docx"], key="bc_docx")
        c1, c2 = st.columns(2)
        if c1.button("Extract & analyze", disabled=up is None, key="bc_extract_btn"):
            try:
                with st.spinner("Extracting term sheet..."):
                    extraction = extract_termsheet(io.BytesIO(up.getvalue()), groq_client=groq_client)
                with st.spinner("Building risk register..."):
                    risks = build_risk_register(extraction, groq_client=groq_client)
                st.session_state.bc_extraction = extraction
                st.session_state.bc_risks = risks
                st.session_state.bc_contract_name = up.name
                st.rerun()
            except (ExtractionError, RiskEngineError) as e:
                st.error(f"Could not process contract: {e}")

        _sample = "test_agreement_extracted.json"
        if os.path.exists(_sample) and c2.button("Load sample (test_agreement)", key="bc_sample_btn"):
            with open(_sample, "r", encoding="utf-8-sig") as fh:
                st.session_state.bc_extraction = json.load(fh)
            _rp = "test_agreement_risks.json"
            if os.path.exists(_rp):
                with open(_rp, "r", encoding="utf-8-sig") as fh:
                    st.session_state.bc_risks = json.load(fh)
            else:
                st.session_state.bc_risks = []
            st.session_state.bc_contract_name = "test_agreement (sample)"
            st.rerun()

    else:
        extraction = st.session_state.bc_extraction
        risks = st.session_state.bc_risks or []

        top_l, top_r = st.columns([3, 1])
        top_l.markdown(f"**Loaded contract:** {st.session_state.bc_contract_name}")
        if top_r.button("Load different contract", key="bc_reset_contract"):
            _bc_reset("contract")
            st.rerun()
        with st.expander("Term sheet extraction"):
            _bc_render_extraction(extraction)
        with st.expander(f"Risk register ({len(risks)} risk(s))"):
            _bc_render_risks(risks)

        st.divider()

        # --- Currency — always visible/editable, independent of step order ---
        st.markdown("### Currency")
        cls_for_currency = st.session_state.bc_classification or {}
        detected_currency = cls_for_currency.get("currency")
        cross_border = bool(cls_for_currency.get("cross_border"))
        counterparty_currency = cls_for_currency.get("counterparty_currency")
        currency_reasoning = cls_for_currency.get("currency_reasoning") or ""

        cur_options = list(bc.CURRENCY_CHOICES)
        default_choice = st.session_state.bc_currency or detected_currency or "USD"
        if default_choice not in cur_options:
            default_choice = "Other"
        cur_col1, cur_col2 = st.columns([1, 2])
        chosen_currency = cur_col1.selectbox(
            "Deal currency", cur_options, index=cur_options.index(default_choice), key="bc_currency_choice",
        )
        st.session_state.bc_currency = chosen_currency
        if chosen_currency == "Other":
            other_code = cur_col2.text_input(
                "Currency code (e.g. INR, AUD, JPY)", value=st.session_state.bc_currency_other,
                key="bc_currency_other_input",
            ).strip().upper()
            st.session_state.bc_currency_other = other_code

        if detected_currency:
            st.caption(
                f"Detected from the term sheet: **{detected_currency}**"
                + (f" — {currency_reasoning}" if currency_reasoning else "")
            )
        elif st.session_state.bc_classification is not None:
            st.caption("Currency wasn't explicitly stated in the term sheet — select one above.")

        _, effective_symbol = _bc_effective_currency()

        if cross_border and counterparty_currency and counterparty_currency != _bc_effective_currency()[0]:
            st.info(
                f"⚠️ This deal appears cross-border"
                + (f": {currency_reasoning}" if currency_reasoning else ".")
                + f" Counterparty currency: **{counterparty_currency}**."
            )
            if st.button("Fetch reference FX rate via Tavily", key="bc_fetch_fx"):
                try:
                    with st.spinner("Fetching a reference exchange rate..."):
                        st.session_state.bc_fx_result = fetch_fx_rate(
                            _bc_effective_currency()[0], counterparty_currency,
                            tavily_client=tavily, groq_client=groq_client,
                        )
                except FinancialEngineError as e:
                    st.error(str(e))
            fx = st.session_state.bc_fx_result
            if fx:
                _src = f"[{fx['source_url']}]({fx['source_url']})" if fx.get("source_url") else "n/a"
                st.markdown(
                    f"**Reference rate:** 1 {fx['from_currency']} = {fx['rate']:.4f} {fx['to_currency']}  \n"
                    f"Source: {_src} — _{fx['raw_quoted_text'] or '(no quoted text)'}_"
                )
                st.caption(fx["disclaimer"])
                st.caption(
                    "This rate is shown for reference only — grid, P&L and chart figures stay in "
                    f"**{_bc_effective_currency()[0]}** and are not auto-converted."
                )

        st.divider()

        # --- 1. Classify the business model ---------------------------------
        st.markdown("### Step 1 — Business model")
        if st.session_state.bc_classification is None:
            if st.button("Classify business model", key="bc_classify_btn"):
                try:
                    with st.spinner("Classifying the deal's business model..."):
                        st.session_state.bc_classification = bc.classify_business_model(
                            extraction, risks, groq_client
                        )
                    st.rerun()
                except bc.BusinessCaseError as e:
                    st.error(str(e))
        else:
            cls = st.session_state.bc_classification
            st.markdown(
                f"**Detected:** `{cls['business_model']}` "
                f"({bc.MODEL_LABELS[cls['business_model']]})  \n"
                f"**Confidence:** {cls['confidence']}  \n"
                f"**Why:** {cls['reasoning']}"
            )
            if cls.get("signals"):
                st.caption("Signals: " + " · ".join(cls["signals"]))

            options = list(bc.BUSINESS_MODELS)
            default_ix = options.index(cls["business_model"]) if cls["business_model"] in options else 0
            choice = st.radio(
                "Confirm or override the business model — you can change this any time, "
                "even after later steps are filled in:",
                options,
                index=default_ix,
                format_func=lambda m: bc.MODEL_LABELS[m],
                key="bc_model_choice",
            )
            cc1, cc2 = st.columns([1, 1])
            if cc1.button("Confirm model", key="bc_confirm_model"):
                changed = (choice != st.session_state.bc_model) or not st.session_state.bc_model_confirmed
                if changed:
                    _bc_reset("model")
                st.session_state.bc_classification = cls
                st.session_state.bc_model = choice
                st.session_state.bc_model_confirmed = True
                st.rerun()
            if st.session_state.bc_model_confirmed and cc2.button("Re-classify", key="bc_reclassify"):
                _bc_reset("model")
                st.rerun()

        # --- 2. Year-by-year input grid -------------------------------------
        if st.session_state.bc_model_confirmed:
            model = st.session_state.bc_model
            detected = (st.session_state.bc_classification or {}).get("business_model")
            currency_code, currency_symbol = _bc_effective_currency()
            st.divider()
            st.markdown(f"### Step 2 — Year-by-year inputs for a **{model}** model ({currency_code})")
            st.caption(
                "Nothing here locks. Change the model above, any grid cell, or any assumption "
                "at any time — Step 3 below always recalculates from whatever is currently in "
                "the grid, not from what it was when you first confirmed."
            )

            # If the user overrode the detected model, check the contract actually
            # has language for the model they picked. If not, the numbers are
            # hypothetical and the user must acknowledge that before proceeding.
            override_gate_ok = True
            if detected and model != detected:
                sig = bc.extraction_signals_for_model(model, extraction)
                if sig["supported"]:
                    st.caption(
                        f"Override accepted: detected **{detected}**, you chose **{model}**. "
                        f"Supporting contract language found: {', '.join(sig['matched'])}."
                    )
                else:
                    st.warning(
                        f"⚠️ You've selected **{model}** (overriding the detected **{detected}**), "
                        f"but this term sheet doesn't contain {model}-type language "
                        f"({sig['hint']}) — any numbers entered here are hypothetical and not "
                        "grounded in the actual contract."
                    )
                    override_gate_ok = st.checkbox(
                        f"I understand this {model} override isn't supported by the contract "
                        "language and will treat the results as hypothetical.",
                        key="bc_override_ack",
                    )
                    if not override_gate_ok:
                        st.info("Acknowledge the warning above to continue with this override.")

            if not override_gate_ok:
                st.stop()

            lmt = st.session_state.bc_inputs.get("license_model_type") or "flat" if model == "licensing" else None

            # Licence structure + tier table live outside the grid — they're a
            # rate schedule choice, not a per-year figure. Always editable.
            if model == "licensing":
                lmt = st.selectbox(
                    "Licence structure", ["flat", "per_unit", "tiered"],
                    index=["flat", "per_unit", "tiered"].index(lmt), key="bc_lmt",
                )
                st.session_state.bc_inputs["license_model_type"] = lmt
                st.session_state.bc_input_meta["license_model_type"] = "user"
                if lmt == "tiered":
                    st.caption(
                        "Tier breaks — marginal bracket pricing. First row must start at 0 units. "
                        "(The Royalty row in the grid below isn't used for a tiered structure.)"
                    )
                    default_rows = st.session_state.bc_inputs.get("tier_breaks") or [
                        {"min_units": 0.0, "rate_per_unit": 0.0}
                    ]
                    edited_tiers = st.data_editor(
                        pd.DataFrame(default_rows), num_rows="dynamic", key="bc_tiers",
                        column_config={
                            "min_units": st.column_config.NumberColumn("Units from", min_value=0.0),
                            "rate_per_unit": st.column_config.NumberColumn(
                                f"Rate / unit ({currency_symbol.strip()})", min_value=0.0),
                        },
                    )
                    st.session_state.bc_inputs["tier_breaks"] = [
                        {"min_units": float(r["min_units"]), "rate_per_unit": float(r["rate_per_unit"])}
                        for _, r in edited_tiers.iterrows()
                        if pd.notna(r["min_units"]) and pd.notna(r["rate_per_unit"])
                    ] or None
                else:
                    st.caption("Flat structure: the Royalty row below becomes the annual flat licence fee.")

            # --- projection horizon ---------------------------------------
            default_years = bc.default_grid_years(extraction)
            years_val = st.number_input(
                "Projection years (grid spans Year 0 through Year N)",
                min_value=1, max_value=40, step=1,
                value=int(st.session_state.bc_grid_years or default_years),
                key="bc_grid_years_input",
            )
            if st.session_state.bc_grid_years is not None and int(years_val) != st.session_state.bc_grid_years:
                _bc_grid_editor_reset()
            st.session_state.bc_grid_years = int(years_val)
            n_years = st.session_state.bc_grid_years

            # --- OPEX line items --------------------------------------------
            if st.session_state.bc_grid_opex_items is None:
                st.session_state.bc_grid_opex_items = [
                    {"key": "opex__0__opex", "label": bc.DEFAULT_OPEX_LABEL}
                ]
            st.markdown("**OPEX line items**")
            oc1, oc2 = st.columns([3, 1])
            new_opex_label = oc1.text_input(
                "New OPEX line item", value="", key="bc_grid_new_opex_label",
                placeholder="e.g. Support & operations", label_visibility="collapsed",
            )
            if oc2.button("Add line", key="bc_grid_add_opex"):
                items = st.session_state.bc_grid_opex_items
                label = new_opex_label.strip() or f"OPEX {len(items) + 1}"
                slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "opex"
                new_key = f"opex__{len(items)}__{slug}"
                st.session_state.bc_grid_opex_items = items + [{"key": new_key, "label": label}]
                _bc_grid_editor_reset()
                st.rerun()
            if len(st.session_state.bc_grid_opex_items) > 1:
                remove_pick = st.selectbox(
                    "Remove an OPEX line item", ["(none)"] +
                    [i["label"] for i in st.session_state.bc_grid_opex_items],
                    key="bc_grid_remove_opex_pick",
                )
                if remove_pick != "(none)" and st.button("Remove selected line", key="bc_grid_remove_opex_btn"):
                    keep = [i for i in st.session_state.bc_grid_opex_items if i["label"] != remove_pick]
                    removed = [i for i in st.session_state.bc_grid_opex_items if i["label"] == remove_pick]
                    st.session_state.bc_grid_opex_items = keep
                    for i in removed:
                        st.session_state.bc_grid_values.pop(i["key"], None)
                        st.session_state.bc_grid_states.pop(i["key"], None)
                    _bc_grid_editor_reset()
                    st.rerun()

            row_specs = _bc_grid_row_specs(model, lmt, currency_symbol)
            _bc_grid_ensure(model, extraction, row_specs, n_years)

            # --- legend + single editable, colored grid ----------------------
            st.markdown("#### Year-by-year grid")
            _bc_grid_legend()
            display_df, year_cols = _bc_grid_df(row_specs, n_years)
            st.caption(
                "Cell color shows each value's state (legend above); edit any non-grey cell "
                "directly and press **Save grid edits**. This grid stays editable even after "
                "Step 3 appears below."
            )

            with st.expander("Row definitions"):
                for r in row_specs:
                    if r.get("help"):
                        # Escape "$" - a currency-suffixed label ("... ($)") can otherwise
                        # be misread as an inline LaTeX math delimiter by st.caption.
                        safe_label = r["label"].replace("$", "\\$")
                        st.caption(f"**{safe_label}** — {r['help']}")

            st.caption("Row label is editable too — rename a row if the auto-generated label doesn't fit this deal.")
            grid_response = AgGrid(
                display_df, gridOptions=_bc_grid_options(display_df, year_cols, currency_symbol),
                key="bc_grid_editor", height=min(120 + 40 * len(display_df), 600),
                update_mode=GridUpdateMode.MODEL_CHANGED, data_return_mode=DataReturnMode.AS_INPUT,
                allow_unsafe_jscode=True, theme="streamlit", fit_columns_on_grid_load=True,
                # Default "client_wins" makes the grid ignore server data after the first
                # edit - so a later Fill/Save silently kept showing stale (blank) cells in
                # the browser while the P&L below was already computing off the correct,
                # updated values underneath. "server_wins" keeps the grid in sync with
                # bc_grid_values, which is the actual source of truth here.
                server_sync_strategy="server_wins",
            )
            edited_df = pd.DataFrame(grid_response.data)[["Row label"] + year_cols]
            if st.button("Save grid edits", key="bc_grid_save"):
                for i, r in enumerate(row_specs):
                    name = r["name"]
                    new_label = str(edited_df.iloc[i]["Row label"]).strip()
                    if new_label and new_label != r["label"]:
                        st.session_state.bc_grid_row_labels[name] = new_label
                    for y, col in enumerate(year_cols):
                        old_state = st.session_state.bc_grid_states[name][y]
                        if old_state == "computed" or (r["kind"] == "seed" and y > 0):
                            continue
                        raw = edited_df.iloc[i][col]
                        new_val = None if pd.isna(raw) else float(raw)
                        old_val = st.session_state.bc_grid_values[name][y]
                        if not _bc_values_close(new_val, old_val):
                            st.session_state.bc_grid_values[name][y] = new_val
                            st.session_state.bc_grid_states[name][y] = "blank" if new_val is None else "user"
                            st.session_state.bc_grid_assumption_notes.pop(f"{name}|{y}", None)
                if model == "subscription":
                    bc.recompute_subscription_waterfall(st.session_state.bc_grid_values, n_years)
                    for y in range(1, n_years + 1):
                        st.session_state.bc_grid_states["beginning_base"][y] = "computed"
                    for y in range(n_years + 1):
                        st.session_state.bc_grid_states["ending_base"][y] = "computed"
                # Saving never turns a cell "assumed" - it only ever produces "user" or
                # "blank" - so it never introduces anything new that needs review. Leave
                # bc_grid_assumptions_ack as-is: if the user already accepted the standing
                # assumptions, editing an unrelated (already-known) value and saving goes
                # straight back to Step 3 instead of re-litigating assumptions they already
                # approved. review_dismissed still resets so that IF a review is still
                # outstanding (ack was never given), it re-opens in full rather than
                # staying collapsed in "please edit" mode.
                st.session_state.bc_grid_review_dismissed = False
                st.session_state.bc_grid_save_flash = True
                _bc_grid_editor_reset()
                st.rerun()
            if st.session_state.bc_grid_save_flash:
                st.success("Grid saved.")
                st.session_state.bc_grid_save_flash = False

            # --- fill remaining blanks ---------------------------------------
            blanks = _bc_grid_blanks(row_specs, n_years)
            if blanks:
                st.warning(f"{len(blanks)} cell(s) still blank.")
                st.markdown(
                    "**Would you like me to propose assumptions for the rest, reasoning about how "
                    "each value might reasonably change year to year?**"
                )
                if st.button("Fill remaining blanks with assumptions", key="bc_grid_fill"):
                    with st.spinner("Drafting year-by-year assumptions..."):
                        proposals = bc.propose_grid_assumptions(
                            model, extraction, row_specs, st.session_state.bc_grid_values,
                            st.session_state.bc_grid_states, n_years, groq_client,
                        )
                    for r in row_specs:
                        name = r["name"]
                        for y, info in (proposals.get(name) or {}).items():
                            st.session_state.bc_grid_values[name][y] = info["value"]
                            st.session_state.bc_grid_states[name][y] = "assumed"
                            st.session_state.bc_grid_assumption_notes[f"{name}|{y}"] = info["rationale"]
                    if model == "subscription":
                        bc.recompute_subscription_waterfall(st.session_state.bc_grid_values, n_years)
                    st.session_state.bc_grid_assumptions_ack = False
                    st.session_state.bc_grid_review_dismissed = False
                    _bc_grid_editor_reset()
                    st.rerun()

            # --- review outstanding assumptions ---------------------------------
            # Shown whenever any cell is currently "assumed" and not yet acknowledged
            # (not just right after Fill) - so editing an already-reviewed grid and
            # saving reliably re-asks for confirmation, and "Edit values" always has
            # a way back to this prompt. This works the same whether Step 3 is
            # already showing or not - it's never a hard lock either way.
            has_assumed = _bc_grid_has_assumed(row_specs, n_years)
            needs_review = has_assumed and not st.session_state.bc_grid_assumptions_ack
            if needs_review and not st.session_state.bc_grid_review_dismissed:
                st.markdown("#### Here are the assumptions I've made")
                st.caption(
                    "One line per row, summarizing the pattern behind that row's assumed values — "
                    "the grid above still has the exact figures, fully visible and editable."
                )
                for r in row_specs:
                    name = r["name"]
                    points = [
                        (y, st.session_state.bc_grid_values[name][y])
                        for y in range(n_years + 1)
                        if st.session_state.bc_grid_states[name][y] == "assumed"
                    ]
                    if not points:
                        continue
                    rationale_by_year = {
                        y: st.session_state.bc_grid_assumption_notes.get(f"{name}|{y}", "")
                        for y, _ in points
                    }
                    line = bc.summarize_assumed_row(
                        r["label"], r["kind"], points, rationale_by_year, currency_symbol,
                    )
                    st.markdown(f"- {line}")
                st.markdown("**Do these look reasonable?**")
                yc, ec = st.columns(2)
                if yc.button("Yes, looks good — continue to Step 3", key="bc_grid_review_yes"):
                    st.session_state.bc_grid_assumptions_ack = True
                    st.session_state.bc_grid_confirmed = True
                    st.rerun()
                if ec.button("Edit values", key="bc_grid_review_edit"):
                    st.session_state.bc_grid_review_dismissed = True
                    st.rerun()
            elif needs_review and st.session_state.bc_grid_review_dismissed:
                st.info(
                    "Edit any cell (assumed or otherwise) in the grid above and press "
                    "**Save grid edits** to re-open the assumption review."
                )

            # --- reveal gate for Step 3 — a one-way ratchet, not a lock ---------
            # Confirming just reveals Step 3 the first time. It never hides the
            # grid again, never freezes it, and never needs re-clicking after
            # later edits - Step 3 (once revealed) just keeps recalculating live.
            remaining_blanks = _bc_grid_blanks(row_specs, n_years)
            if not st.session_state.bc_grid_confirmed:
                if remaining_blanks:
                    st.info("Fill (or assume) every blank cell to reveal Step 3 — P&L.")
                elif needs_review:
                    pass  # review block above already explains what's needed
                elif st.button("Confirm inputs & continue to Step 3", key="bc_grid_confirm"):
                    st.session_state.bc_grid_confirmed = True
                    st.rerun()
            elif remaining_blanks:
                st.caption(
                    f"{len(remaining_blanks)} cell(s) are currently blank — Step 3 below treats "
                    "them as 0 until you fill them in."
                )

            # --- Step 3 — P&L computed live from the grid's current values ------
            if st.session_state.bc_grid_confirmed:
                st.divider()
                st.markdown("### Step 3 — P&L (from the year-by-year grid)")
                try:
                    pl = bc.build_pl_from_grid(
                        model, row_specs, st.session_state.bc_grid_values, n_years,
                        license_model_type=lmt, tier_breaks=st.session_state.bc_inputs.get("tier_breaks"),
                    )
                except bc.BusinessCaseError as e:
                    st.error(str(e))
                    pl = None

                if pl is not None:
                    for _note in pl["notes"]:
                        st.info(_note)
                    if model == "licensing":
                        st.caption(
                            "For a licence deal the **Revenue** row is the licence fee the "
                            "provider receives (equivalently, the OEM's licensing cost)."
                        )
                    st.dataframe(_bc_format_money_df(pl["df"], currency_symbol), width="stretch")

                    _pl_df = pl["df"]
                    chart_rows = [
                        {
                            "year": col,
                            "Revenue": float(_pl_df.loc["Revenue", col]),
                            "Cost (CAPEX+OPEX)": float(_pl_df.loc["CAPEX", col] + _pl_df.loc["OPEX", col]),
                            "Net Profit": float(_pl_df.loc["Net Profit", col]),
                        }
                        for col in _pl_df.columns
                    ]
                    st.altair_chart(
                        generate_pl_chart(
                            chart_rows, title="Revenue vs. Cost vs. Net Profit by year",
                            currency_symbol=currency_symbol,
                        ),
                        width="stretch",
                    )
                    st.caption("Grouped bars — Revenue, Cost, and Net Profit side by side for each year.")

                    disc_rate = st.number_input(
                        "Annual discount rate", min_value=0.0, max_value=1.0,
                        value=0.10, step=0.01, format="%.2f", key="bc_grid_discount_rate",
                    )
                    nr = None
                    try:
                        nr = bc.compute_npv_roi(pl, discount_rate=disc_rate)
                    except Exception as e:  # noqa: BLE001 - surface, don't crash the tab
                        st.error(f"NPV / ROI calculation failed: {e}")

                    if nr is not None:
                        mcol1, mcol2 = st.columns(2)
                        mcol1.metric(f"NPV @ {disc_rate:.0%}", f"{currency_symbol}{nr['npv']['npv']:,.0f}")
                        if nr["roi"] is not None:
                            mcol2.metric("ROI (cumulative profit ÷ CAPEX+OPEX)", f"{nr['roi']['roi_pct']:.1f}%")
                        else:
                            mcol2.info("ROI needs a non-zero investment base (CAPEX + OPEX).")

