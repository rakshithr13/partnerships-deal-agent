import streamlit as st
import chromadb
from tavily import TavilyClient
from groq import Groq
from pypdf import PdfReader
import docx
import io
import json
import os
import copy
from dotenv import load_dotenv

import pandas as pd

import business_case as bc
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
# TAB 3: CONVERSATIONAL BUSINESS CASE (steps 1-3)
# ==========================================
with tab3:
    st.subheader("Partnership Business Case")
    st.caption(
        "Steps 1-3: classify the deal's business model, gather the revenue-model "
        "inputs, and run Low / Medium / High scenarios. P&L / NPV / cost gates come next."
    )

    # --- session state --------------------------------------------------------
    _BC_DEFAULTS = {
        "bc_extraction": None,       # dict from termsheet_extractor
        "bc_risks": None,            # list from risk_engine
        "bc_contract_name": None,
        "bc_classification": None,   # {business_model, reasoning, signals, confidence}
        "bc_model": None,            # confirmed model
        "bc_model_confirmed": False,
        "bc_inputs": {},             # field -> value (or None if still missing)
        "bc_input_meta": {},         # field -> "real" | "user" | "assumed"
        "bc_proposed": None,         # field -> {value, rationale, source}
        "bc_inputs_confirmed": False,
        "bc_scenarios": None,        # run_scenarios() output
        "bc_override_ack": False,    # user acknowledged an unsupported model override
    }
    for _k, _v in _BC_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = copy.deepcopy(_v)

    def _bc_reset(from_step):
        steps = {
            "contract": ["bc_extraction", "bc_risks", "bc_contract_name", "bc_classification",
                         "bc_model", "bc_model_confirmed", "bc_inputs", "bc_input_meta",
                         "bc_proposed", "bc_inputs_confirmed", "bc_scenarios", "bc_override_ack"],
            "model": ["bc_classification", "bc_model", "bc_model_confirmed", "bc_inputs",
                      "bc_input_meta", "bc_proposed", "bc_inputs_confirmed", "bc_scenarios",
                      "bc_override_ack"],
            "scenarios": ["bc_scenarios"],
        }
        for key in steps[from_step]:
            st.session_state[key] = copy.deepcopy(_BC_DEFAULTS[key])

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
            st.json(extraction)
        with st.expander(f"Risk register ({len(risks)} risk(s))"):
            st.json(risks)

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
                "Confirm or override the business model:",
                options,
                index=default_ix,
                format_func=lambda m: bc.MODEL_LABELS[m],
                key="bc_model_choice",
            )
            cc1, cc2 = st.columns([1, 1])
            if cc1.button("Confirm model", key="bc_confirm_model"):
                _bc_reset("model")
                st.session_state.bc_classification = cls
                st.session_state.bc_model = choice
                st.session_state.bc_model_confirmed = True
                st.rerun()
            if st.session_state.bc_model_confirmed and cc2.button("Re-classify", key="bc_reclassify"):
                _bc_reset("model")
                st.rerun()

        # --- 2. Gather the revenue-model inputs ----------------------------
        if st.session_state.bc_model_confirmed:
            model = st.session_state.bc_model
            detected = (st.session_state.bc_classification or {}).get("business_model")
            st.divider()
            st.markdown(f"### Step 2 — Inputs for a **{model}** model")

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

            prefill = bc.prefill_inputs_from_extraction(model, extraction)

            # licensing: structure + tier table live outside the form so they
            # re-run immediately and drive which fields are required.
            if model == "licensing":
                cur_lmt = st.session_state.bc_inputs.get("license_model_type") or "flat"
                lmt = st.selectbox(
                    "Licence structure", ["flat", "per_unit", "tiered"],
                    index=["flat", "per_unit", "tiered"].index(cur_lmt), key="bc_lmt",
                )
                st.session_state.bc_inputs["license_model_type"] = lmt
                st.session_state.bc_input_meta["license_model_type"] = "user"
                if lmt == "tiered":
                    st.caption("Tier breaks — marginal bracket pricing. First row must start at 0 units.")
                    default_rows = st.session_state.bc_inputs.get("tier_breaks") or [
                        {"min_units": 0.0, "rate_per_unit": 0.0}
                    ]
                    edited = st.data_editor(
                        pd.DataFrame(default_rows), num_rows="dynamic", key="bc_tiers",
                        column_config={
                            "min_units": st.column_config.NumberColumn("Units from", min_value=0.0),
                            "rate_per_unit": st.column_config.NumberColumn("Rate / unit ($)", min_value=0.0),
                        },
                    )
                    st.session_state.bc_inputs["tier_breaks"] = [
                        {"min_units": float(r["min_units"]), "rate_per_unit": float(r["rate_per_unit"])}
                        for _, r in edited.iterrows()
                        if pd.notna(r["min_units"]) and pd.notna(r["rate_per_unit"])
                    ] or None

            req = bc.required_fields(model, st.session_state.bc_inputs)
            specs_by_name = {s["name"]: s for s in bc.MODEL_INPUT_SPECS[model]}

            with st.form("bc_inputs_form"):
                st.write("Supply what you have. Pre-filled fields come straight from the term sheet.")
                new_vals = {}
                for name in req:
                    spec = specs_by_name[name]
                    if spec["kind"] in ("choice", "tiers"):
                        continue
                    cur = st.session_state.bc_inputs.get(name)
                    pf = prefill.get(name)
                    default = cur if cur is not None else (pf["value"] if pf else None)

                    kwargs = {"help": spec.get("help")}
                    if spec["kind"] == "rate":
                        kwargs.update(min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
                    elif spec["kind"] == "years":
                        kwargs.update(min_value=0.0, step=0.5)
                    else:
                        kwargs.update(min_value=0.0, step=1.0)

                    new_vals[name] = st.number_input(
                        spec["label"],
                        value=(float(default) if default is not None else None),
                        **kwargs,
                    )
                    if pf:
                        st.caption(f"↳ pre-filled from {pf['source']}")

                if st.form_submit_button("Save inputs"):
                    for name, val in new_vals.items():
                        if val is None:
                            st.session_state.bc_inputs[name] = None
                            st.session_state.bc_input_meta.pop(name, None)
                            continue
                        st.session_state.bc_inputs[name] = val
                        pf = prefill.get(name)
                        if pf is not None and abs(val - float(pf["value"])) < 1e-9:
                            st.session_state.bc_input_meta[name] = "real"
                        else:
                            st.session_state.bc_input_meta[name] = "user"
                    st.session_state.bc_proposed = None
                    st.session_state.bc_inputs_confirmed = False
                    st.rerun()

            miss = bc.missing_fields(model, st.session_state.bc_inputs)

            if miss:
                st.warning("Still missing: " + ", ".join(miss))
                st.markdown(
                    "**Would you like me to propose assumptions for the rest based on "
                    "the commercial terms and industry standards?**"
                )
                if st.button("Yes — propose assumptions", key="bc_propose_btn"):
                    with st.spinner("Drafting clearly-labelled industry-standard assumptions..."):
                        st.session_state.bc_proposed = bc.propose_assumptions(
                            model, extraction, miss, groq_client
                        )
                    st.rerun()

            if st.session_state.bc_proposed:
                st.markdown("#### Proposed assumptions")
                st.caption("Nothing here is applied until you tick **Accept** and press the button.")
                accepts, overrides = {}, {}
                for field, info in st.session_state.bc_proposed.items():
                    with st.container(border=True):
                        tag = "industry standard (Groq)" if info["source"] == "groq" else "industry standard (fallback)"
                        st.markdown(f"**{field}** — _{tag}_")
                        st.caption(info["rationale"])
                        if info["value"] is None:
                            st.error("No responsible default — supply a real figure in the form above.")
                            continue
                        overrides[field] = st.number_input(
                            f"Value for {field}",
                            value=float(info["value"]),
                            min_value=0.0,
                            max_value=(1.0 if field in bc._RATE_FIELDS else None),
                            key=f"bc_assume_val_{field}",
                        )
                        accepts[field] = st.checkbox(
                            f"Accept this assumption for {field}", key=f"bc_assume_ok_{field}"
                        )
                if st.button("Apply accepted assumptions", key="bc_apply_assume"):
                    for field, ok in accepts.items():
                        if ok:
                            st.session_state.bc_inputs[field] = overrides[field]
                            st.session_state.bc_input_meta[field] = "assumed"
                    st.rerun()

            if not bc.missing_fields(model, st.session_state.bc_inputs):
                st.markdown("#### Confirmed inputs")
                _rows = []
                for name in bc.required_fields(model, st.session_state.bc_inputs):
                    val = st.session_state.bc_inputs.get(name)
                    if isinstance(val, bool):
                        shown = str(val)
                    elif isinstance(val, (int, float)):
                        shown = f"{val:,.4g}"
                    else:
                        shown = json.dumps(val)
                    _rows.append({
                        "input": name,
                        "value": shown,
                        "source": st.session_state.bc_input_meta.get(name, "user"),
                    })
                st.dataframe(pd.DataFrame(_rows), hide_index=True, width="stretch")
                st.caption("source: **real** = from term sheet · **user** = you entered it · **assumed** = accepted assumption")

                bcol1, bcol2 = st.columns([1, 1])
                if bcol1.button("Confirm inputs & continue", key="bc_confirm_inputs"):
                    st.session_state.bc_inputs_confirmed = True
                    st.session_state.bc_scenarios = None
                    st.rerun()
                if bcol2.button("Edit inputs", key="bc_edit_inputs"):
                    # Go back to the form WITHOUT discarding saved values — the
                    # form repopulates from st.session_state.bc_inputs. Only drop
                    # the confirmation + any downstream scenario results.
                    st.session_state.bc_inputs_confirmed = False
                    st.session_state.bc_scenarios = None
                    st.rerun()

        # --- 3. Low / Medium / High scenarios -----------------------------
        if st.session_state.bc_inputs_confirmed:
            model = st.session_state.bc_model
            st.divider()
            st.markdown("### Step 3 — Low / Medium / High scenarios")

            if st.session_state.bc_scenarios is None:
                if st.button("Run scenarios", key="bc_run_scen"):
                    try:
                        st.session_state.bc_scenarios = bc.run_scenarios(
                            model, st.session_state.bc_inputs
                        )
                    except bc.BusinessCaseError as e:
                        st.error(str(e))
                    st.rerun()

            sc = st.session_state.bc_scenarios
            if sc:
                st.caption(
                    "Driver field(s): " + ", ".join(sc["driver_fields"])
                    + "  ·  Low/Med/High = pessimistic / base / optimistic on the driver(s)."
                )
                if sc["note"]:
                    st.info(sc["note"])

                cols = st.columns(3)
                for col, name in zip(cols, bc.SCENARIO_NAMES):
                    s = sc["scenarios"][name]
                    with col:
                        st.markdown(f"**{name.title()}**")
                        st.metric(sc["headline_label"], f"${s['headline']:,.0f}")
                        if s["adjusted"]:
                            st.caption("Driver values used:")
                            for k, v in s["adjusted"].items():
                                mult = s["multipliers"].get(k)
                                st.caption(f"• {k} = {v:,.4g}  (×{mult})")
                        yearly = s["result"].get("yearly")
                        if yearly:
                            ydf = pd.DataFrame(yearly)[
                                [c for c in ("year", "active_base", "revenue") if c in yearly[0]]
                            ]
                            st.dataframe(ydf, hide_index=True, width="stretch")

                if st.button("Re-run scenarios", key="bc_rerun_scen"):
                    _bc_reset("scenarios")
                    st.rerun()
