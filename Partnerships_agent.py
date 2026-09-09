import streamlit as st
import chromadb
from tavily import TavilyClient
from groq import Groq
from pypdf import PdfReader
import docx
import os
from dotenv import load_dotenv

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

tab1, tab2 = st.tabs(["🌐 Partner & Competitor Intel", "📑 Multi-MSA Contract Portfolio"])

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
