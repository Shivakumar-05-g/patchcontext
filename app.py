import streamlit as st
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent))

from src import config
from src.vector_store import load_vector_store
from src.rag_chain import generate_answer
from src.hallucination_guard import HallucinationGuard
from src.citations import convert_citations_to_markdown_links

# Page Configuration
st.set_page_config(
    page_title="PatchContext - FastAPI RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS for rich aesthetics and clean B.Tech intern style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Mono&display=swap');
    
    /* Font overrides */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stCodeBlock, code {
        font-family: 'Space Mono', monospace !important;
    }

    /* Primary title banner gradient */
    .title-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.2);
    }
    
    .title-banner h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    
    .title-banner p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Badges for document types */
    .badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .badge-commit {
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    
    .badge-pr {
        background-color: #f3e8ff;
        color: #6b21a8;
        border: 1px solid #e9d5ff;
    }
    
    .badge-issue {
        background-color: #ffedd5;
        color: #9a3412;
        border: 1px solid #fed7aa;
    }

    /* Answer and context cards */
    .response-card {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }

    /* NLI Guard status cards */
    .guard-banner-safe {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        color: #065f46;
        margin-bottom: 1.5rem;
    }
    
    .guard-banner-warning {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        color: #991b1b;
        margin-bottom: 1.5rem;
    }

    /* Micro animations */
    .hover-grow {
        transition: transform 0.2s ease-in-out;
    }
    
    .hover-grow:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Helper to check if FAISS index exists
def is_index_available():
    index_file = config.VECTORSTORE_PATH / "index.faiss"
    pkl_file = config.VECTORSTORE_PATH / "index.pkl"
    return index_file.exists() and pkl_file.exists()

# Initialize Hallucination Guard (cached in session state)
if "hallucination_guard" not in st.session_state:
    with st.spinner("Initializing Hallucination Guard NLI model..."):
        # We start with local NLI enabled; if it fails, it falls back to LLM automatically
        st.session_state.hallucination_guard = HallucinationGuard(use_local_nli=True)

# App Title Banner
st.markdown("""
<div class="title-banner">
    <h1>PatchContext</h1>
    <p>Grounded historical RAG pipeline over the FastAPI repository's development history.</p>
</div>
""", unsafe_allow_html=True)

# Check index availability first
if not is_index_available():
    st.error("❌ FAISS Vector Store Index Not Found!")
    st.markdown("""
    The vector store does not exist. You must run the data ingestion script first to collect data and build the FAISS index.
    
    To fix this, run the following command in your terminal:
    ```bash
    python scripts/ingest.py
    ```
    """)
    st.stop()

# Sidebar controls
st.sidebar.header("🛠️ Configuration")

st.sidebar.markdown("### Retrieval Settings")
k_param = st.sidebar.slider("Chunks to retrieve (k)", min_value=1, max_value=15, value=config.DEFAULT_K)
fetch_k_param = st.sidebar.slider("Initial candidates (fetch_k)", min_value=5, max_value=40, value=config.DEFAULT_FETCH_K)

st.sidebar.markdown("### Model Details")
st.sidebar.info(f"""
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local CPU)
- **Vector DB**: FAISS (Index persisted)
- **Generator**: ChatGroq (`{config.GROQ_MODEL}`)
- **NLI Model**: `cross-encoder/nli-deberta-v3-small`
""")

# Example Questions section
st.markdown("### 💡 Example Questions")
examples = [
    "Why did FastAPI modify APIRoute handler caching or routing cache?",
    "Why did FastAPI refactor router route building to make it thread-safe?",
    "What changes did Sebastian Ramirez merge recently regarding thread safety?",
    "Is there any issue or PR details on candidate cache corruption?"
]

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

cols = st.columns(2)
for i, ex in enumerate(examples):
    col_idx = i % 2
    if cols[col_idx].button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state.query_input = ex

# Input field
st.text_input(
    "Ask a question about FastAPI's historical decisions:",
    key="query_input",
    placeholder="e.g. Why was this behavior changed?"
)

if st.button("Query PatchContext", type="primary"):
    query = st.session_state.query_input.strip()

    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Patching context, retrieving history, and generating grounded answer..."):
            # 1. Generate answer using Groq and MMR retrieval
            rag_response = generate_answer(query, k=k_param, fetch_k=fetch_k_param)
            
            answer = rag_response.get("answer", "")
            retrieved_docs = rag_response.get("retrieved_docs", [])
            
            # 2. Run Hallucination Guard (NLI Audit + Citation Validation)
            guard_report = st.session_state.hallucination_guard.check_hallucinations(answer, retrieved_docs)
            
            # 3. Format Answer with clickable markdown hyperlinks
            validated_cits = guard_report.get("validated_citations", [])
            formatted_answer = convert_citations_to_markdown_links(answer, validated_cits)
            
            # Main Layout for Results
            st.markdown("### 🤖 Answer")
            st.markdown(f'<div class="response-card">{formatted_answer}</div>', unsafe_allow_html=True)
            
            # NLI Hallucination Guard Audit Results
            st.markdown("### 🛡️ NLI Hallucination Guard Report")
            is_safe = guard_report.get("is_safe", False)
            
            if is_safe:
                st.markdown("""
                <div class="guard-banner-safe">
                    <strong>✅ GROUNDED:</strong> All statements in this answer are fully supported (entailed) by the retrieved repository history context, and all citations are authentic.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="guard-banner-warning">
                    <strong>⚠️ POTENTIAL HALLUCINATION DETECTED:</strong> Some claims in the answer could not be verified against the retrieved context, or fabricated citations were found. See details below.
                </div>
                """, unsafe_allow_html=True)
                
            # Citation verification details
            invalid_cits = guard_report.get("invalid_citations", [])
            if invalid_cits:
                st.markdown("#### ❌ Invalid/Fabricated Citations")
                for cit in invalid_cits:
                    st.error(f"Fabricated citation referenced in answer: `{cit['raw']}` (Not found in retrieved metadata!)")
            
            with st.expander("🔍 Statement-by-Statement NLI Verification Details", expanded=not is_safe):
                for report in guard_report.get("statement_reports", []):
                    status_icon = "✅"
                    status_color = "green"
                    if report["flagged"]:
                        status_icon = "⚠️"
                        status_color = "orange" if report["nli_label"] == "neutral" else "red"
                        
                    st.markdown(f"""
                    - **Statement**: "{report['statement']}"
                      - **NLI Verdict**: <span style="color:{status_color};font-weight:bold;">{report['nli_label'].upper()}</span> (confidence: {report['confidence']:.2f})
                      - **Audited via**: `{report['method']}`
                    """, unsafe_allow_html=True)
                    st.divider()

            # Expandable Retrieved Source Documents
            st.markdown("### 📂 Retrieved Historical Context (Source Documents)")
            for idx, doc in enumerate(retrieved_docs):
                source_type = doc.metadata.get("source_type", "unknown")
                source_url = doc.metadata.get("source_url", "#")
                author = doc.metadata.get("author", "unknown")
                date = doc.metadata.get("created_date", "n/a")
                title = doc.metadata.get("title", "No Title")
                
                badge_class = f"badge-{source_type.replace('_', '')}"
                
                # Expandable card for each source document
                with st.expander(f"Document [{idx+1}] | {source_type.upper()}: {title[:60]}...", expanded=False):
                    st.markdown(f"""
                    <span class="badge {badge_class}">{source_type.upper()}</span>
                    * **Author**: `{author}`
                    * **Created Date**: `{date}`
                    * **GitHub Link**: [View on GitHub]({source_url})
                    """, unsafe_allow_html=True)
                    st.code(doc.page_content, language="markdown")
