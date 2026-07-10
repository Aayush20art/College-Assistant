import os
import time
import tempfile
import hashlib
from typing import TypedDict, Annotated

import streamlit as st
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------
# Inject the Mistral API key from Streamlit secrets into the environment
# BEFORE any module-level client initialization happens.
# -----------------------------------------------------------------------
if "MISTRAL_API_KEY" in st.secrets:
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]


# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================
# CUSTOM CSS — Blue & White light theme with animations
# =========================================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root{
    --primary-blue:#2563EB;
    --deep-blue:#1E40AF;
    --sky-blue:#60A5FA;
    --pale-blue:#EFF6FF;
    --ice-blue:#DBEAFE;
    --white:#FFFFFF;
    --slate:#334155;
    --slate-light:#64748B;
}

/* ---------- Global ---------- */
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--slate);
}

.stApp {
    background: linear-gradient(160deg, #FFFFFF 0%, #F0F6FF 45%, #E8F1FF 100%);
    background-size: 400% 400%;
    animation: gradientFlow 18s ease infinite;
}

@keyframes gradientFlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ---------- Fade-in for main block ---------- */
.main .block-container {
    animation: fadeInUp 0.7s ease both;
    padding-top: 1.5rem;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------- Header banner ---------- */
.hero-banner {
    background: linear-gradient(120deg, var(--primary-blue), var(--sky-blue), var(--deep-blue));
    background-size: 200% 200%;
    animation: gradientFlow 8s ease infinite;
    border-radius: 20px;
    padding: 2.1rem 2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
    position: relative;
    overflow: hidden;
}

.hero-banner::after{
    content: "";
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 60%);
    animation: shimmer 6s linear infinite;
}

@keyframes shimmer {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.hero-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.1rem;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 2;
}

.hero-subtitle {
    font-family: 'Inter', sans-serif;
    color: rgba(255,255,255,0.9);
    font-size: 1rem;
    margin-top: 0.4rem;
    position: relative;
    z-index: 2;
}

/* ---------- Pulsing status dot ---------- */
.status-dot {
    display:inline-block;
    width:10px; height:10px;
    border-radius:50%;
    background:#22C55E;
    margin-right:8px;
    box-shadow: 0 0 0 rgba(34,197,94,0.6);
    animation: pulse 1.8s infinite;
}

@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(34,197,94,0.55); }
    70%  { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F5F9FF 0%, #EAF2FF 100%);
    border-right: 1px solid #D6E4FF;
}

section[data-testid="stSidebar"] .block-container{
    animation: fadeInUp 0.8s ease both;
}

/* ---------- Cards ---------- */
.info-card {
    background: var(--white);
    border: 1px solid #DCE9FF;
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 4px 14px rgba(30, 64, 175, 0.06);
    transition: all 0.35s cubic-bezier(.2,.8,.2,1);
}

.info-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.16);
    border-color: var(--sky-blue);
}

.info-card h4{
    margin:0 0 0.3rem 0;
    color: var(--deep-blue);
    font-family:'Poppins',sans-serif;
    font-size:0.95rem;
}

.info-card p{
    margin:0;
    color: var(--slate-light);
    font-size:0.85rem;
}

/* ---------- Category badge ---------- */
.badge {
    display:inline-block;
    padding: 0.28rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family:'Inter',sans-serif;
    letter-spacing:0.3px;
    animation: badgePop 0.4s ease;
}

@keyframes badgePop {
    from { transform: scale(0.7); opacity:0; }
    to   { transform: scale(1); opacity:1; }
}

.badge-academic { background:#DBEAFE; color:#1D4ED8; }
.badge-fee      { background:#E0F2FE; color:#0369A1; }
.badge-general  { background:#EDE9FE; color:#6D28D9; }

/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"] {
    animation: fadeInUp 0.45s ease both;
    border-radius: 16px !important;
}

/* ---------- Buttons ---------- */
.stButton>button {
    background: linear-gradient(120deg, var(--primary-blue), var(--deep-blue));
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(37,99,235,0.25);
}

.stButton>button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 20px rgba(37,99,235,0.35);
    filter: brightness(1.07);
}

.stButton>button:active {
    transform: translateY(0) scale(0.98);
}

/* ---------- Radio / selectbox ---------- */
div[role="radiogroup"] label {
    transition: all 0.25s ease;
    border-radius: 10px;
}

/* ---------- Divider glow ---------- */
.glow-divider {
    height: 2px;
    margin: 1.2rem 0;
    background: linear-gradient(90deg, transparent, var(--sky-blue), transparent);
    background-size: 200% 100%;
    animation: slideGlow 3s linear infinite;
}

@keyframes slideGlow {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ---------- Typing indicator ---------- */
.typing-dots span {
    display:inline-block;
    width:6px; height:6px;
    margin-right:3px;
    background: var(--primary-blue);
    border-radius:50%;
    animation: bounceDot 1.2s infinite ease-in-out;
}
.typing-dots span:nth-child(2){ animation-delay: 0.15s; }
.typing-dots span:nth-child(3){ animation-delay: 0.3s; }

@keyframes bounceDot {
    0%, 80%, 100% { transform: translateY(0); opacity:0.5; }
    40% { transform: translateY(-6px); opacity:1; }
}

/* ---------- Footer ---------- */
.footer-note {
    text-align:center;
    color: var(--slate-light);
    font-size: 0.78rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #E2ECFB;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #EFF6FF; }
::-webkit-scrollbar-thumb { background: #93C5FD; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary-blue); }

</style>
""", unsafe_allow_html=True)


# =========================================================================
# BACKEND — unchanged logic, wrapped for Streamlit caching
# =========================================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def build_retriever_from_bytes(file_hash: str, file_bytes: bytes):
    """
    Builds a FAISS retriever from raw PDF bytes.
    Cached on file_hash, so re-uploading the same file won't rebuild the index,
    but a different file (different hash) will.
    """
    embeddings = get_embeddings()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        document = loader.load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(
        model="mistral-small-2506",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.4,
    )


class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


def make_classifier_node(llm):
    def classifier_node(state: State) -> dict:
        """Look at the latest user message and decide which path to take."""
        last_message = state['messages'][-1].content

        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotion, course structure, summer training, or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {last_message}\n\n"
            "Return only one word: academic, fee, or general."
        )

        response = llm.invoke(prompt)
        category = response.content.strip().lower()

        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"

        return {"query_type": category}

    return classifier_node


def make_academic_rag_node(academic_retriever):
    def academic_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the academics handbook."""
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    return academic_rag_node


def make_fee_rag_node(fee_retriever):
    def fee_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the fee structure PDF."""
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    return fee_rag_node


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def make_response_node(llm):
    def response_node(state: State) -> dict:
        """Generates the final answer, personalized using the student's programme."""
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrieved_context"]

        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant talking to a {programme} student. "
                f"Answer this question using your own general knowledge:\n\n{query}"
            )
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                f"Use the following context from the official college documents to answer "
                f"the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a clear, friendly, and precise answer."
            )

        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    return response_node


def route_query(state: State):
    if state['query_type'] == 'academic':
        return "academic"
    elif state['query_type'] == 'fee':
        return "fee_rag"
    else:
        return "general"


@st.cache_resource(show_spinner=False)
def build_app(_academic_retriever, _fee_retriever, academic_hash: str, fee_hash: str):
    """
    Builds and compiles the LangGraph exactly as in the original script.
    academic_hash/fee_hash are included as cache keys (even though the
    retriever objects themselves are prefixed with '_' to skip hashing)
    so that a new pair of uploaded PDFs triggers a fresh graph build.
    """
    llm = get_llm()
    academic_retriever = _academic_retriever
    fee_retriever = _fee_retriever

    graph = StateGraph(State)

    graph.add_node("classifier", make_classifier_node(llm))
    graph.add_node("academic", make_academic_rag_node(academic_retriever))
    graph.add_node("fee_rag", make_fee_rag_node(fee_retriever))
    graph.add_node("general", general_node)
    graph.add_node("response", make_response_node(llm))

    graph.add_edge(START, "classifier")

    graph.add_conditional_edges("classifier", route_query)

    graph.add_edge("academic", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    return graph.compile()


# =========================================================================
# SESSION STATE
# =========================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of dicts: {role, content, category?}

if "programme" not in st.session_state:
    st.session_state.programme = None

if "app_ready" not in st.session_state:
    st.session_state.app_ready = False

if "academic_pdf" not in st.session_state:
    st.session_state.academic_pdf = None   # bytes

if "fee_pdf" not in st.session_state:
    st.session_state.fee_pdf = None        # bytes


# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; margin-bottom:1rem;">
            <div style="font-size:2.4rem;">🎓</div>
            <div style="font-family:'Poppins',sans-serif; font-weight:700; font-size:1.15rem; color:#1E40AF;">
                College Assistant
            </div>
            <div style="font-size:0.8rem; color:#64748B;">Academic &amp; Fee Support Bot</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 👤 Select Your Programme")
    programme_map = {
        "BCA": "BCA",
        "Bachelor's": "Bachelor's",
        "B.Com (H)": "B.com (H)",
    }
    selected_label = st.radio(
        "Programme",
        list(programme_map.keys()),
        label_visibility="collapsed",
    )
    st.session_state.programme = programme_map[selected_label]

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📎 Upload College Documents")

    academic_file = st.file_uploader(
        "Academics Handbook (PDF)",
        type=["pdf"],
        key="academic_uploader",
    )
    fee_file = st.file_uploader(
        "Fee Structure (PDF)",
        type=["pdf"],
        key="fee_uploader",
    )

    if academic_file is not None:
        st.session_state.academic_pdf = academic_file.getvalue()
    if fee_file is not None:
        st.session_state.fee_pdf = fee_file.getvalue()

    docs_ready = st.session_state.academic_pdf is not None and st.session_state.fee_pdf is not None

    if docs_ready:
        st.markdown(
            '<div style="color:#16A34A; font-size:0.82rem; font-weight:600;">'
            '✅ Both documents loaded — ready to chat!</div>',
            unsafe_allow_html=True,
        )
    else:
        missing = []
        if st.session_state.academic_pdf is None:
            missing.append("Academics Handbook")
        if st.session_state.fee_pdf is None:
            missing.append("Fee Structure")
        st.markdown(
            f'<div style="color:#B45309; font-size:0.82rem; font-weight:600;">'
            f'⏳ Waiting for: {", ".join(missing)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-card">
            <h4>📘 Academic Queries</h4>
            <p>Attendance, exams, grading, credits, promotion, course structure &amp; more.</p>
        </div>
        <div class="info-card">
            <h4>💰 Fee Queries</h4>
            <p>Tuition, payments, refunds, late charges, scholarships &amp; more.</p>
        </div>
        <div class="info-card">
            <h4>💬 General Chat</h4>
            <p>Greetings and casual conversation, answered directly.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("""
        <div class="footer-note">
            <span class="status-dot"></span>Powered by Mistral &amp; LangGraph
        </div>
    """, unsafe_allow_html=True)


# =========================================================================
# HERO BANNER
# =========================================================================
docs_ready = st.session_state.academic_pdf is not None and st.session_state.fee_pdf is not None

hero_subtitle = (
    f'Chatting as a <b>{st.session_state.programme}</b> student — ask about academics, fees, or anything else!'
    if docs_ready else
    'Upload the Academics Handbook and Fee Structure PDFs in the sidebar to get started.'
)

st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">🎓 College Assistant</div>
        <div class="hero-subtitle">
            <span class="status-dot"></span>
            {hero_subtitle}
        </div>
    </div>
""", unsafe_allow_html=True)


# =========================================================================
# CHAT DISPLAY
# =========================================================================
badge_map = {
    "academic": '<span class="badge badge-academic">📘 Academic</span>',
    "fee": '<span class="badge badge-fee">💰 Fee</span>',
    "general": '<span class="badge badge-general">💬 General</span>',
}

chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        if docs_ready:
            st.markdown("""
                <div class="info-card" style="text-align:center; padding:2rem;">
                    <h4 style="font-size:1.1rem;">👋 Welcome!</h4>
                    <p>Ask me anything about attendance, exams, fees, scholarships, or just say hi.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="info-card" style="text-align:center; padding:2rem;">
                    <h4 style="font-size:1.1rem;">📎 Upload documents to begin</h4>
                    <p>Use the sidebar to upload the <b>Academics Handbook</b> and <b>Fee Structure</b> PDFs.</p>
                </div>
            """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "human" else "assistant"
        avatar = "🧑‍🎓" if role == "user" else "🎓"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant" and msg.get("category"):
                st.markdown(badge_map.get(msg["category"], ""), unsafe_allow_html=True)
            st.markdown(msg["content"])


# =========================================================================
# CHAT INPUT — runs the exact same LangGraph invocation as the CLI script
# =========================================================================
user_query = st.chat_input(
    "Type your question here..." if docs_ready else "Upload both PDFs in the sidebar to start chatting...",
    disabled=not docs_ready,
)

if user_query and docs_ready:
    st.session_state.chat_history.append({"role": "human", "content": user_query})

    with chat_container:
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🎓"):
            placeholder = st.empty()
            placeholder.markdown("""
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            """, unsafe_allow_html=True)

            try:
                academic_bytes = st.session_state.academic_pdf
                fee_bytes = st.session_state.fee_pdf
                academic_hash = _hash_bytes(academic_bytes)
                fee_hash = _hash_bytes(fee_bytes)

                academic_retriever = build_retriever_from_bytes(academic_hash, academic_bytes)
                fee_retriever = build_retriever_from_bytes(fee_hash, fee_bytes)

                compiled_app = build_app(academic_retriever, fee_retriever, academic_hash, fee_hash)

                result = compiled_app.invoke({
                    "programme": st.session_state.programme,
                    "messages": [("human", user_query)]
                })

                answer = result["messages"][-1].content
                category = result.get("query_type", "general")

                placeholder.empty()
                st.markdown(badge_map.get(category, ""), unsafe_allow_html=True)
                st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": answer,
                    "category": category,
                })

            except Exception as e:
                placeholder.empty()
                st.error(f"Something went wrong while generating the response: {e}")

    st.rerun()


# =========================================================================
# FOOTER
# =========================================================================
st.markdown("""
    <div class="footer-note">
        By Aayush Sharma
    </div>
""", unsafe_allow_html=True)