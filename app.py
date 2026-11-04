import os
import time
import tempfile
from datetime import datetime

import streamlit as st
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="College Assistant Hub", page_icon="🎓", layout="wide")

try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it in your App Settings.")
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def html(s: str) -> str:
    """Flatten indentation so Streamlit's markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in s.strip().splitlines())


ROUTES = {
    "academic": ("📚", "Academic Handbook"),
    "fee": ("💳", "Fee Structure"),
    "general": ("💬", "General Knowledge"),
}

SUGGESTIONS = [
    ("📚 Attendance rules", "What is the minimum attendance required to sit for exams?"),
    ("📝 Grading system", "How are grades and CGPA calculated?"),
    ("🎯 Credits to graduate", "How many credits do I need to complete my degree?"),
    ("💳 Tuition fees", "What is the tuition fee structure for my programme?"),
    ("🔁 Refund policy", "What is the fee refund policy?"),
    ("💡 Study tips", "Give me some tips to manage my semester effectively."),
]


def route_badge(route: str) -> str:
    icon, label = ROUTES.get(route, ROUTES["general"])
    return html(f'<div class="route-badge route-{route}"><i class="dot"></i>{icon} Answered via {label}</div>')


def status_pill(label: str, ok: bool) -> str:
    cls = "ok" if ok else "wait"
    text = "Ready" if ok else "Waiting"
    return html(f"""
        <div class="status {cls}">
            <i class="dot"></i><span class="status-label">{label}</span><span class="status-text">{text}</span>
        </div>
    """)


# ---------------------------------------------------------------------------
# Black & Medium Blue Theme + Animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --blue-300: #93c5fd;
        --blue-400: #60a5fa;
        --blue-500: #3b82f6;
        --blue-600: #2563eb;
        --blue-700: #1d4ed8;
        --blue-900: #1e3a8a;
        --bg: #000000;
        --panel: #070b14;
        --text: #e2e8f0;
        --muted: #94a3b8;
    }

    /* ---------- Base ---------- */
    .stApp {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp label {
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 2rem !important; max-width: 1100px; }
    ::selection { background: var(--blue-600); color: #fff; }

    /* ---------- Animated backdrop (grid + orbs) ---------- */
    .stApp::before {
        content: "";
        position: fixed; inset: 0;
        background-image:
            linear-gradient(rgba(59,130,246,0.07) 1px, transparent 1px),
            linear-gradient(90deg, rgba(59,130,246,0.07) 1px, transparent 1px);
        background-size: 48px 48px;
        animation: gridMove 14s linear infinite;
        pointer-events: none; z-index: 0;
        mask-image: radial-gradient(ellipse at center, #000 30%, transparent 80%);
        -webkit-mask-image: radial-gradient(ellipse at center, #000 30%, transparent 80%);
    }
    .orb {
        position: fixed; border-radius: 50%;
        filter: blur(90px); opacity: 0.28;
        pointer-events: none; z-index: 0;
        mix-blend-mode: screen;
    }
    .orb.o1 { width: 420px; height: 420px; background: #2563eb; top: -120px; left: -100px; animation: drift1 18s ease-in-out infinite; }
    .orb.o2 { width: 360px; height: 360px; background: #1d4ed8; bottom: -120px; right: -80px; animation: drift2 22s ease-in-out infinite; }
    .orb.o3 { width: 260px; height: 260px; background: #60a5fa; top: 45%; left: 55%; opacity: 0.14; animation: drift3 26s ease-in-out infinite; }

    /* ---------- Keyframes ---------- */
    @keyframes gridMove   { from { background-position: 0 0, 0 0; } to { background-position: 48px 48px, 48px 48px; } }
    @keyframes drift1     { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(120px,90px) scale(1.2); } }
    @keyframes drift2     { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-140px,-70px) scale(1.15); } }
    @keyframes drift3     { 0%,100% { transform: translate(0,0); } 33% { transform: translate(-120px,60px); } 66% { transform: translate(80px,-90px); } }
    @keyframes fadeUp     { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeIn     { from { opacity: 0; } to { opacity: 1; } }
    @keyframes shimmer    { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
    @keyframes floatY     { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    @keyframes pulseGlow  { 0%,100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.55); } 70% { box-shadow: 0 0 0 14px rgba(59,130,246,0); } }
    @keyframes dotPulse   { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.7); opacity: 0.45; } }
    @keyframes spin       { to { transform: rotate(360deg); } }
    @keyframes blink      { 50% { border-color: transparent; } }
    @keyframes typing     { 0% { width: 0; } 55%,100% { width: 37ch; } }
    @keyframes marquee    { from { transform: translateX(0); } to { transform: translateX(-50%); } }
    @keyframes bounceDot  { 0%,80%,100% { transform: translateY(0); opacity: 0.4; } 40% { transform: translateY(-7px); opacity: 1; } }
    @keyframes sweep      { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
    @keyframes borderFlow { 0% { background-position: 0% 50%; } 100% { background-position: 300% 50%; } }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        display: flex; align-items: center; gap: 1.6rem;
        padding: 1.8rem 2rem;
        background: linear-gradient(135deg, rgba(37,99,235,0.14), rgba(0,0,0,0.65));
        border: 1px solid rgba(59,130,246,0.35);
        border-radius: 20px;
        backdrop-filter: blur(8px);
        overflow: hidden;
        animation: fadeUp 0.7s ease both;
        margin-bottom: 1rem;
    }
    .hero::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, transparent, var(--blue-500), var(--blue-300), var(--blue-500), transparent);
        background-size: 300% 100%;
        animation: borderFlow 4s linear infinite;
    }
    .hero::after {
        content: ""; position: absolute; top: 0; bottom: 0; width: 40%;
        background: linear-gradient(100deg, transparent, rgba(96,165,250,0.08), transparent);
        animation: sweep 5s ease-in-out infinite;
    }
    .hero-icon {
        position: relative; z-index: 1;
        width: 86px; height: 86px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.6rem;
        border-radius: 22px;
        background: radial-gradient(circle at 30% 30%, #3b82f6, #1e3a8a);
        animation: floatY 4s ease-in-out infinite, pulseGlow 2.6s ease-out infinite;
    }
    .hero-icon::after {
        content: ""; position: absolute; inset: -9px; border-radius: 28px;
        border: 2px dashed rgba(96,165,250,0.5);
        animation: spin 14s linear infinite;
    }
    .hero-text { position: relative; z-index: 1; min-width: 0; }
    .hero-title {
        font-size: 2.5rem; font-weight: 800; letter-spacing: -1px; line-height: 1.15;
        background: linear-gradient(90deg, #60a5fa, #ffffff, #3b82f6, #93c5fd, #60a5fa);
        background-size: 200% auto;
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; color: transparent;
        animation: shimmer 5s linear infinite;
    }
    .typewriter {
        display: inline-block; overflow: hidden; white-space: nowrap; max-width: 100%;
        font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: var(--blue-300);
        border-right: 2px solid var(--blue-400);
        width: 37ch;
        animation: typing 6s steps(37) infinite alternate, blink 0.8s step-end infinite;
        margin-top: 0.35rem;
    }
    .hero-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.9rem; }
    .tag {
        font-size: 0.75rem; font-weight: 600; padding: 0.28rem 0.7rem;
        border-radius: 999px; color: var(--blue-300);
        background: rgba(37,99,235,0.14); border: 1px solid rgba(59,130,246,0.35);
        transition: all 0.25s ease;
    }
    .tag:hover { background: rgba(37,99,235,0.35); transform: translateY(-2px); color: #fff; }

    /* ---------- Marquee ticker ---------- */
    .ticker {
        overflow: hidden; border-top: 1px solid rgba(59,130,246,0.2); border-bottom: 1px solid rgba(59,130,246,0.2);
        padding: 0.55rem 0; margin: 0.6rem 0 1.6rem 0;
        mask-image: linear-gradient(90deg, transparent, #000 10%, #000 90%, transparent);
        -webkit-mask-image: linear-gradient(90deg, transparent, #000 10%, #000 90%, transparent);
    }
    .ticker-track { display: inline-flex; white-space: nowrap; width: max-content; animation: marquee 28s linear infinite; }
    .ticker:hover .ticker-track { animation-play-state: paused; }
    .ticker-item { color: var(--muted); font-size: 0.85rem; font-weight: 500; padding: 0 1.4rem; }
    .ticker-item b { color: var(--blue-400); margin-right: 0.4rem; }

    /* ---------- Cards ---------- */
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.4rem; }
    .card {
        position: relative; padding: 1.3rem 1.3rem 1.2rem;
        background: linear-gradient(160deg, rgba(10,16,32,0.95), rgba(0,0,0,0.9));
        border: 1px solid rgba(30,58,138,0.9);
        border-radius: 16px; overflow: hidden;
        animation: fadeUp 0.7s ease both; animation-delay: var(--d, 0s);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .card::before {
        content: ""; position: absolute; inset: 0;
        background: radial-gradient(circle at var(--x, 20%) -20%, rgba(59,130,246,0.25), transparent 60%);
        opacity: 0; transition: opacity 0.3s ease;
    }
    .card:hover { transform: translateY(-6px); border-color: var(--blue-500); box-shadow: 0 14px 34px rgba(37,99,235,0.28); }
    .card:hover::before { opacity: 1; }
    .card-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 34px; height: 34px; border-radius: 10px; margin-bottom: 0.7rem;
        font-weight: 700; color: #fff; background: linear-gradient(135deg, var(--blue-600), var(--blue-900));
        animation: pulseGlow 3s ease-out infinite;
    }
    .card-icon { font-size: 1.8rem; margin-bottom: 0.5rem; display: inline-block; animation: floatY 4s ease-in-out infinite; animation-delay: var(--d, 0s); }
    .card-title { font-weight: 700; color: #fff; margin-bottom: 0.25rem; font-size: 1.02rem; }
    .card-desc { color: var(--muted); font-size: 0.86rem; line-height: 1.5; }

    .section-title {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
        color: var(--blue-400); margin: 0.4rem 0 0.8rem; display: flex; align-items: center; gap: 0.6rem;
    }
    .section-title::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(59,130,246,0.5), transparent); }

    /* ---------- Ready banner ---------- */
    .ready-banner {
        display: flex; align-items: center; gap: 0.8rem;
        padding: 0.8rem 1.1rem; margin-bottom: 1.2rem;
        border-radius: 12px; border: 1px solid rgba(34,197,94,0.4);
        background: linear-gradient(90deg, rgba(34,197,94,0.12), rgba(0,0,0,0));
        color: #bbf7d0; font-weight: 500; font-size: 0.92rem;
        animation: fadeUp 0.6s ease both;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050914 0%, #000000 100%) !important;
        border-right: 1px solid var(--blue-900) !important;
    }
    section[data-testid="stSidebar"] > div { animation: fadeIn 0.8s ease both; }
    .side-logo { display: flex; align-items: center; gap: 0.7rem; padding: 0.3rem 0 1rem; }
    .side-logo .mark {
        width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, var(--blue-500), var(--blue-900)); font-size: 1.3rem;
        animation: pulseGlow 2.8s ease-out infinite;
    }
    .side-logo .name { font-weight: 800; color: #fff; line-height: 1.1; }
    .side-logo .sub { font-size: 0.72rem; color: var(--muted); }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-size: 0.78rem !important; letter-spacing: 0.16em; text-transform: uppercase; color: var(--blue-400) !important;
    }

    .status {
        display: flex; align-items: center; gap: 0.55rem; padding: 0.5rem 0.8rem; margin-bottom: 0.5rem;
        border-radius: 10px; font-size: 0.82rem; border: 1px solid var(--blue-900); background: rgba(10,16,32,0.8);
        transition: all 0.3s ease;
    }
    .status .status-label { flex: 1; color: var(--text); }
    .status .status-text { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
    .status.ok { border-color: rgba(34,197,94,0.45); }
    .status.ok .status-text { color: #4ade80; }
    .status.wait .status-text { color: #fbbf24; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; animation: dotPulse 1.6s ease-in-out infinite; }
    .status.ok .dot { background: #22c55e; }
    .status.wait .dot { background: #f59e0b; }

    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin: 0.4rem 0 0.8rem; }
    .stat {
        text-align: center; padding: 0.7rem 0.2rem; border-radius: 12px;
        background: rgba(10,16,32,0.9); border: 1px solid var(--blue-900);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .stat:hover { transform: translateY(-3px); border-color: var(--blue-500); }
    .stat .n { font-size: 1.4rem; font-weight: 800; color: #fff; }
    .stat .l { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }

    /* ---------- Uploader ---------- */
    [data-testid="stFileUploader"] section {
        background: rgba(10,16,32,0.7) !important;
        border: 2px dashed var(--blue-600) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--blue-400) !important;
        background: rgba(17,24,45,0.9) !important;
        box-shadow: 0 0 22px rgba(59,130,246,0.25);
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, var(--blue-600), var(--blue-700)) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important;
        font-weight: 600 !important; padding: 0.6rem 1.1rem !important; width: 100%;
        box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button::after, .stDownloadButton > button::after {
        content: ""; position: absolute; top: 0; left: -120%; width: 60%; height: 100%;
        background: linear-gradient(100deg, transparent, rgba(255,255,255,0.28), transparent);
        transition: left 0.6s ease;
    }
    .stButton > button:hover::after, .stDownloadButton > button:hover::after { left: 140%; }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 22px rgba(37,99,235,0.55) !important;
    }
    .stButton > button:active { transform: scale(0.97); }

    /* Suggestion chips (main-area buttons) */
    [data-testid="stMain"] .stButton > button, .main .stButton > button {
        background: rgba(10,16,32,0.85) !important;
        border: 1px solid var(--blue-900) !important;
        color: var(--blue-300) !important; box-shadow: none !important;
        text-align: left; justify-content: flex-start;
    }
    [data-testid="stMain"] .stButton > button:hover, .main .stButton > button:hover {
        border-color: var(--blue-500) !important; color: #fff !important;
        background: rgba(37,99,235,0.22) !important;
        box-shadow: 0 8px 22px rgba(37,99,235,0.3) !important;
    }

    /* ---------- Inputs ---------- */
    .stSelectbox > div > div, .stTextInput > div > div > input {
        background-color: rgba(10,16,32,0.9) !important;
        border: 1px solid var(--blue-900) !important;
        color: #fff !important; border-radius: 10px !important;
        transition: all 0.25s ease;
    }
    .stSelectbox > div > div:focus-within, .stTextInput > div > div > input:focus {
        border-color: var(--blue-500) !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.25) !important;
    }
    [data-testid="stToggle"] label, .stCheckbox label { color: var(--text) !important; }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"], .stChatInputContainer > div {
        background: rgba(10,16,32,0.95) !important;
        border: 1px solid var(--blue-900) !important;
        border-radius: 16px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stChatInput"]:focus-within, .stChatInputContainer > div:focus-within {
        border-color: var(--blue-500) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.22), 0 0 30px rgba(37,99,235,0.25) !important;
    }
    [data-testid="stChatInput"] textarea, .stChatInputContainer textarea { color: #fff !important; }
    [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent !important; }

    /* ---------- Chat bubbles ---------- */
    [data-testid="stChatMessage"] {
        border-radius: 16px !important; padding: 1.1rem 1.2rem !important; margin-bottom: 1rem !important;
        border: 1px solid var(--blue-900) !important;
        background: rgba(7,11,20,0.92) !important;
        animation: fadeUp 0.5s ease both;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    [data-testid="stChatMessage"]:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(37,99,235,0.18); }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(7,11,20,0.95)) !important;
        border-right: 4px solid var(--blue-500) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 4px solid var(--blue-600) !important;
    }
    [data-testid="stChatMessageAvatarUser"], [data-testid="chatAvatarIcon-user"] { background: var(--blue-500) !important; color: #fff !important; }
    [data-testid="stChatMessageAvatarAssistant"], [data-testid="chatAvatarIcon-assistant"] {
        background: var(--blue-700) !important; color: #fff !important; animation: pulseGlow 3s ease-out infinite;
    }

    /* Route badges */
    .route-badge {
        display: inline-flex; align-items: center; gap: 0.5rem; margin-top: 0.6rem;
        font-size: 0.72rem; font-weight: 600; padding: 0.25rem 0.7rem; border-radius: 999px;
        border: 1px solid; animation: fadeIn 0.8s ease both;
    }
    .route-badge .dot { width: 6px; height: 6px; }
    .route-academic { color: #93c5fd; border-color: rgba(59,130,246,0.5); background: rgba(37,99,235,0.14); }
    .route-academic .dot { background: #3b82f6; }
    .route-fee { color: #86efac; border-color: rgba(34,197,94,0.5); background: rgba(34,197,94,0.1); }
    .route-fee .dot { background: #22c55e; }
    .route-general { color: #c4b5fd; border-color: rgba(139,92,246,0.5); background: rgba(139,92,246,0.1); }
    .route-general .dot { background: #8b5cf6; }

    /* Thinking indicator */
    .thinking { display: flex; align-items: center; gap: 0.7rem; color: var(--blue-300); font-size: 0.9rem; }
    .thinking .bars span {
        display: inline-block; width: 9px; height: 9px; margin-right: 5px; border-radius: 50%;
        background: var(--blue-500); animation: bounceDot 1.2s ease-in-out infinite;
    }
    .thinking .bars span:nth-child(2) { animation-delay: 0.15s; }
    .thinking .bars span:nth-child(3) { animation-delay: 0.3s; }
    .thinking-line { position: relative; height: 3px; margin-top: 0.6rem; border-radius: 3px; background: rgba(59,130,246,0.15); overflow: hidden; }
    .thinking-line::after { content: ""; position: absolute; inset: 0; width: 40%; background: linear-gradient(90deg, transparent, var(--blue-400), transparent); animation: sweep 1.2s ease-in-out infinite; }

    hr { border-top: 1px solid var(--blue-900) !important; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #000; }
    ::-webkit-scrollbar-thumb { background: var(--blue-900); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--blue-600); }

    @media (max-width: 700px) {
        .hero { flex-direction: column; text-align: center; padding: 1.4rem 1rem; }
        .hero-title { font-size: 1.8rem; }
        .hero-tags { justify-content: center; }
    }
</style>

<div class="orb o1"></div>
<div class="orb o2"></div>
<div class="orb o3"></div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend (unchanged logic)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⚡ Initializing AI and reading documents...")
def setup_app(academic_bytes, fee_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_acad:
        tmp_acad.write(academic_bytes)
        acad_path = tmp_acad.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_fee:
        tmp_fee.write(fee_bytes)
        fee_path = tmp_fee.name

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def build_retriever(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        document = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(document)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    academic_retriever = build_retriever(acad_path)
    fee_retriever = build_retriever(fee_path)

    os.remove(acad_path)
    os.remove(fee_path)

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4)

    class State(TypedDict):
        programme: str
        messages: Annotated[list, add_messages]
        query_type: str
        retrieved_context: str

    def classifier_node(state: State) -> dict:
        last_message = state['messages'][-1].content

        messages = [
            SystemMessage(content=(
                "You are a routing AI. Classify the user query into exactly one category: "
                "'academic', 'fee', or 'general'.\n"
                "Use 'academic' for attendance, exams, grading, credits, courses, or degrees.\n"
                "Use 'fee' for tuition, payment, refunds, or money.\n"
                "Use 'general' for anything else.\n"
                "Return ONLY ONE WORD."
            )),
            HumanMessage(content=last_message)
        ]

        response = llm.invoke(messages)
        category = response.content.strip().lower()

        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"

        return {"query_type": category}

    def academic_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def fee_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def general_node(state: State) -> dict:
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State) -> dict:
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrieved_context"]

        if context == "NO_RETRIEVAL_NEEDED":
            messages = [
                SystemMessage(content=f"You are a friendly college assistant talking to a {programme} student. Answer using your general knowledge."),
                HumanMessage(content=query)
            ]
        else:
            messages = [
                SystemMessage(content=(
                    f"You are a college assistant helping a {programme} student. "
                    f"Use the following official context to answer the question accurately. "
                    f"Highlight specific figures relevant to {programme} if they exist.\n\n"
                    f"Context:\n{context}"
                )),
                HumanMessage(content=query)
            ]

        response = llm.invoke(messages)
        return {"messages": [("ai", response.content.strip())]}

    def route_query(state: State):
        if state['query_type'] == 'academic':
            return "academic"
        elif state['query_type'] == 'fee':
            return "fee_rag"
        else:
            return "general"

    graph = StateGraph(State)
    graph.add_node("classifier", classifier_node)
    graph.add_node("academic", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "stats" not in st.session_state:
    st.session_state["stats"] = {"academic": 0, "fee": 0, "general": 0}


def clear_chat():
    st.session_state["chat_messages"] = []
    st.session_state["stats"] = {"academic": 0, "fee": 0, "general": 0}


def build_transcript(programme: str) -> str:
    lines = [
        "College Assistant Hub - Chat Transcript",
        f"Programme: {programme}",
        f"Exported: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        "=" * 50,
        "",
    ]
    for m in st.session_state["chat_messages"]:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"[{who}]")
        lines.append(m["content"])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(html("""
        <div class="side-logo">
            <div class="mark">🎓</div>
            <div><div class="name">College Hub</div><div class="sub">AI Student Assistant</div></div>
        </div>
    """), unsafe_allow_html=True)

    st.header("📂 Document Control")
    academic_file = st.file_uploader("Academics Handbook (PDF)", type="pdf", key="acad")
    fee_file = st.file_uploader("Fee Structure (PDF)", type="pdf", key="fee")

    st.markdown(status_pill("Academics Handbook", bool(academic_file)), unsafe_allow_html=True)
    st.markdown(status_pill("Fee Structure", bool(fee_file)), unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.header("🎓 Student Profile")
    student_programme = st.selectbox(
        "Which programme are you in?",
        ["BCA", "Bachelor's", "B.com (H)"]
    )

    st.header("⚙️ Experience")
    typing_effect = st.toggle("✨ Typing animation", value=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.header("📊 Session Stats")
    msgs = st.session_state["chat_messages"]
    total_q = sum(1 for m in msgs if m["role"] == "user")
    s = st.session_state["stats"]
    st.markdown(html(f"""
        <div class="stats">
            <div class="stat"><div class="n">{total_q}</div><div class="l">Questions</div></div>
            <div class="stat"><div class="n">{s['academic']}</div><div class="l">Academic</div></div>
            <div class="stat"><div class="n">{s['fee']}</div><div class="l">Fees</div></div>
        </div>
    """), unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download Chat",
        data=build_transcript(student_programme),
        file_name="college_assistant_chat.txt",
        mime="text/plain",
        disabled=not msgs,
    )
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.button("🗑️ Clear Chat", on_click=clear_chat)


# ---------------------------------------------------------------------------
# Hero + ticker (always visible)
# ---------------------------------------------------------------------------
st.markdown(html("""
    <div class="hero">
        <div class="hero-icon">🎓</div>
        <div class="hero-text">
            <div class="hero-title">College Assistant Hub</div>
            <div><span class="typewriter">Your smart guide to academics &amp; fees</span></div>
            <div class="hero-tags">
                <span class="tag">⚡ Groq Powered</span>
                <span class="tag">🧠 RAG Search</span>
                <span class="tag">🕸️ LangGraph Routing</span>
                <span class="tag">🔒 Official Docs Only</span>
            </div>
        </div>
    </div>
"""), unsafe_allow_html=True)

ticker_items = ["Attendance", "Exams", "Grading", "Credits", "Tuition Fees", "Refunds", "Degrees", "Courses", "Payments"]
ticker_html = "".join(f'<span class="ticker-item"><b>✦</b>{t}</span>' for t in ticker_items) * 2
st.markdown(f'<div class="ticker"><div class="ticker-track">{ticker_html}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if academic_file and fee_file:
    app = setup_app(academic_file.getvalue(), fee_file.getvalue())

    st.markdown(html(f"""
        <div class="ready-banner">
            <i class="dot" style="background:#22c55e"></i>
            Knowledge base ready &nbsp;•&nbsp; Personalised for <b>{student_programme}</b>
        </div>
    """), unsafe_allow_html=True)

    # Replay history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("route"):
                st.markdown(route_badge(msg["route"]), unsafe_allow_html=True)

    # Suggestion chips (only on an empty chat)
    clicked_query = None
    if not st.session_state["chat_messages"]:
        st.markdown('<div class="section-title">💡 Try asking</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (label, q) in enumerate(SUGGESTIONS):
            with cols[i % 3]:
                if st.button(label, key=f"chip_{i}", help=q):
                    clicked_query = q

    typed_query = st.chat_input("Ask about attendance, fees, exams, credits...")
    user_query = clicked_query or typed_query

    if user_query:
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(html("""
                <div class="thinking">
                    <div class="bars"><span></span><span></span><span></span></div>
                    <em>Routing your question &amp; searching documents…</em>
                </div>
                <div class="thinking-line"></div>
            """), unsafe_allow_html=True)

            try:
                result = app.invoke({
                    "programme": student_programme,
                    "messages": [("human", user_query)]
                })
                answer = result['messages'][-1].content
                route = result.get("query_type", "general")

                if typing_effect:
                    words = answer.split(" ")
                    delay = min(0.03, 2.5 / max(len(words), 1))
                    shown = ""
                    for w in words:
                        shown += w + " "
                        placeholder.markdown(shown + "▌")
                        time.sleep(delay)
                placeholder.markdown(answer)
                st.markdown(route_badge(route), unsafe_allow_html=True)

                st.session_state["stats"][route] = st.session_state["stats"].get(route, 0) + 1
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": answer, "route": route}
                )
                st.rerun()  # refresh sidebar stats & download button

            except Exception as e:
                placeholder.empty()
                st.error(f"An error occurred: {str(e)}")

else:
    st.markdown('<div class="section-title">🚀 Get started in 3 steps</div>', unsafe_allow_html=True)
    st.markdown(html("""
        <div class="grid">
            <div class="card" style="--d:0.05s">
                <div class="card-num">1</div>
                <div class="card-title">Upload Academics Handbook</div>
                <div class="card-desc">Drop your official handbook PDF in the sidebar. Attendance, exams &amp; grading rules live here.</div>
            </div>
            <div class="card" style="--d:0.2s">
                <div class="card-num">2</div>
                <div class="card-title">Upload Fee Structure</div>
                <div class="card-desc">Add the fee structure PDF so I can answer tuition, payment and refund questions.</div>
            </div>
            <div class="card" style="--d:0.35s">
                <div class="card-num">3</div>
                <div class="card-title">Ask Anything</div>
                <div class="card-desc">Pick your programme and start chatting. Answers are tailored to your course.</div>
            </div>
        </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="section-title">✨ What makes it smart</div>', unsafe_allow_html=True)
    st.markdown(html("""
        <div class="grid">
            <div class="card" style="--d:0.1s">
                <div class="card-icon" style="--d:0s">🧭</div>
                <div class="card-title">Smart Routing</div>
                <div class="card-desc">Every question is classified and sent to the right knowledge source automatically.</div>
            </div>
            <div class="card" style="--d:0.25s">
                <div class="card-icon" style="--d:0.4s">📖</div>
                <div class="card-title">Document-grounded</div>
                <div class="card-desc">Answers come from your official PDFs using vector search, not guesses.</div>
            </div>
            <div class="card" style="--d:0.4s">
                <div class="card-icon" style="--d:0.8s">🎯</div>
                <div class="card-title">Programme Aware</div>
                <div class="card-desc">BCA, Bachelor's or B.Com (H) — figures are highlighted for your course.</div>
            </div>
            <div class="card" style="--d:0.55s">
                <div class="card-icon" style="--d:1.2s">⚡</div>
                <div class="card-title">Lightning Fast</div>
                <div class="card-desc">Powered by Groq inference for near-instant responses.</div>
            </div>
        </div>
    """), unsafe_allow_html=True)

    st.info("👈 Upload both PDFs in the sidebar to initialize the assistant.")
