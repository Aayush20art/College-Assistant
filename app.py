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

st.set_page_config(page_title="College Assistant", page_icon="🎓", layout="wide")

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
    "academic": "Academic Handbook",
    "fee": "Fee Structure",
    "general": "General Knowledge",
}

SUGGESTION_GROUPS = [
    ("Academics", [
        ("Attendance rules", "What is the minimum attendance required to sit for exams?"),
        ("Grading & CGPA", "How are grades and CGPA calculated?"),
        ("Credits to graduate", "How many credits do I need to complete my degree?"),
    ]),
    ("Fees", [
        ("Tuition fees", "What is the tuition fee structure for my programme?"),
        ("Refund policy", "What is the fee refund policy?"),
        ("Payment deadlines", "What are the fee payment deadlines and late fee rules?"),
    ]),
    ("General", [
        ("Semester tips", "Give me some tips to manage my semester effectively."),
        ("Exam preparation", "How should I prepare for my upcoming exams?"),
        ("Time management", "How can I balance studies and extracurriculars?"),
    ]),
]


FOLLOW_UPS = {
    "academic": [
        ("Exam eligibility", "What are the eligibility conditions to appear in exams?"),
        ("Re-appear rules", "What are the rules for re-appear or backlog exams?"),
    ],
    "fee": [
        ("Installment options", "Are there installment or scholarship options for fees?"),
        ("Late fee charges", "What are the late fee charges?"),
    ],
    "general": [
        ("Weekly study plan", "Can you share a weekly study routine I can follow?"),
        ("Ask about fees", "What is the tuition fee structure for my programme?"),
    ],
}

STYLE_NOTES = {
    "Concise": "\n\nKeep the answer brief: 2-4 sentences or a short list.",
    "Balanced": "",
    "Detailed": "\n\nGive a thorough, well-structured answer with relevant specifics and steps.",
}


def route_badge(route: str, secs=None) -> str:
    label = ROUTES.get(route, ROUTES["general"])
    timing = f'<span class="secs">&middot; {secs}s</span>' if secs else ""
    return html(f'<div class="route-badge route-{route}"><i class="pulse-dot"></i>Answered via {label}{timing}</div>')


def status_pill(label: str, ok: bool) -> str:
    cls = "ok" if ok else "wait"
    text = "Ready" if ok else "Pending"
    return html(f"""
        <div class="status {cls}">
            <i class="pulse-dot"></i><span class="status-label">{label}</span><span class="status-text">{text}</span>
        </div>
    """)


def stepper(a_ok: bool, f_ok: bool) -> str:
    def state(done, active):
        return "done" if done else ("active" if active else "wait")

    items = [
        (state(a_ok, not a_ok), "Academics Handbook", "Upload the PDF from the sidebar", "1"),
        (state(f_ok, a_ok and not f_ok), "Fee Structure", "Upload the fee PDF as well", "2"),
        ("wait", "Start asking", "Get answers instantly", "3"),
    ]
    parts = []
    for i, (st_, title, sub, num) in enumerate(items):
        badge = "&#10003;" if st_ == "done" else num
        parts.append(f"""
            <div class="step {st_}" style="--d:{0.05 + i * 0.12}s">
                <div class="step-badge">{badge}</div>
                <div class="step-body"><div class="step-title">{title}</div><div class="step-sub">{sub}</div></div>
            </div>
        """)
        if i < len(items) - 1:
            on = "on" if st_ == "done" else ""
            parts.append(f'<div class="step-line {on}"></div>')
    return html('<div class="stepper">' + "".join(parts) + "</div>")


# ---------------------------------------------------------------------------
# Theme: clean white + orange
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --o1: #ea580c; --o2: #f97316; --o3: #fb923c; --o4: #fed7aa; --o5: #fff7ed;
        --ink: #1f2937; --muted: #6b7280; --line: #e5e7eb; --soft: #fafafa;
        --grad: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
    }

    /* ---------- Base ---------- */
    .stApp {
        background: radial-gradient(900px 420px at 90% -8%, #fff1e6 0%, rgba(255,255,255,0) 65%), #ffffff !important;
        color: var(--ink) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] td,
    .stApp [data-testid="stMarkdownContainer"] th { color: var(--ink); }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: var(--ink) !important; font-family: 'Inter', sans-serif !important; }
    .stApp a { color: var(--o1); }
    .stApp code { background: var(--o5) !important; color: #c2410c !important; border-radius: 6px; }
    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.8rem !important; max-width: 1020px; }
    ::selection { background: var(--o4); color: var(--ink); }

    /* ---------- Keyframes (kept subtle) ---------- */
    @keyframes fadeUp  { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeIn  { from { opacity: 0; } to { opacity: 1; } }
    @keyframes growX   { from { transform: scaleX(0); } to { transform: scaleX(1); } }
    @keyframes dotBeat { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.55); opacity: 0.5; } }
    @keyframes bounceDot { 0%,80%,100% { transform: translateY(0); opacity: 0.4; } 40% { transform: translateY(-6px); opacity: 1; } }
    @keyframes sweep   { 0% { transform: translateX(-110%); } 100% { transform: translateX(300%); } }
    @keyframes lineFlow { 0% { background-position: 0% 0%; } 100% { background-position: 200% 0%; } }

    /* ---------- Hero ---------- */
    .hero { padding: 1.6rem 0 1.8rem; animation: fadeUp 0.7s ease both; border-bottom: 1px solid var(--line); margin-bottom: 1.8rem; }
    .eyebrow {
        display: flex; align-items: center; gap: 0.7rem; font-size: 0.74rem; font-weight: 700;
        letter-spacing: 0.16em; text-transform: uppercase; color: var(--o1); margin-bottom: 1rem;
    }
    .eyebrow::before { content: ""; width: 30px; height: 2px; background: var(--o1); border-radius: 2px; transform-origin: left; animation: growX 0.9s ease both 0.2s; }
    .hero-title {
        font-family: 'Playfair Display', serif; font-weight: 700; font-size: 3.1rem; line-height: 1.12;
        letter-spacing: -0.5px; color: var(--ink); max-width: 760px;
    }
    .accent { position: relative; color: var(--o1); white-space: nowrap; }
    .accent::after {
        content: ""; position: absolute; left: 0; right: 0; bottom: 0.02em; height: 0.12em; border-radius: 4px;
        background: var(--o4); z-index: -1; transform-origin: left; animation: growX 1s ease both 0.5s;
    }
    .hero-desc { color: var(--muted); font-size: 1.02rem; line-height: 1.7; margin: 1rem 0 1.2rem; max-width: 600px; }
    .meta { display: flex; flex-wrap: wrap; gap: 0.4rem 1.4rem; font-size: 0.82rem; color: var(--muted); font-weight: 500; }
    .meta span { display: inline-flex; align-items: center; gap: 0.5rem; }
    .meta span::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--o2); }
    .hero.compact { padding: 0.6rem 0 1.2rem; margin-bottom: 1.2rem; }
    .hero.compact .hero-title { font-size: 2rem; }
    .hero.compact .hero-desc, .hero.compact .meta { display: none; }
    .hero.compact .eyebrow { margin-bottom: 0.6rem; }

    /* ---------- Section titles ---------- */
    .section-title {
        display: flex; align-items: center; gap: 0.8rem; font-size: 0.74rem; font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin: 0.2rem 0 1rem;
    }
    .section-title::after { content: ""; flex: 1; height: 1px; background: var(--line); }

    /* ---------- Stepper ---------- */
    .stepper { display: flex; align-items: center; margin-bottom: 1.4rem; }
    .step {
        flex: 1; display: flex; align-items: center; gap: 0.9rem; padding: 1.1rem 1.2rem; border-radius: 14px;
        background: #fff; border: 1px solid var(--line);
        animation: fadeUp 0.6s ease both; animation-delay: var(--d, 0s); transition: all 0.25s ease;
    }
    .step:hover { transform: translateY(-3px); box-shadow: 0 10px 26px rgba(17,24,39,0.07); }
    .step-badge {
        flex-shrink: 0; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.88rem; color: var(--muted); background: var(--soft); border: 1.5px solid var(--line);
    }
    .step-title { font-weight: 600; color: var(--ink); font-size: 0.95rem; }
    .step-sub { color: var(--muted); font-size: 0.8rem; margin-top: 0.1rem; }
    .step.active { border-color: var(--o3); box-shadow: 0 6px 22px rgba(249,115,22,0.14); }
    .step.active .step-badge { background: var(--grad); color: #fff; border-color: transparent; }
    .step.done { background: var(--o5); border-color: var(--o4); }
    .step.done .step-badge { background: var(--o1); color: #fff; border-color: transparent; }
    .step.wait { opacity: 0.65; }
    .step-line { flex: 0 0 28px; height: 2px; background: var(--line); }
    .step-line.on { background: linear-gradient(90deg, var(--o2), var(--o4), var(--o2)); background-size: 200% 100%; animation: lineFlow 2.4s linear infinite; }

    /* ---------- Ready banner & categories ---------- */
    .ready-banner {
        display: inline-flex; align-items: center; gap: 0.6rem; padding: 0.45rem 1rem; margin-bottom: 1.2rem;
        border-radius: 999px; background: var(--o5); border: 1px solid var(--o4);
        color: #9a3412; font-weight: 600; font-size: 0.82rem; animation: fadeUp 0.5s ease both;
    }
    .cat-head {
        display: flex; align-items: center; gap: 0.5rem; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: var(--muted); margin-bottom: 0.55rem;
    }
    .cat-head::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--o2); }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid var(--line) !important; }
    section[data-testid="stSidebar"] > div { animation: fadeIn 0.6s ease both; }
    .side-logo { display: flex; align-items: center; gap: 0.75rem; padding: 0.2rem 0 1.1rem; margin-bottom: 0.3rem; border-bottom: 1px solid var(--line); }
    .side-logo .mark {
        width: 40px; height: 40px; border-radius: 11px; display: flex; align-items: center; justify-content: center;
        background: var(--grad); color: #fff; font-weight: 700; font-size: 0.95rem; letter-spacing: 0.04em;
        box-shadow: 0 6px 16px rgba(234,88,12,0.3);
    }
    .side-logo .name { font-weight: 700; color: var(--ink); line-height: 1.15; font-size: 1rem; }
    .side-logo .sub { font-size: 0.72rem; color: var(--muted); }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-size: 0.72rem !important; letter-spacing: 0.14em; text-transform: uppercase; color: var(--o1) !important; font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: var(--ink) !important; font-weight: 500; }

    .status {
        display: flex; align-items: center; gap: 0.6rem; padding: 0.55rem 0.8rem; margin-bottom: 0.45rem;
        border-radius: 10px; font-size: 0.82rem; background: #fff; border: 1px solid var(--line); transition: all 0.3s ease;
    }
    .status .status-label { flex: 1; color: var(--ink); font-weight: 500; }
    .status .status-text { font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
    .status.ok { border-color: var(--o4); background: var(--o5); }
    .status.ok .status-text { color: var(--o1); }
    .pulse-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
    .status.ok .pulse-dot { background: var(--o2); animation: dotBeat 1.8s ease-in-out infinite; }
    .status.wait .pulse-dot { background: #d1d5db; }

    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin: 0.4rem 0 0.9rem; }
    .stat { text-align: center; padding: 0.7rem 0.2rem; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); transition: transform 0.25s ease; }
    .stat:hover { transform: translateY(-3px); }
    .stat .n { font-size: 1.35rem; font-weight: 700; color: var(--o1); }
    .stat .l { font-size: 0.62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }

    /* ---------- Uploader ---------- */
    [data-testid="stFileUploader"] section {
        background: var(--soft) !important; border: 1.5px dashed #d1d5db !important; border-radius: 12px !important; transition: all 0.25s ease !important;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--o2) !important; background: var(--o5) !important; }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploaderFile"] { color: var(--muted) !important; }
    [data-testid="stFileUploaderFileName"] { color: var(--ink) !important; font-weight: 600; }
    [data-testid="stFileUploaderDropzone"] button {
        background: #fff !important; color: var(--o1) !important; border: 1px solid var(--o4) !important; border-radius: 8px !important; font-weight: 600 !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        width: 100%; background: var(--grad) !important; color: #fff !important; border: none !important; border-radius: 10px !important;
        font-weight: 600 !important; padding: 0.55rem 1.1rem !important;
        box-shadow: 0 4px 12px rgba(234,88,12,0.22) !important; transition: all 0.25s ease !important;
    }
    .stButton > button *, .stDownloadButton > button * { color: inherit !important; }
    .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(234,88,12,0.32) !important; }
    .stButton > button:active { transform: scale(0.98); }
    .stDownloadButton > button:disabled, .stButton > button:disabled { opacity: 0.4; box-shadow: none !important; }

    /* Suggestion chips (main-area buttons) */
    [data-testid="stMain"] .stButton > button, .main .stButton > button {
        background: #fff !important; color: var(--ink) !important; border: 1px solid var(--line) !important;
        box-shadow: none !important; justify-content: flex-start; text-align: left; font-weight: 500 !important; margin-bottom: 0.1rem;
    }
    [data-testid="stMain"] .stButton > button:hover, .main .stButton > button:hover {
        background: var(--o5) !important; color: #c2410c !important; border-color: var(--o3) !important;
        box-shadow: 0 6px 16px rgba(249,115,22,0.14) !important;
    }

    /* ---------- Inputs ---------- */
    [data-baseweb="select"] > div { background: #fff !important; border: 1px solid var(--line) !important; border-radius: 10px !important; color: var(--ink) !important; }
    [data-baseweb="select"] * { color: var(--ink) !important; }
    [data-baseweb="select"] > div:focus-within { border-color: var(--o2) !important; box-shadow: 0 0 0 3px rgba(249,115,22,0.18) !important; }
    [data-baseweb="popover"] ul, [data-baseweb="menu"] { background: #fff !important; }
    [data-baseweb="popover"] li { color: var(--ink) !important; }
    [data-baseweb="popover"] li:hover { background: var(--o5) !important; }
    [data-testid="stToggle"] label p, .stCheckbox label p { color: var(--ink) !important; }

    /* ---------- Chat input ---------- */
    [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent !important; }
    [data-testid="stChatInput"] {
        background: #fff !important; border: 1px solid var(--line) !important; border-radius: 14px !important;
        box-shadow: 0 6px 22px rgba(17,24,39,0.08); transition: all 0.25s ease;
    }
    [data-testid="stChatInput"]:focus-within { border-color: var(--o2) !important; box-shadow: 0 0 0 3px rgba(249,115,22,0.18), 0 8px 26px rgba(17,24,39,0.08) !important; }
    [data-testid="stChatInput"] textarea { color: var(--ink) !important; background: transparent !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #9ca3af !important; }
    [data-testid="stChatInput"] button { background: var(--grad) !important; border-radius: 10px !important; }
    [data-testid="stChatInput"] button svg { color: #fff !important; fill: #fff !important; }

    /* ---------- Chat bubbles ---------- */
    [data-testid="stChatMessage"] {
        border-radius: 16px !important; padding: 1rem 1.25rem !important; margin-bottom: 0.9rem !important;
        background: #fff !important; border: 1px solid var(--line) !important; box-shadow: 0 2px 10px rgba(17,24,39,0.04);
        margin-right: 6%; animation: fadeUp 0.45s ease both; transition: box-shadow 0.25s ease;
    }
    [data-testid="stChatMessage"]:hover { box-shadow: 0 8px 22px rgba(17,24,39,0.08); }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse; margin-right: 0; margin-left: 14%;
        background: var(--grad) !important; border: none !important; box-shadow: 0 8px 22px rgba(234,88,12,0.25);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) *,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) * { color: #fff !important; }
    [data-testid="stChatMessageAvatarUser"], [data-testid="chatAvatarIcon-user"] { background: #1f2937 !important; color: #fff !important; }
    [data-testid="stChatMessageAvatarAssistant"], [data-testid="chatAvatarIcon-assistant"] { background: var(--o1) !important; color: #fff !important; }

    /* Route badges */
    .route-badge {
        display: inline-flex; align-items: center; gap: 0.5rem; margin-top: 0.7rem; font-size: 0.7rem; font-weight: 600;
        padding: 0.25rem 0.7rem; border-radius: 999px; border: 1px solid; animation: fadeIn 0.7s ease both;
    }
    .route-badge .pulse-dot { width: 6px; height: 6px; }
    .route-academic { color: #c2410c; border-color: var(--o4); background: var(--o5); }
    .route-academic .pulse-dot { background: var(--o2); }
    .route-fee { color: #92400e; border-color: #fde68a; background: #fffbeb; }
    .route-fee .pulse-dot { background: #f59e0b; }
    .route-general { color: #475569; border-color: #e2e8f0; background: #f8fafc; }
    .route-general .pulse-dot { background: #94a3b8; }

    /* Thinking indicator */
    .thinking { display: flex; align-items: center; gap: 0.8rem; color: var(--muted); font-size: 0.88rem; }
    .thinking .bars span { display: inline-block; width: 8px; height: 8px; margin-right: 4px; border-radius: 50%; background: var(--o2); animation: bounceDot 1.2s ease-in-out infinite; }
    .thinking .bars span:nth-child(2) { animation-delay: 0.15s; }
    .thinking .bars span:nth-child(3) { animation-delay: 0.3s; }
    .thinking-line { position: relative; height: 3px; margin-top: 0.7rem; border-radius: 3px; background: var(--o5); overflow: hidden; }
    .thinking-line::after { content: ""; position: absolute; top: 0; bottom: 0; width: 30%; border-radius: 3px; background: var(--grad); animation: sweep 1.3s ease-in-out infinite; }

    /* ================= Extra motion layer (subtle, infinite) ================= */
    @keyframes glowDriftA { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-60px,50px) scale(1.15); } }
    @keyframes glowDriftB { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(70px,-40px) scale(1.1); } }
    @keyframes dotDrift   { from { background-position: 0 0; } to { background-position: 26px 26px; } }
    @keyframes caret      { 50% { border-color: transparent; } }
    @keyframes typeLoop   { 0% { width: 0; } 40%,62% { width: 43ch; } 100% { width: 0; } }
    @keyframes ringPing   { 0% { box-shadow: 0 0 0 0 rgba(249,115,22,0.45); } 80%,100% { box-shadow: 0 0 0 12px rgba(249,115,22,0); } }
    @keyframes inputBreath { 0%,100% { box-shadow: 0 6px 22px rgba(17,24,39,0.08), 0 0 0 0 rgba(249,115,22,0); } 50% { box-shadow: 0 6px 22px rgba(17,24,39,0.08), 0 0 0 5px rgba(249,115,22,0.12); } }
    @keyframes textShimmer { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
    @keyframes lineSlide  { 0% { background-position: 0% 0; } 100% { background-position: 300% 0; } }
    @keyframes markGlow   { 0%,100% { box-shadow: 0 6px 16px rgba(234,88,12,0.3), 0 0 0 0 rgba(249,115,22,0.35); } 50% { box-shadow: 0 6px 16px rgba(234,88,12,0.3), 0 0 0 9px rgba(249,115,22,0); } }

    /* Ambient background: two soft glows + faint drifting dot grid */
    .glow { position: fixed; border-radius: 50%; filter: blur(80px); pointer-events: none; z-index: 0; opacity: 0.22; }
    .glow.g1 { width: 420px; height: 420px; background: #fdba74; top: -140px; right: -80px; animation: glowDriftA 22s ease-in-out infinite; }
    .glow.g2 { width: 340px; height: 340px; background: #fed7aa; bottom: -120px; left: 18%; animation: glowDriftB 26s ease-in-out infinite; }
    .dotgrid {
        position: fixed; inset: 0; pointer-events: none; z-index: 0;
        background-image: radial-gradient(rgba(234,88,12,0.12) 1px, transparent 1px); background-size: 26px 26px;
        animation: dotDrift 12s linear infinite;
        mask-image: radial-gradient(ellipse at 85% 0%, #000 0%, transparent 60%);
        -webkit-mask-image: radial-gradient(ellipse at 85% 0%, #000 0%, transparent 60%);
    }

    /* Hero: sliding accent line + typewriter loop + shimmering underline */
    .hero { border-bottom: 0; position: relative; }
    .hero::after {
        content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px; border-radius: 2px;
        background: linear-gradient(90deg, var(--line) 0%, var(--o3) 25%, var(--line) 50%, var(--o3) 75%, var(--line) 100%);
        background-size: 300% 100%; animation: lineSlide 7s linear infinite;
    }
    .accent::after {
        background: linear-gradient(90deg, #fed7aa, #fdba74, #fed7aa); background-size: 200% 100%;
        animation: growX 1s ease both 0.5s, textShimmer 3s linear infinite 1.5s;
    }
    .typeline { margin-top: 0.9rem; }
    .typewriter {
        display: inline-block; overflow: hidden; white-space: nowrap; max-width: 100%; vertical-align: bottom;
        font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: #c2410c;
        border-right: 2px solid var(--o2); width: 43ch;
        animation: typeLoop 10s steps(43) infinite, caret 0.8s step-end infinite;
    }

    /* Small living details */
    .side-logo .mark { animation: markGlow 3.2s ease-in-out infinite; }
    .step.active .step-badge { animation: ringPing 2s ease-out infinite; }
    .ready-banner {
        background: linear-gradient(90deg, var(--o5), #ffedd5, var(--o5)); background-size: 200% 100%;
        animation: fadeUp 0.5s ease both, textShimmer 6s linear infinite;
    }
    [data-testid="stChatInput"] { animation: inputBreath 4.5s ease-in-out infinite; }
    .thinking em {
        font-style: normal; font-weight: 500;
        background: linear-gradient(90deg, #6b7280 0%, #ea580c 50%, #6b7280 100%); background-size: 200% auto;
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        animation: textShimmer 2s linear infinite;
    }
    .route-badge .secs { opacity: 0.7; font-weight: 500; }
    .avg { text-align: center; font-size: 0.74rem; color: var(--muted); margin: -0.35rem 0 0.8rem; }
    .avg b { color: var(--o1); }

    /* Suggestion / follow-up chips: arrow slides in on hover */
    [data-testid="stMain"] .stButton { animation: fadeUp 0.5s ease both; }
    [data-testid="stMain"] .stButton > button { position: relative; padding-right: 2rem !important; }
    [data-testid="stMain"] .stButton > button::after {
        content: "\2192"; position: absolute; right: 14px; top: 50%; margin-top: -0.7em; line-height: 1.4;
        opacity: 0; transform: translateX(-8px); transition: all 0.25s ease; color: var(--o1);
    }
    [data-testid="stMain"] .stButton > button:hover::after { opacity: 1; transform: translateX(0); }

    hr { border-top: 1px solid var(--line) !important; }
    [data-testid="stAlert"] { border-radius: 12px; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #e5e7eb; border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--o3); }

    @media (max-width: 760px) {
        .hero-title { font-size: 2.1rem; }
        .stepper { flex-direction: column; align-items: stretch; gap: 0.7rem; }
        .step-line { display: none; }
        [data-testid="stChatMessage"] { margin-right: 0; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { margin-left: 0; }
    }
</style>
<div class="glow g1"></div><div class="glow g2"></div><div class="dotgrid"></div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend (unchanged logic)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI and reading documents...")
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
        style: str
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
        style_note = STYLE_NOTES.get(state.get("style", "Balanced"), "")

        if context == "NO_RETRIEVAL_NEEDED":
            messages = [
                SystemMessage(content=f"You are a friendly college assistant talking to a {programme} student. Answer using your general knowledge." + style_note),
                HumanMessage(content=query)
            ]
        else:
            messages = [
                SystemMessage(content=(
                    f"You are a college assistant helping a {programme} student. "
                    f"Use the following official context to answer the question accurately. "
                    f"Highlight specific figures relevant to {programme} if they exist.\n\n"
                    f"Context:\n{context}"
                    + style_note
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
    st.toast("Chat cleared", icon="🧹")


def ask(q: str):
    """Callback: queue a question (suggestion chips / follow-ups)."""
    st.session_state["pending_query"] = q


def regenerate():
    """Callback: drop the last answer and ask the same question again."""
    chat = st.session_state["chat_messages"]
    if len(chat) >= 2 and chat[-1]["role"] == "assistant":
        last_answer = chat.pop()
        last_question = chat.pop()
        r = last_answer.get("route")
        if r:
            st.session_state["stats"][r] = max(0, st.session_state["stats"].get(r, 0) - 1)
        st.session_state["pending_query"] = last_question["content"]


def build_transcript(programme: str) -> str:
    lines = [
        "College Assistant - Chat Transcript",
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
            <div class="mark">CA</div>
            <div><div class="name">College Assistant</div><div class="sub">AI-powered student helper</div></div>
        </div>
    """), unsafe_allow_html=True)

    st.header("Documents")
    academic_file = st.file_uploader("Academics Handbook (PDF)", type="pdf", key="acad")
    fee_file = st.file_uploader("Fee Structure (PDF)", type="pdf", key="fee")

    st.markdown(status_pill("Academics Handbook", bool(academic_file)), unsafe_allow_html=True)
    st.markdown(status_pill("Fee Structure", bool(fee_file)), unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.header("Profile")
    student_programme = st.selectbox(
        "Which programme are you in?",
        ["BCA", "Bachelor's", "B.com (H)"]
    )

    st.header("Preferences")
    typing_effect = st.toggle("Typing animation", value=True)
    answer_style = st.selectbox("Answer style", ["Balanced", "Concise", "Detailed"])

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.header("Session")
    msgs = st.session_state["chat_messages"]
    total_q = sum(1 for m in msgs if m["role"] == "user")
    s = st.session_state["stats"]
    st.markdown(html(f"""
        <div class="stats">
            <div class="stat"><div class="n">{total_q}</div><div class="l">Asked</div></div>
            <div class="stat"><div class="n">{s['academic']}</div><div class="l">Academic</div></div>
            <div class="stat"><div class="n">{s['fee']}</div><div class="l">Fees</div></div>
        </div>
    """), unsafe_allow_html=True)

    times = [m["secs"] for m in msgs if m.get("secs")]
    if times:
        st.markdown(f'<div class="avg">Avg. response <b>{sum(times) / len(times):.1f}s</b></div>', unsafe_allow_html=True)

    st.download_button(
        "Download chat",
        data=build_transcript(student_programme),
        file_name="college_assistant_chat.txt",
        mime="text/plain",
        disabled=not msgs,
    )
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.button("Clear chat", on_click=clear_chat)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
ready = bool(academic_file and fee_file)
compact = "compact" if ready else ""

st.markdown(html(f"""
    <div class="hero {compact}">
        <div class="eyebrow">AI College Assistant</div>
        <div class="hero-title">Your college questions, <span class="accent">answered.</span></div>
        <div class="typeline"><span class="typewriter">Ask about attendance, fees, exams &amp; credits</span></div>
        <div class="hero-desc">Upload your official handbook and fee structure. Get precise, document-backed answers tailored to your programme in seconds.</div>
        <div class="meta">
            <span>Powered by Groq</span>
            <span>Retrieval-augmented answers</span>
            <span>Smart query routing</span>
        </div>
    </div>
"""), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if ready:
    app = setup_app(academic_file.getvalue(), fee_file.getvalue())

    if not st.session_state.get("ready_toast"):
        st.toast("Knowledge base ready", icon="✅")
        st.session_state["ready_toast"] = True

    st.markdown(html(f"""
        <div class="ready-banner">
            <i class="pulse-dot" style="background:#f97316; animation: dotBeat 1.8s ease-in-out infinite"></i>
            Knowledge base ready &nbsp;·&nbsp; Personalised for {student_programme}
        </div>
    """), unsafe_allow_html=True)

    # Replay history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("route"):
                st.markdown(route_badge(msg["route"], msg.get("secs")), unsafe_allow_html=True)

    chat = st.session_state["chat_messages"]
    if not chat:
        # Suggestions on an empty chat
        st.markdown('<div class="section-title">Suggested questions</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for ci, (name, items) in enumerate(SUGGESTION_GROUPS):
            with cols[ci]:
                st.markdown(f'<div class="cat-head">{name}</div>', unsafe_allow_html=True)
                for li, (label, q) in enumerate(items):
                    st.button(label, key=f"chip_{ci}_{li}", help=q, on_click=ask, args=(q,))
    elif chat[-1]["role"] == "assistant":
        # Follow-ups + regenerate after each answer
        st.markdown('<div class="section-title">Continue</div>', unsafe_allow_html=True)
        follow_ups = FOLLOW_UPS.get(chat[-1].get("route", "general"), FOLLOW_UPS["general"])
        cols = st.columns(len(follow_ups) + 1)
        for i, (label, q) in enumerate(follow_ups):
            with cols[i]:
                st.button(label, key=f"fu_{len(chat)}_{i}", help=q, on_click=ask, args=(q,))
        with cols[-1]:
            st.button("Regenerate", key=f"regen_{len(chat)}", help="Ask the same question again", on_click=regenerate)

    typed_query = st.chat_input("Ask a question about your college...")
    user_query = st.session_state.pop("pending_query", None) or typed_query

    if user_query:
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(html("""
                <div class="thinking">
                    <div class="bars"><span></span><span></span><span></span></div>
                    <em>Searching your documents…</em>
                </div>
                <div class="thinking-line"></div>
            """), unsafe_allow_html=True)

            try:
                t0 = time.time()
                result = app.invoke({
                    "programme": student_programme,
                    "style": answer_style,
                    "messages": [("human", user_query)]
                })
                secs = round(time.time() - t0, 1)
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
                st.markdown(route_badge(route, secs), unsafe_allow_html=True)

                st.session_state["stats"][route] = st.session_state["stats"].get(route, 0) + 1
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": answer, "route": route, "secs": secs}
                )
                st.rerun()  # refresh sidebar stats & download button

            except Exception as e:
                placeholder.empty()
                st.error(f"An error occurred: {str(e)}")

else:
    st.markdown('<div class="section-title">Get started</div>', unsafe_allow_html=True)
    st.markdown(stepper(bool(academic_file), bool(fee_file)), unsafe_allow_html=True)
