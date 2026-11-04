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

st.set_page_config(page_title="Campus Buddy", page_icon="🌸", layout="wide")

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

SUGGESTION_GROUPS = [
    ("📚", "Academics", [
        ("Attendance rules", "What is the minimum attendance required to sit for exams?"),
        ("Grading & CGPA", "How are grades and CGPA calculated?"),
        ("Credits to graduate", "How many credits do I need to complete my degree?"),
    ]),
    ("💳", "Fees", [
        ("Tuition fees", "What is the tuition fee structure for my programme?"),
        ("Refund policy", "What is the fee refund policy?"),
        ("Payment deadlines", "What are the fee payment deadlines and late fee rules?"),
    ]),
    ("💬", "Campus Life", [
        ("Semester tips", "Give me some tips to manage my semester effectively."),
        ("Exam prep", "How should I prepare for my upcoming exams?"),
        ("Time management", "How can I balance studies and extracurriculars?"),
    ]),
]


def route_badge(route: str) -> str:
    icon, label = ROUTES.get(route, ROUTES["general"])
    return html(f'<div class="route-badge route-{route}"><i class="pulse-dot"></i>{icon} Answered via {label}</div>')


def status_pill(label: str, ok: bool) -> str:
    cls = "ok" if ok else "wait"
    text = "Ready" if ok else "Waiting"
    return html(f"""
        <div class="status {cls}">
            <i class="pulse-dot"></i><span class="status-label">{label}</span><span class="status-text">{text}</span>
        </div>
    """)


def stepper(a_ok: bool, f_ok: bool) -> str:
    def state(done, active):
        return "done" if done else ("active" if active else "wait")

    s1 = state(a_ok, not a_ok)
    s2 = state(f_ok, a_ok and not f_ok)
    s3 = "wait"
    items = [
        (s1, "📘", "Academics Handbook", "Upload the PDF in the sidebar", "1"),
        (s2, "💳", "Fee Structure", "Upload the fee PDF too", "2"),
        (s3, "💬", "Start chatting", "Ask anything, instantly", "3"),
    ]
    parts = []
    for i, (st_, emoji, title, sub, num) in enumerate(items):
        badge = "✓" if st_ == "done" else num
        parts.append(f"""
            <div class="step {st_}" style="--d:{0.05 + i * 0.15}s">
                <div class="step-badge">{badge}</div>
                <div class="step-emoji">{emoji}</div>
                <div class="step-title">{title}</div>
                <div class="step-sub">{sub}</div>
            </div>
        """)
        if i < len(items) - 1:
            on = "on" if st_ == "done" else ""
            parts.append(f'<div class="step-line {on}"></div>')
    return html('<div class="stepper">' + "".join(parts) + "</div>")


# ---------------------------------------------------------------------------
# Light Theme: Lavender + Pink + White
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --p1: #8b5cf6; --p2: #a78bfa; --p3: #c4b5fd; --p4: #ede9fe;
        --k1: #ec4899; --k2: #f472b6; --k3: #fbcfe8; --k4: #fdf2f8;
        --ink: #3b2a5a; --muted: #7c6a9c; --line: #e9d5ff;
        --grad: linear-gradient(135deg, #8b5cf6 0%, #c026d3 55%, #ec4899 100%);
    }

    /* ---------- Base ---------- */
    .stApp {
        background: linear-gradient(135deg, #fdf4ff 0%, #f5f3ff 40%, #fff1f8 75%, #faf5ff 100%) !important;
        background-size: 220% 220% !important;
        animation: bgShift 22s ease-in-out infinite;
        color: var(--ink) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] td,
    .stApp [data-testid="stMarkdownContainer"] th { color: var(--ink); }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: var(--ink) !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
    .stApp a { color: var(--p1); }
    .stApp code { background: var(--p4) !important; color: #6d28d9 !important; border-radius: 6px; }
    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.6rem !important; max-width: 1080px; }
    ::selection { background: var(--k3); color: var(--ink); }

    /* ---------- Floating blobs & sparkles ---------- */
    .blob { position: fixed; border-radius: 50%; filter: blur(70px); pointer-events: none; z-index: 0; opacity: 0.55; }
    .blob.b1 { width: 380px; height: 380px; background: #ddd6fe; top: -90px; right: 8%; animation: blobA 20s ease-in-out infinite; }
    .blob.b2 { width: 340px; height: 340px; background: #fbcfe8; bottom: -90px; left: 12%; animation: blobB 24s ease-in-out infinite; }
    .blob.b3 { width: 260px; height: 260px; background: #e9d5ff; top: 40%; right: -60px; animation: blobC 28s ease-in-out infinite; }
    .spark { position: fixed; color: var(--p2); pointer-events: none; z-index: 0; animation: twinkle 3.5s ease-in-out infinite; }
    .spark.s1 { top: 12%; left: 6%; font-size: 1.3rem; }
    .spark.s2 { top: 28%; right: 5%; font-size: 1rem; color: var(--k2); animation-delay: 1s; }
    .spark.s3 { bottom: 18%; left: 4%; font-size: 1.1rem; color: var(--k2); animation-delay: 2s; }
    .spark.s4 { bottom: 10%; right: 9%; font-size: 1.4rem; animation-delay: 0.6s; }

    /* ---------- Keyframes ---------- */
    @keyframes bgShift  { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
    @keyframes blobA    { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-90px,80px) scale(1.2); } }
    @keyframes blobB    { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(110px,-70px) scale(1.15); } }
    @keyframes blobC    { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-80px,-100px); } }
    @keyframes twinkle  { 0%,100% { opacity: 0.25; transform: scale(0.8) rotate(0deg); } 50% { opacity: 1; transform: scale(1.3) rotate(25deg); } }
    @keyframes fadeUp   { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeIn   { from { opacity: 0; } to { opacity: 1; } }
    @keyframes gradMove { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
    @keyframes floatY   { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-9px); } }
    @keyframes spin     { to { transform: rotate(360deg); } }
    @keyframes spinRev  { to { transform: rotate(-360deg); } }
    @keyframes ping     { 0% { box-shadow: 0 0 0 0 rgba(139,92,246,0.5); } 80%,100% { box-shadow: 0 0 0 14px rgba(139,92,246,0); } }
    @keyframes dotBeat  { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.6); opacity: 0.5; } }
    @keyframes marquee  { from { transform: translateX(0); } to { transform: translateX(-50%); } }
    @keyframes rotateWords { 0%,18% { transform: translateY(0); } 25%,43% { transform: translateY(-1.25em); } 50%,68% { transform: translateY(-2.5em); } 75%,93% { transform: translateY(-3.75em); } 100% { transform: translateY(-5em); } }
    @keyframes bounceDot { 0%,80%,100% { transform: translateY(0); opacity: 0.45; } 40% { transform: translateY(-8px); opacity: 1; } }
    @keyframes sweep    { 0% { transform: translateX(-110%); } 100% { transform: translateX(260%); } }
    @keyframes wave     { 0%,60%,100% { transform: rotate(0); } 10%,30% { transform: rotate(16deg); } 20%,40% { transform: rotate(-10deg); } 50% { transform: rotate(8deg); } }
    @keyframes lineFlow { 0% { background-position: 0% 0%; } 100% { background-position: 200% 0%; } }

    /* ---------- Hero ---------- */
    .hero {
        position: relative; z-index: 1;
        display: grid; grid-template-columns: 1.35fr 1fr; align-items: center; gap: 1rem;
        padding: 2rem 2.4rem;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(255,255,255,0.9);
        border-radius: 32px;
        box-shadow: 0 20px 60px rgba(139,92,246,0.14), 0 2px 0 rgba(255,255,255,0.8) inset;
        backdrop-filter: blur(14px);
        animation: fadeUp 0.8s ease both;
        overflow: hidden;
    }
    .hero::after {
        content: ""; position: absolute; top: 0; bottom: 0; width: 28%;
        background: linear-gradient(100deg, transparent, rgba(255,255,255,0.7), transparent);
        animation: sweep 7s ease-in-out infinite; pointer-events: none;
    }
    .eyebrow {
        display: inline-flex; align-items: center; gap: 0.55rem;
        font-size: 0.78rem; font-weight: 700; color: #7e22ce;
        background: linear-gradient(90deg, var(--p4), var(--k4)); border: 1px solid var(--line);
        padding: 0.35rem 0.85rem; border-radius: 999px; margin-bottom: 1rem;
    }
    .wave { display: inline-block; transform-origin: 70% 70%; animation: wave 2.6s ease-in-out infinite; }
    .hero-title { font-size: 2.55rem; font-weight: 800; line-height: 1.15; letter-spacing: -1.2px; color: var(--ink); }
    .rotator { display: inline-block; height: 1.25em; overflow: hidden; vertical-align: bottom; }
    .rotator ul { list-style: none; margin: 0; padding: 0; animation: rotateWords 10s ease-in-out infinite; }
    .rotator li { height: 1.25em; line-height: 1.25em; }
    .grad {
        background: var(--grad); background-size: 200% auto;
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; color: transparent;
        animation: gradMove 5s linear infinite;
    }
    .hero-desc { color: var(--muted); font-size: 1rem; line-height: 1.65; margin: 0.9rem 0 1.1rem; max-width: 520px; }
    .pills { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .pill {
        font-size: 0.76rem; font-weight: 600; color: #6d28d9;
        background: #fff; border: 1px solid var(--line); padding: 0.32rem 0.75rem; border-radius: 999px;
        box-shadow: 0 2px 8px rgba(139,92,246,0.08); transition: all 0.25s ease;
    }
    .pill:hover { transform: translateY(-3px); background: var(--grad); color: #fff; border-color: transparent; }

    .hero-visual { display: flex; justify-content: center; align-items: center; }
    .orbit { position: relative; width: 250px; height: 250px; }
    .orbit .ring { position: absolute; inset: 0; border-radius: 50%; border: 2px dashed var(--p3); animation: spin 40s linear infinite; }
    .orbit .ring2 { position: absolute; inset: 34px; border-radius: 50%; border: 2px dotted var(--k3); animation: spinRev 30s linear infinite; }
    .orbit .core {
        position: absolute; inset: 78px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 2.6rem; background: var(--grad); box-shadow: 0 12px 34px rgba(192,38,211,0.4);
        animation: floatY 4s ease-in-out infinite, ping 2.8s ease-out infinite;
    }
    .orbit .arm { position: absolute; inset: 0; animation: spin 18s linear infinite; animation-delay: var(--dl); }
    .orbit .sat {
        position: absolute; top: -4px; left: 50%; margin-left: -25px; width: 50px; height: 50px; border-radius: 16px;
        display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
        background: #fff; box-shadow: 0 8px 22px rgba(139,92,246,0.25); border: 1px solid var(--line);
        animation: spinRev 18s linear infinite; animation-delay: var(--dl);
    }
    .hero.compact { grid-template-columns: 1fr; padding: 1.2rem 1.8rem; }
    .hero.compact .hero-visual, .hero.compact .hero-desc { display: none; }
    .hero.compact .hero-title { font-size: 1.8rem; }
    .hero.compact .eyebrow { margin-bottom: 0.5rem; }

    /* ---------- Ticker ---------- */
    .ticker {
        position: relative; z-index: 1; overflow: hidden; margin: 1.1rem 0 1.5rem;
        mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent);
        -webkit-mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent);
    }
    .ticker-track { display: inline-flex; width: max-content; gap: 0.7rem; animation: marquee 32s linear infinite; }
    .ticker:hover .ticker-track { animation-play-state: paused; }
    .tick {
        white-space: nowrap; font-size: 0.82rem; font-weight: 600; color: #7e22ce;
        background: rgba(255,255,255,0.85); border: 1px solid var(--line); padding: 0.4rem 1rem; border-radius: 999px;
    }
    .tick:nth-child(even) { color: #be185d; background: rgba(253,242,248,0.9); border-color: var(--k3); }

    /* ---------- Section titles ---------- */
    .section-title {
        position: relative; z-index: 1; display: flex; align-items: center; gap: 0.7rem;
        font-size: 0.8rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; color: #9333ea; margin: 0.4rem 0 0.9rem;
    }
    .section-title::after { content: ""; flex: 1; height: 2px; border-radius: 2px; background: linear-gradient(90deg, var(--p3), transparent); }

    /* ---------- Stepper ---------- */
    .stepper { position: relative; z-index: 1; display: flex; align-items: stretch; gap: 0; margin-bottom: 1.6rem; }
    .step {
        flex: 1; text-align: center; padding: 1.5rem 1rem 1.3rem; border-radius: 24px; position: relative;
        background: rgba(255,255,255,0.8); border: 1px solid var(--line);
        box-shadow: 0 10px 30px rgba(139,92,246,0.08);
        animation: fadeUp 0.7s ease both; animation-delay: var(--d, 0s);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .step:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(139,92,246,0.2); }
    .step-badge {
        position: absolute; top: -14px; left: 50%; margin-left: -15px; width: 30px; height: 30px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.85rem;
        background: #fff; color: var(--muted); border: 2px solid var(--line);
    }
    .step-emoji { font-size: 2rem; margin-bottom: 0.4rem; display: inline-block; animation: floatY 4s ease-in-out infinite; }
    .step-title { font-weight: 700; color: var(--ink); font-size: 1rem; }
    .step-sub { color: var(--muted); font-size: 0.82rem; margin-top: 0.2rem; }
    .step.active { border-color: var(--p2); background: #fff; }
    .step.active .step-badge { background: var(--grad); color: #fff; border-color: transparent; animation: ping 2s ease-out infinite; }
    .step.done { border-color: #f9a8d4; background: linear-gradient(160deg, #fff, var(--k4)); }
    .step.done .step-badge { background: var(--k1); color: #fff; border-color: transparent; }
    .step.wait { opacity: 0.7; }
    .step-line { flex: 0 0 56px; align-self: center; height: 4px; border-radius: 4px; background: var(--line); margin: 0 -6px; z-index: 0; }
    .step-line.on { background: linear-gradient(90deg, var(--p1), var(--k1), var(--p1)); background-size: 200% 100%; animation: lineFlow 2s linear infinite; }

    /* ---------- Feature cards ---------- */
    .grid { position: relative; z-index: 1; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.4rem; }
    .card {
        padding: 1.3rem; border-radius: 22px; border: 1px solid var(--line); background: rgba(255,255,255,0.85);
        animation: fadeUp 0.7s ease both; animation-delay: var(--d, 0s);
        transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
    }
    .card.pk { background: linear-gradient(160deg, #fff, var(--k4)); border-color: var(--k3); }
    .card.pu { background: linear-gradient(160deg, #fff, var(--p4)); }
    .card:hover { transform: translateY(-6px) rotate(-0.6deg); box-shadow: 0 16px 38px rgba(192,38,211,0.18); }
    .card-icon { font-size: 1.9rem; display: inline-block; margin-bottom: 0.5rem; animation: floatY 4s ease-in-out infinite; animation-delay: var(--d, 0s); }
    .card-title { font-weight: 700; color: var(--ink); margin-bottom: 0.25rem; }
    .card-desc { color: var(--muted); font-size: 0.86rem; line-height: 1.55; }

    /* ---------- Ready banner & category headers ---------- */
    .ready-banner {
        position: relative; z-index: 1; display: flex; align-items: center; gap: 0.7rem;
        padding: 0.7rem 1.2rem; margin-bottom: 1.1rem; border-radius: 999px;
        background: linear-gradient(90deg, #fff, var(--k4), #fff); border: 1px solid var(--k3);
        color: #9d174d; font-weight: 600; font-size: 0.88rem; animation: fadeUp 0.6s ease both;
    }
    .cat-head { display: flex; align-items: center; gap: 0.5rem; font-weight: 800; color: var(--ink); margin-bottom: 0.5rem; font-size: 0.95rem; }
    .cat-head .em { display: inline-flex; width: 30px; height: 30px; align-items: center; justify-content: center; border-radius: 10px; background: var(--p4); animation: floatY 3.5s ease-in-out infinite; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(250,245,255,0.95)) !important;
        border-right: 1px solid var(--line) !important; backdrop-filter: blur(12px);
    }
    section[data-testid="stSidebar"] > div { animation: fadeIn 0.8s ease both; }
    .side-logo { display: flex; align-items: center; gap: 0.7rem; padding: 0.2rem 0 1rem; }
    .side-logo .mark {
        width: 44px; height: 44px; border-radius: 15px; display: flex; align-items: center; justify-content: center;
        background: var(--grad); font-size: 1.4rem; animation: ping 2.8s ease-out infinite;
    }
    .side-logo .name { font-weight: 800; color: var(--ink); line-height: 1.1; font-size: 1.05rem; }
    .side-logo .sub { font-size: 0.72rem; color: var(--muted); }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-size: 0.74rem !important; letter-spacing: 0.14em; text-transform: uppercase; color: #9333ea !important; font-weight: 800 !important;
    }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: var(--ink) !important; font-weight: 600; }

    .status {
        display: flex; align-items: center; gap: 0.55rem; padding: 0.5rem 0.85rem; margin-bottom: 0.5rem;
        border-radius: 14px; font-size: 0.82rem; background: #fff; border: 1px solid var(--line); transition: all 0.3s ease;
    }
    .status .status-label { flex: 1; color: var(--ink); font-weight: 600; }
    .status .status-text { font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }
    .status.ok { border-color: #f9a8d4; background: var(--k4); }
    .status.ok .status-text { color: #be185d; }
    .status.wait .status-text { color: #a78bfa; }
    .pulse-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; animation: dotBeat 1.6s ease-in-out infinite; }
    .status.ok .pulse-dot { background: var(--k1); }
    .status.wait .pulse-dot { background: var(--p2); }

    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin: 0.4rem 0 0.8rem; }
    .stat { text-align: center; padding: 0.7rem 0.2rem; border-radius: 16px; background: #fff; border: 1px solid var(--line); transition: transform 0.25s ease; }
    .stat:nth-child(2) { background: var(--p4); }
    .stat:nth-child(3) { background: var(--k4); border-color: var(--k3); }
    .stat:hover { transform: translateY(-4px) scale(1.04); }
    .stat .n { font-size: 1.45rem; font-weight: 800; background: var(--grad); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent; }
    .stat .l { font-size: 0.64rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }

    /* ---------- Uploader ---------- */
    [data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.85) !important; border: 2px dashed var(--p2) !important; border-radius: 18px !important; transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--k2) !important; background: var(--k4) !important; box-shadow: 0 8px 24px rgba(236,72,153,0.15); }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploaderFile"] { color: var(--muted) !important; }
    [data-testid="stFileUploaderFileName"] { color: var(--ink) !important; font-weight: 600; }
    [data-testid="stFileUploaderDropzone"] button {
        background: #fff !important; color: #7e22ce !important; border: 1px solid var(--p3) !important; border-radius: 999px !important; font-weight: 600 !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        position: relative; overflow: hidden; width: 100%;
        background: var(--grad) !important; background-size: 160% 160% !important;
        color: #fff !important; border: none !important; border-radius: 999px !important;
        font-weight: 700 !important; padding: 0.55rem 1.1rem !important;
        box-shadow: 0 8px 20px rgba(192,38,211,0.28) !important; transition: all 0.3s ease !important;
    }
    .stButton > button *, .stDownloadButton > button * { color: inherit !important; }
    .stButton > button::after, .stDownloadButton > button::after {
        content: ""; position: absolute; top: 0; left: -120%; width: 55%; height: 100%;
        background: linear-gradient(100deg, transparent, rgba(255,255,255,0.5), transparent); transition: left 0.6s ease;
    }
    .stButton > button:hover::after, .stDownloadButton > button:hover::after { left: 140%; }
    .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-3px); box-shadow: 0 14px 28px rgba(236,72,153,0.38) !important; background-position: 100% 50% !important; }
    .stButton > button:active { transform: scale(0.96); }
    .stDownloadButton > button:disabled, .stButton > button:disabled { opacity: 0.45; box-shadow: none !important; }

    /* Suggestion chips (main-area buttons) */
    [data-testid="stMain"] .stButton > button, .main .stButton > button {
        background: #fff !important; color: #6d28d9 !important; border: 1px solid var(--line) !important;
        box-shadow: 0 3px 10px rgba(139,92,246,0.08) !important; justify-content: flex-start; text-align: left; margin-bottom: 0.15rem;
    }
    [data-testid="stMain"] .stButton > button:hover, .main .stButton > button:hover {
        background: var(--grad) !important; color: #fff !important; border-color: transparent !important; box-shadow: 0 10px 24px rgba(192,38,211,0.3) !important;
    }

    /* ---------- Inputs ---------- */
    [data-baseweb="select"] > div { background: #fff !important; border: 1px solid var(--line) !important; border-radius: 14px !important; color: var(--ink) !important; }
    [data-baseweb="select"] * { color: var(--ink) !important; }
    [data-baseweb="select"] > div:focus-within { border-color: var(--p1) !important; box-shadow: 0 0 0 3px rgba(139,92,246,0.2) !important; }
    [data-baseweb="popover"] ul, [data-baseweb="menu"] { background: #fff !important; }
    [data-baseweb="popover"] li { color: var(--ink) !important; }
    [data-baseweb="popover"] li:hover { background: var(--p4) !important; }
    [data-testid="stToggle"] label p, .stCheckbox label p { color: var(--ink) !important; }

    /* ---------- Chat input ---------- */
    [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent !important; }
    [data-testid="stChatInput"] {
        background: #fff !important; border: 1.5px solid var(--line) !important; border-radius: 999px !important;
        box-shadow: 0 10px 30px rgba(139,92,246,0.14); transition: all 0.3s ease;
    }
    [data-testid="stChatInput"]:focus-within { border-color: var(--p1) !important; box-shadow: 0 0 0 4px rgba(139,92,246,0.18), 0 12px 34px rgba(236,72,153,0.18) !important; }
    [data-testid="stChatInput"] textarea { color: var(--ink) !important; background: transparent !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #a897c9 !important; }
    [data-testid="stChatInput"] button { background: var(--grad) !important; border-radius: 50% !important; }
    [data-testid="stChatInput"] button svg { color: #fff !important; fill: #fff !important; }

    /* ---------- Chat bubbles ---------- */
    [data-testid="stChatMessage"] {
        position: relative; z-index: 1; border-radius: 24px !important; padding: 1rem 1.3rem !important; margin-bottom: 1rem !important;
        background: #fff !important; border: 1px solid var(--line) !important; box-shadow: 0 8px 26px rgba(139,92,246,0.1);
        margin-right: 8%; animation: fadeUp 0.5s ease both; transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    [data-testid="stChatMessage"]:hover { transform: translateY(-2px); box-shadow: 0 14px 34px rgba(139,92,246,0.18); }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse; margin-right: 0; margin-left: 12%;
        background: var(--grad) !important; background-size: 160% 160% !important; border: none !important;
        box-shadow: 0 12px 30px rgba(192,38,211,0.3); animation: fadeUp 0.5s ease both, gradMove 8s linear infinite;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) *,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) * { color: #fff !important; }
    [data-testid="stChatMessageAvatarUser"], [data-testid="chatAvatarIcon-user"] { background: var(--k1) !important; color: #fff !important; }
    [data-testid="stChatMessageAvatarAssistant"], [data-testid="chatAvatarIcon-assistant"] {
        background: var(--grad) !important; color: #fff !important; animation: ping 3s ease-out infinite;
    }

    /* Route badges */
    .route-badge {
        display: inline-flex; align-items: center; gap: 0.5rem; margin-top: 0.7rem; font-size: 0.72rem; font-weight: 700;
        padding: 0.28rem 0.8rem; border-radius: 999px; border: 1px solid; animation: fadeIn 0.8s ease both;
    }
    .route-badge .pulse-dot { width: 6px; height: 6px; }
    .route-academic { color: #6d28d9; border-color: var(--p3); background: var(--p4); }
    .route-academic .pulse-dot { background: var(--p1); }
    .route-fee { color: #be185d; border-color: var(--k3); background: var(--k4); }
    .route-fee .pulse-dot { background: var(--k1); }
    .route-general { color: #4338ca; border-color: #c7d2fe; background: #eef2ff; }
    .route-general .pulse-dot { background: #6366f1; }

    /* Thinking indicator */
    .thinking { display: flex; align-items: center; gap: 0.8rem; color: #7e22ce; font-size: 0.9rem; font-weight: 500; }
    .thinking .bars span { display: inline-block; width: 10px; height: 10px; margin-right: 5px; border-radius: 50%; background: var(--p1); animation: bounceDot 1.2s ease-in-out infinite; }
    .thinking .bars span:nth-child(2) { background: #c026d3; animation-delay: 0.15s; }
    .thinking .bars span:nth-child(3) { background: var(--k1); animation-delay: 0.3s; }
    .thinking-line { position: relative; height: 4px; margin-top: 0.7rem; border-radius: 4px; background: var(--p4); overflow: hidden; }
    .thinking-line::after { content: ""; position: absolute; top: 0; bottom: 0; width: 35%; border-radius: 4px; background: var(--grad); animation: sweep 1.3s ease-in-out infinite; }

    .footer-note { text-align: center; color: var(--muted); font-size: 0.78rem; margin: 2rem 0 5rem; position: relative; z-index: 1; }
    hr { border-top: 1px solid var(--line) !important; }
    [data-testid="stAlert"] { border-radius: 18px; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--p3); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--p1); }

    @media (max-width: 760px) {
        .hero { grid-template-columns: 1fr; padding: 1.4rem 1.2rem; }
        .hero-visual { display: none; }
        .hero-title { font-size: 1.9rem; }
        .stepper { flex-direction: column; gap: 0.9rem; }
        .step-line { display: none; }
        [data-testid="stChatMessage"] { margin-right: 0; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { margin-left: 0; }
    }
</style>

<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>
<div class="spark s1">✦</div><div class="spark s2">✦</div><div class="spark s3">✦</div><div class="spark s4">✦</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend (unchanged logic)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="✨ Initializing AI and reading documents...")
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
        "Campus Buddy - Chat Transcript",
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
            <div class="mark">🌸</div>
            <div><div class="name">Campus Buddy</div><div class="sub">Your AI college helper</div></div>
        </div>
    """), unsafe_allow_html=True)

    st.header("📂 Documents")
    academic_file = st.file_uploader("Academics Handbook (PDF)", type="pdf", key="acad")
    fee_file = st.file_uploader("Fee Structure (PDF)", type="pdf", key="fee")

    st.markdown(status_pill("Academics Handbook", bool(academic_file)), unsafe_allow_html=True)
    st.markdown(status_pill("Fee Structure", bool(fee_file)), unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.header("🎓 Your Profile")
    student_programme = st.selectbox(
        "Which programme are you in?",
        ["BCA", "Bachelor's", "B.com (H)"]
    )

    st.header("⚙️ Experience")
    typing_effect = st.toggle("✨ Typing animation", value=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.header("📊 This Session")
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

    st.download_button(
        "⬇️ Download Chat",
        data=build_transcript(student_programme),
        file_name="campus_buddy_chat.txt",
        mime="text/plain",
        disabled=not msgs,
    )
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.button("🗑️ Clear Chat", on_click=clear_chat)


# ---------------------------------------------------------------------------
# Hero + ticker
# ---------------------------------------------------------------------------
ready = bool(academic_file and fee_file)
compact = "compact" if ready else ""

st.markdown(html(f"""
    <div class="hero {compact}">
        <div class="hero-left">
            <div class="eyebrow"><span class="wave">👋</span> Hi there! I'm your campus buddy</div>
            <div class="hero-title">
                Got questions about
                <span class="rotator"><ul>
                    <li class="grad">attendance?</li>
                    <li class="grad">tuition fees?</li>
                    <li class="grad">exam rules?</li>
                    <li class="grad">your credits?</li>
                    <li class="grad">attendance?</li>
                </ul></span><br/>
                Get answers in seconds.
            </div>
            <div class="hero-desc">Upload your official handbook and fee structure, and I'll search them for you, with answers tailored to your programme.</div>
            <div class="pills">
                <span class="pill">⚡ Groq Powered</span>
                <span class="pill">🧠 RAG Search</span>
                <span class="pill">🕸️ LangGraph Routing</span>
                <span class="pill">🔒 From Official Docs</span>
            </div>
        </div>
        <div class="hero-visual">
            <div class="orbit">
                <div class="ring"></div><div class="ring2"></div>
                <div class="arm" style="--dl:0s"><div class="sat" style="--dl:0s">📚</div></div>
                <div class="arm" style="--dl:-6s"><div class="sat" style="--dl:-6s">💳</div></div>
                <div class="arm" style="--dl:-12s"><div class="sat" style="--dl:-12s">💬</div></div>
                <div class="core">🎓</div>
            </div>
        </div>
    </div>
"""), unsafe_allow_html=True)

ticker_items = ["Attendance", "Exams", "Grading", "Credits", "Tuition Fees", "Refunds", "Degrees", "Courses", "Payments", "Deadlines"]
ticker_html = "".join(f'<span class="tick">✦ {t}</span>' for t in ticker_items) * 2
st.markdown(f'<div class="ticker"><div class="ticker-track">{ticker_html}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if ready:
    app = setup_app(academic_file.getvalue(), fee_file.getvalue())

    st.markdown(html(f"""
        <div class="ready-banner">
            <i class="pulse-dot" style="background:#ec4899"></i>
            Knowledge base ready &nbsp;•&nbsp; Personalised for <b>{student_programme}</b>
        </div>
    """), unsafe_allow_html=True)

    # Replay history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("route"):
                st.markdown(route_badge(msg["route"]), unsafe_allow_html=True)

    # Category-wise suggestions (only on an empty chat)
    clicked_query = None
    if not st.session_state["chat_messages"]:
        st.markdown('<div class="section-title">💡 Try asking</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for ci, (emoji, name, items) in enumerate(SUGGESTION_GROUPS):
            with cols[ci]:
                st.markdown(f'<div class="cat-head"><span class="em">{emoji}</span>{name}</div>', unsafe_allow_html=True)
                for li, (label, q) in enumerate(items):
                    if st.button(label, key=f"chip_{ci}_{li}", help=q):
                        clicked_query = q

    typed_query = st.chat_input("Ask me anything about your college...")
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
                    <em>Finding the best answer for you…</em>
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
    st.markdown('<div class="section-title">🚀 Let\'s get you set up</div>', unsafe_allow_html=True)
    st.markdown(stepper(bool(academic_file), bool(fee_file)), unsafe_allow_html=True)

    st.markdown('<div class="section-title">✨ Why you\'ll love it</div>', unsafe_allow_html=True)
    st.markdown(html("""
        <div class="grid">
            <div class="card pu" style="--d:0.1s">
                <div class="card-icon">🧭</div>
                <div class="card-title">Smart Routing</div>
                <div class="card-desc">Every question is sent to the right source automatically.</div>
            </div>
            <div class="card pk" style="--d:0.25s">
                <div class="card-icon">📖</div>
                <div class="card-title">Grounded in Your PDFs</div>
                <div class="card-desc">Answers come from your official documents, not guesses.</div>
            </div>
            <div class="card pu" style="--d:0.4s">
                <div class="card-icon">🎯</div>
                <div class="card-title">Programme Aware</div>
                <div class="card-desc">BCA, Bachelor's or B.Com (H): figures highlighted for you.</div>
            </div>
            <div class="card pk" style="--d:0.55s">
                <div class="card-icon">⚡</div>
                <div class="card-title">Lightning Fast</div>
                <div class="card-desc">Powered by Groq for near-instant replies.</div>
            </div>
        </div>
    """), unsafe_allow_html=True)

st.markdown('<div class="footer-note">Made with 💜 for students</div>', unsafe_allow_html=True)
