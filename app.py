import os
import tempfile
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
# Elegant "Midnight Glass" Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* 1. Clean, Deep Background */
    .stApp, .main {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* 2. Refined Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #e0e7ff !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }

    /* 3. Smooth Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b !important;
    }
    
    p, span, div, label {
        color: #cbd5e1 !important;
    }

    /* 4. Elegant File Uploader with Subtle Breathing Glow */
    [data-testid="stFileUploader"] {
        background-color: #1e293b !important;
        border: 1px dashed #6366f1 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
        animation: subtlePulse 4s infinite alternate;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #818cf8 !important;
        background-color: #334155 !important;
        transform: translateY(-2px);
    }
    @keyframes subtlePulse {
        0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
        100% { box-shadow: 0 0 15px 0 rgba(99, 102, 241, 0.15); }
    }

    /* 5. Modern Gradient Buttons with Hover Lift */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2), 0 2px 4px -1px rgba(79, 70, 229, 0.1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3), 0 4px 6px -2px rgba(79, 70, 229, 0.15) !important;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    }

    /* 6. Clean Inputs */
    .stSelectbox > div > div, .stTextInput > div > div > input {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .stSelectbox > div > div:focus-within, .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    /* 7. Chat Input Container */
    .stChatInputContainer > div {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    .stChatInputContainer > div:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.1) !important;
    }
    .stChatInputContainer textarea {
        color: #f8fafc !important;
    }

    /* 8. Polished Chat Bubbles with Fade-In */
    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid #334155 !important;
        transition: transform 0.2s ease !important;
        animation: smoothFadeIn 0.5s ease forwards;
    }
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    @keyframes smoothFadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Assistant Chat Bubble */
    [data-testid="stChatMessage"][data-baseweb="card"]:nth-child(even) {
        background-color: #1e293b !important;
        border-left: 4px solid #6366f1 !important;
    }
    /* User Chat Bubble */
    [data-testid="stChatMessage"][data-baseweb="card"]:nth-child(odd) {
        background-color: #172033 !important;
        border-right: 4px solid #8b5cf6 !important;
    }

    /* 9. Minimalist Avatars */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #8b5cf6 !important;
        color: #ffffff !important;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
    }
    
    hr {
        border-top: 1px solid #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

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

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

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
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("🎓 College Assistant")

with st.sidebar:
    st.header("📂 Upload Documents")
    academic_file = st.file_uploader("Upload Academics Handbook (PDF)", type="pdf", key="acad")
    fee_file = st.file_uploader("Upload Fee Structure (PDF)", type="pdf", key="fee")
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    st.header("🎓 Student Profile")
    student_programme = st.selectbox(
        "Which programme are you in?",
        ["BCA", "Bachelor's", "B.com (H)"]
    )
    
    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("Clear Chat"):
        st.session_state["chat_messages"] = []
        st.rerun()

if academic_file and fee_file:
    app = setup_app(academic_file.getvalue(), fee_file.getvalue())
    
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Type your question here...")

    if user_query:
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Analyzing query..."):
            try:
                result = app.invoke({
                    "programme": student_programme,
                    "messages": [("human", user_query)]
                })
                answer = result['messages'][-1].content

                st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

else:
    st.info("👋 Welcome! Please upload both the **Academics Handbook** and the **Fee Structure** PDFs in the sidebar to initialize the AI.")
