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

st.set_page_config(page_title="College Assistant", page_icon="✨", layout="wide")

# Get API key from Streamlit secrets
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it in your App Settings.")
    st.stop()

# ---------------------------------------------------------------------------
# Infinite Animation & Creative UI Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&family=Righteous&display=swap');

    /* 1. Infinite Liquid Gradient Background */
    .stApp, .main {
        background: linear-gradient(-45deg, #1f005c, #5b0060, #870160, #ac255e, #ca485c, #e16b5c, #f39060, #ffb56b);
        background-size: 400% 400%;
        animation: liquidGradient 15s ease infinite alternate;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
    }
    @keyframes liquidGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 2. Creative Headings with Infinite Glow */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Righteous', cursive !important;
        color: #ffffff !important;
        letter-spacing: 2px !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        animation: textGlow 3s ease-in-out infinite alternate;
    }
    @keyframes textGlow {
        0% { text-shadow: 0 0 5px rgba(255, 255, 255, 0.5); }
        100% { text-shadow: 0 0 20px rgba(255, 255, 255, 1), 0 0 30px #ffb56b; }
    }

    /* 3. Glassmorphism Sidebar with Floating Effect */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Text rendering override for global elements */
    p, span, div, label {
        color: #ffffff !important;
    }

    /* 4. Floating & Glowing File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px dashed rgba(255, 255, 255, 0.5) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        animation: floatAnimation 4s ease-in-out infinite;
    }
    [data-testid="stFileUploader"]:hover {
        transform: scale(1.05) translateY(-5px) !important;
        border-color: #ffb56b !important;
        box-shadow: 0 10px 30px rgba(255, 181, 107, 0.4) !important;
    }
    @keyframes floatAnimation {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }

    /* 5. Buttons with Infinite Pulse and Expanding Hover */
    .stButton > button {
        background: linear-gradient(90deg, #ff7eb3, #ff758c) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 30px !important;
        font-family: 'Righteous', cursive !important;
        font-size: 1.1rem !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 4px 15px rgba(255, 117, 140, 0.4) !important;
        animation: pulseButton 2.5s infinite;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 10px 25px rgba(255, 117, 140, 0.7) !important;
        background: linear-gradient(90deg, #ff758c, #ff7eb3) !important;
        letter-spacing: 2px !important;
    }
    @keyframes pulseButton {
        0% { box-shadow: 0 0 0 0 rgba(255, 117, 140, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 117, 140, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 117, 140, 0); }
    }

    /* 6. Selectbox & Input Fields - Smooth Transitions */
    .stSelectbox > div > div, .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    .stSelectbox > div > div:focus-within, .stTextInput > div > div > input:focus {
        border-color: #ffb56b !important;
        box-shadow: 0 0 15px rgba(255, 181, 107, 0.6) !important;
        background: rgba(0, 0, 0, 0.4) !important;
        transform: scale(1.02);
    }

    /* 7. Chat Input Container Glow */
    .stChatInputContainer > div {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        transition: all 0.4s ease;
    }
    .stChatInputContainer > div:focus-within {
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.5) !important;
        border-color: #ffffff !important;
    }
    .stChatInputContainer textarea {
        color: #ffffff !important;
    }

    /* 8. Interactive Chat Bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.4s ease !important;
        animation: slideUp 0.6s cubic-bezier(0.23, 1, 0.320, 1) both;
    }
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-5px) scale(1.01) !important;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3) !important;
    }
    @keyframes slideUp {
        0% { opacity: 0; transform: translateY(30px) scale(0.95); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    /* Assistant Chat Bubble */
    [data-testid="stChatMessage"][data-baseweb="card"]:nth-child(even) {
        background: linear-gradient(135deg, rgba(255, 117, 140, 0.15), rgba(255, 126, 179, 0.15)) !important;
        border-left: 4px solid #ff758c !important;
    }
    /* User Chat Bubble */
    [data-testid="stChatMessage"][data-baseweb="card"]:nth-child(odd) {
        background: linear-gradient(135deg, rgba(255, 181, 107, 0.15), rgba(243, 144, 96, 0.15)) !important;
        border-right: 4px solid #ffb56b !important;
    }

    /* 9. Floating Avatars */
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(45deg, #f39060, #ffb56b) !important;
        color: #ffffff !important;
        animation: floatAvatar 3s ease-in-out infinite;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(45deg, #ac255e, #ff758c) !important;
        color: #ffffff !important;
        animation: floatAvatar 3s ease-in-out infinite alternate-reverse;
    }
    @keyframes floatAvatar {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-4px) rotate(5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    hr {
        border-top: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)


# Cache the heavy AI loading and PDF reading so it only runs once per unique file upload
@st.cache_resource(show_spinner="Initializing AI and reading documents...")
def setup_app(academic_bytes, fee_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_acad:
        tmp_acad.write(academic_bytes)
        acad_path = tmp_acad.name
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_fee:
        tmp_fee.write(fee_bytes)
        fee_path = tmp_fee.name

    # 1. Build the RAG Retrievers
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    def build_retriever(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        document = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(document)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    academic_retriever = build_retriever(acad_path)
    fee_retriever = build_retriever(fee_path)

    os.remove(acad_path)
    os.remove(fee_path)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
    )

    # 2. Create a state
    class State(TypedDict):
        programme: str
        messages: Annotated[list, add_messages]
        query_type: str
        retrieved_context: str

    # 3. Creating the nodes
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

    # 4. Router function
    def route_query(state: State):
        if state['query_type'] == 'academic':
            return "academic"
        elif state['query_type'] == 'fee':
            return "fee_rag"
        else:
            return "general"

    # 5. Building the graph
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
st.title("✨ College Assistant ✨")

# Sidebar for file uploads
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
    if st.button("CLEAR CHAT 🚀"):
        st.session_state["chat_messages"] = []
        st.rerun()

# Only initialize the app and show chat if both files are uploaded
if academic_file and fee_file:
    app = setup_app(academic_file.getvalue(), fee_file.getvalue())
    
    # Initialize chat history using robust dictionary notation
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Display previous messages
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Box
    user_query = st.chat_input("Type your question here...")

    if user_query:
        # 1. Display User Message
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Invoke LangGraph with Error Handling
        with st.spinner("Checking documents and analyzing query..."):
            try:
                result = app.invoke({
                    "programme": student_programme,
                    "messages": [("human", user_query)]
                })
                answer = result['messages'][-1].content

                # 3. Display AI Response
                st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Check if your Groq API key is valid and you have enough usage quota left.")

else:
    st.info("👋 Welcome to the Interactive College Hub! Please upload both the **Academics Handbook** and the **Fee Structure** PDFs in the sidebar to the left to initiate the AI matrix.")
