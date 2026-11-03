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

# Get API key from Streamlit secrets
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it in your App Settings.")
    st.stop()

# ---------------------------------------------------------------------------
# Retro Vintage UI Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

    /* Main App Background */
    .stApp, .main {
        background-color: #F5EEDC !important;
        color: #3D3228 !important;
        font-family: 'Lato', sans-serif !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        color: #8B5E3C !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #EAE0CC !important;
        border-right: 1px solid #D6C6AA !important;
    }
    
    /* Text rendering */
    p, span, div {
        color: #3D3228 !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #F5EEDC !important;
        border: 1px dashed #8B5E3C !important;
        border-radius: 6px !important;
        padding: 1rem !important;
        box-shadow: 2px 2px 5px rgba(61, 50, 40, 0.05) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #8B5E3C !important;
        color: #F5EEDC !important;
        border: 1px solid #8B5E3C !important;
        border-radius: 4px !important;
        font-family: 'Lato', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 1px 1px 3px rgba(61, 50, 40, 0.2) !important;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #C17C54 !important;
        border-color: #C17C54 !important;
        color: #F5EEDC !important;
        box-shadow: 2px 2px 6px rgba(61, 50, 40, 0.3) !important;
    }

    /* Selectbox & Input Fields */
    .stSelectbox > div > div, .stTextInput > div > div > input {
        background-color: #EAE0CC !important;
        border: 1px solid #D6C6AA !important;
        color: #3D3228 !important;
        border-radius: 4px !important;
    }
    .stSelectbox > div > div:focus-within, .stTextInput > div > div > input:focus {
        border-color: #8B5E3C !important;
        box-shadow: 0 0 0 1px #8B5E3C !important;
    }

    /* Chat Input Container */
    .stChatInputContainer > div {
        background-color: #EAE0CC !important;
        border: 1px solid #8B5E3C !important;
        border-radius: 6px !important;
    }
    .stChatInputContainer textarea {
        color: #3D3228 !important;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        border-radius: 8px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid #D6C6AA !important;
        box-shadow: 1px 2px 4px rgba(61, 50, 40, 0.05) !important;
    }
    
    /* Assistant Chat Bubble */
    [data-testid="stChatMessage"][data-baseweb="card"]:nth-child(even) {
        background-color: #EAE0CC !important; 
    }
    /* User Chat Bubble */
    [data-testid="stChatMessage"][data-baseweb="card"]:nth-child(odd) {
        background-color: #F5EEDC !important;
        border-color: #8B5E3C !important;
    }

    /* Avatars */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #A3A380 !important;
        color: #F5EEDC !important;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #8B5E3C !important;
        color: #F5EEDC !important;
    }
    
    /* Subtle Vintage textures / details */
    hr {
        border-top: 1px solid #D6C6AA !important;
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
st.title("🎓 College Assistant")

# Sidebar for file uploads
with st.sidebar:
    st.header("1. Upload Documents")
    academic_file = st.file_uploader("Upload Academics Handbook (PDF)", type="pdf", key="acad")
    fee_file = st.file_uploader("Upload Fee Structure (PDF)", type="pdf", key="fee")
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    st.header("2. Student Profile")
    student_programme = st.selectbox(
        "Which programme are you in?",
        ["BCA", "Bachelor's", "B.com (H)"]
    )
    
    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("Clear Chat"):
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
    st.info("👋 Welcome! Please upload both the **Academics Handbook** and the **Fee Structure** PDFs in the sidebar to the left to begin.")
