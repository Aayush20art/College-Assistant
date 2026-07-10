# 🎓 College Assistant AI

An intelligent AI-powered College Assistant built using **LangGraph**, **LangChain**, **Mistral AI**, **FAISS**, and **Streamlit**. The assistant answers students' questions related to academics, fee structure, and general college information by retrieving information from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://college-assistant-ei4i2rzqjfkdbyrodzftp8.streamlit.app/

## 💻 GitHub Repository

🔗 https://github.com/Aayush20art/College-Assistant

---

# 📌 Features

- 📘 Academic Handbook Question Answering
- 💰 Fee Structure Question Answering
- 🤖 Intelligent Query Classification
- 📄 PDF Upload Support
- 🔍 Retrieval-Augmented Generation (RAG)
- ⚡ Fast Semantic Search using FAISS
- 🎯 Personalized responses based on student's programme
- 💬 Clean Chat Interface
- 🎨 Modern Animated Streamlit UI
- 🧠 Powered by Mistral AI

---

# 🏗️ Project Architecture

```
                User Query
                     │
                     ▼
          LangGraph Classifier
                     │
     ┌───────────────┼───────────────┐
     │               │               │
Academic        Fee Structure     General
   RAG               RAG            Chat
     │               │               │
     └───────────────┼───────────────┘
                     ▼
           Mistral AI Response
                     │
                     ▼
              Streamlit Interface
```

---

# 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python
- LangGraph
- LangChain

### AI Models
- Mistral Small 2506

### Vector Database
- FAISS

### Embedding Model
- sentence-transformers/all-MiniLM-L6-v2

### Document Processing
- PyPDFLoader
- RecursiveCharacterTextSplitter

---

# 📂 Project Structure

```
College-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
├── Academic_Handbook.pdf
├── Fee_Structure.pdf
│
└── assets/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Aayush20art/College-Assistant.git
```

Move into the project

```bash
cd College-Assistant
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
MISTRAL_API_KEY=your_api_key_here
```

If deploying on Streamlit Cloud, add

```
MISTRAL_API_KEY
```

inside **Secrets**.

---

# ▶️ Run Locally

```bash
streamlit run app.py
```

---

# 📄 How It Works

1. Upload the Academic Handbook PDF.
2. Upload the Fee Structure PDF.
3. Select your programme.
4. Ask any question.
5. LangGraph classifies the query into:

- Academic
- Fee
- General

6. Relevant documents are searched using FAISS.
7. Retrieved context is sent to Mistral AI.
8. A precise response is generated.

---

# 🧠 AI Workflow

```
PDF Upload
      │
      ▼
Text Extraction
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
User Question
      │
      ▼
Query Classification
      │
      ▼
Relevant Retrieval
      │
      ▼
Mistral AI
      │
      ▼
Final Answer
```

---

# 🎯 Supported Query Types

### 📘 Academic

- Attendance
- Credits
- Examination
- Promotion Rules
- Semester System
- Grading
- Course Structure

### 💰 Fee

- Tuition Fees
- Scholarships
- Refund Policy
- Hostel Fees
- Payment Schedule
- Late Fee

### 💬 General

- Greetings
- Casual Conversation
- Basic College Information

---

# 📸 Application Preview

You can view the live application here:

https://college-assistant-ei4i2rzqjfkdbyrodzftp8.streamlit.app/

---

# 📦 Dependencies

- streamlit
- langgraph
- langchain
- langchain-community
- langchain-mistralai
- langchain-huggingface
- sentence-transformers
- faiss-cpu
- python-dotenv
- pypdf

---

# 🚀 Future Improvements

- Multi-PDF Support
- Conversation Memory
- Voice Assistant
- OCR Support
- Admin Dashboard
- Student Login
- Chat History Export
- Multiple LLM Support
- Cloud Database Integration

---

# 👨‍💻 Author

**Aayush Sharma**

GitHub:
https://github.com/Aayush20art

LinkedIn:
https://www.linkedin.com/in/aayush-sharma20/

---

# ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub!
