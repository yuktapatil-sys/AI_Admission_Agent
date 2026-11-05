import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from ingest import (
    BASE_DIR,
    DOCUMENTS_DIR,
    get_collection,
    ingest_documents,
    retrieve_context,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv(BASE_DIR / ".env", override=True)

st.set_page_config(
    page_title="Indala College | AI Admission Portal",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.75"))
TOP_K = int(os.getenv("TOP_K", "5"))

UNAVAILABLE = (
    "Sorry, this information is not available in the supplied "
    "college documents. Please contact the Indala College "
    "Admission Office for accurate information."
)

SYSTEM_PROMPT = """
You are the official AI admission assistant for Indala College.

Answer using ONLY the supplied college document excerpts.

If information is unavailable, clearly state:
"Sorry, this information is not available in the supplied
college documents. Please contact the Indala College
Admission Office for accurate information."

Do not invent fees, eligibility rules, admission dates,
course details or contact information.

Do not use outside knowledge or assumptions.
Use simple, polite and student-friendly language.
Do not guess document names or page numbers.
"""


# --------------------------------------------------
# CUSTOM CSS - MODERN COLLEGE PORTAL
# --------------------------------------------------


st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #f5f8fc;
}

/* Remove default top spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #102b4e;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button {
    background-color: #159B88 !important;
    color: #ffffff !important;
    border-radius: 9px;
    border: none !important;
    font-weight: 700;
    transition: 0.3s ease;
}

[data-testid="stSidebar"] .stButton button p {
    color: #ffffff !important;
}

[data-testid="stSidebar"] .stButton button:hover {
    background-color: #087C70 !important;
    color: #ffffff !important;
    transform: translateY(-2px);
}

/* PDF File Uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background-color: #e6f4f1 !important;
    border: 2px dashed #159B88 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* File uploader drop area */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #e6f4f1 !important;
    border: none !important;
    border-radius: 10px !important;
}

/* File uploader text */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
    color: #102b4e !important;
}

/* Browse files button */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: #159B88 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button p {
    color: white !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background-color: #087C70 !important;
}

/* College header */
.college-header {
    background: #102b4e;
    padding: 22px 28px;
    border-radius: 14px;
    color: white;
    margin-bottom: 25px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.college-name {
    font-size: 25px;
    font-weight: 800;
    margin: 0;
}

.college-subtitle {
    font-size: 13px;
    color: #d5e5f5;
    margin-top: 5px;
}

.header-tag {
    background: #d9f5ed;
    color: #087c62;
    padding: 8px 14px;
    border-radius: 25px;
    font-size: 12px;
    font-weight: 700;
}

/* Welcome banner */
.welcome-banner {
    background: linear-gradient(110deg, #e5f0ff, #e7f8f4);
    border-radius: 15px;
    padding: 30px;
    margin-bottom: 25px;
    border: 1px solid #d9e8f5;
}

.welcome-title {
    font-size: 30px;
    font-weight: 800;
    color: #102b4e;
    margin-bottom: 10px;
}

.welcome-description {
    color: #53677e;
    font-size: 15px;
    line-height: 1.7;
}

/* Section headings */
.section-heading {
    color: #102b4e;
    font-size: 20px;
    font-weight: 750;
    margin-top: 18px;
    margin-bottom: 15px;
}

/* Information cards */
.info-card {
    background: white;
    border: 1px solid #e1e9f2;
    border-radius: 13px;
    padding: 20px;
    min-height: 135px;
    box-shadow: 0 3px 12px rgba(20, 48, 80, 0.04);
}

.card-icon {
    font-size: 27px;
    margin-bottom: 10px;
}

.card-title {
    color: #102b4e;
    font-size: 16px;
    font-weight: 750;
    margin-bottom: 5px;
}

.card-description {
    color: #718096;
    font-size: 12px;
}

/* Chat area */
[data-testid="stChatMessage"] {
    background-color: white;
    border: 1px solid #e5eaf1;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 12px;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: white;
}

/* General buttons */
.stButton button {
    border-radius: 9px;
    min-height: 43px;
    font-weight: 600;
    border-color: #dce5ef;
}

.stButton button:hover {
    border-color: #159b88;
    color: #087c70;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e1e9f2;
    padding: 15px;
    border-radius: 12px;
}

/* Footer */
.footer {
    text-align: center;
    color: #718096;
    font-size: 12px;
    padding: 20px;
    margin-top: 30px;
    border-top: 1px solid #e1e9f2;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# GROQ AI RESPONSE
# --------------------------------------------------

def generate_answer(question, matches):

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    context_parts = []

    for item in matches:

        context_parts.append(
            f"Source: {item['source']} | Page: {item['page']}\n"
            f"Excerpt: {item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b"
        ),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"College document excerpts:\n{context}\n\n"
                    "Answer only using the supplied evidence. "
                    "If the evidence is insufficient, say so."
                )
            }
        ]
    )

    return response.choices[0].message.content.strip()


# --------------------------------------------------
# SAVE UPLOADED PDF FILES
# --------------------------------------------------

def save_uploaded_files(uploaded_files):

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    saved = []

    for uploaded_file in uploaded_files:

        if not uploaded_file.name.lower().endswith(".pdf"):
            continue

        destination = (
            DOCUMENTS_DIR / Path(uploaded_file.name).name
        )

        destination.write_bytes(
            uploaded_file.getbuffer()
        )

        saved.append(destination)

    return saved


# --------------------------------------------------
# COLLEGE HEADER
# --------------------------------------------------


# --------------------------------------------------
# COLLEGE HEADER
# --------------------------------------------------

st.markdown("""
<div class="college-header">
<div>
<div class="college-name">🎓 INDALA COLLEGE</div>
<div class="college-subtitle">AI-Powered Admission Assistance Portal</div>
</div>
<div class="header-tag">● ADMISSION ASSISTANT</div>
</div>
""", unsafe_allow_html=True)



# --------------------------------------------------
# SIDEBAR - ADMINISTRATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:15px 5px 25px;">
        <div style="font-size:42px;">🎓</div>
        <div style="font-size:21px; font-weight:800;">
            INDALA COLLEGE
        </div>
        <div style="font-size:12px; opacity:0.8;">
            Admission Portal
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🏠 Navigation")

    st.caption("Welcome to the admission assistant.")

    st.divider()

    st.markdown("### 📁 Document Administration")

    st.caption(
        "Upload official college PDFs and process them "
        "to update the AI knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Select PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "💾 Save Uploaded PDFs",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning("Please select PDF documents first.")

        else:

            saved = save_uploaded_files(uploaded_files)

            st.success(
                f"{len(saved)} PDF file(s) saved successfully."
            )

    if st.button(
        "⚙️ Process Documents",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Processing college documents..."
            ):

                result = ingest_documents()

            st.success(
                f"Processing complete! "
                f"{result['stored_chunks']} chunks stored."
            )

            if result["skipped_files"]:

                st.warning(
                    "Some PDFs did not contain selectable text: "
                    + ", ".join(result["skipped_files"])
                )

        except Exception as exc:

            st.error(f"Document processing failed: {exc}")

    st.divider()

    st.markdown("### 📊 Database Status")

    try:

        chunk_count = get_collection().count()

    except Exception:

        chunk_count = 0

    st.metric(
        "Stored Text Chunks",
        chunk_count
    )

    st.divider()

    st.markdown("### 🤖 AI Configuration")

    st.write("AI Provider: Groq")

    st.write(
        "Model: " +
        os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    )

    st.divider()

    st.caption(
        "Upload only official college documents "
        "authorized for sharing."
    )


# --------------------------------------------------
# WELCOME BANNER
# --------------------------------------------------

st.markdown("""
<div style="
    background: linear-gradient(135deg, #e3f2fd, #e0f7f4);
    padding: 25px;
    border-radius: 12px;
    color: #153451;
">
    <h2>Welcome to Indala College! 👋</h2>
    <p>
        Your AI-powered admission assistant is here to help you
        explore engineering courses, understand eligibility,
        and learn about admission procedures.
    </p>
    <p>
        Ask your question below and get answers based on
        official college documents.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

try:
    total_chunks = get_collection().count()
except Exception:
    total_chunks = 0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📚 Knowledge Base",
        "Available" if total_chunks > 0 else "Not Processed"
    )

with col2:
    st.metric(
        "📄 Document Chunks",
        total_chunks
    )

with col3:
    st.metric(
        "🤖 AI Assistant",
        "Online"
    )


# --------------------------------------------------
# QUICK ACCESS CARDS
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">Explore Admission Information</div>',
    unsafe_allow_html=True
)

card1, card2, card3, card4 = st.columns(4)

with card1:

    st.markdown("""
    <div class="info-card">
        <div class="card-icon">📚</div>
        <div class="card-title">Courses</div>
        <div class="card-description">
            Explore engineering branches and courses.
        </div>
    </div>
    """, unsafe_allow_html=True)

with card2:

    st.markdown("""
    <div class="info-card">
        <div class="card-icon">🎯</div>
        <div class="card-title">Eligibility</div>
        <div class="card-description">
            Understand admission eligibility criteria.
        </div>
    </div>
    """, unsafe_allow_html=True)

with card3:

    st.markdown("""
    <div class="info-card">
        <div class="card-icon">💰</div>
        <div class="card-title">Fee Structure</div>
        <div class="card-description">
            Find available course fee information.
        </div>
    </div>
    """, unsafe_allow_html=True)

with card4:

    st.markdown("""
    <div class="info-card">
        <div class="card-icon">📝</div>
        <div class="card-title">Admission</div>
        <div class="card-description">
            Learn about the admission procedure.
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# CHAT SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">💬 Chat with Admission Assistant</div>',
    unsafe_allow_html=True
)

st.caption(
    "Select a suggested question or type your own question."
)


# --------------------------------------------------
# QUICK QUESTION BUTTONS
# --------------------------------------------------

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

qcol1, qcol2 = st.columns(2)

with qcol1:

    if st.button(
        "Which engineering courses are available?",
        use_container_width=True
    ):
        st.session_state.pending_question = (
            "Which engineering courses are available?"
        )

    if st.button(
        "What are the eligibility criteria?",
        use_container_width=True
    ):
        st.session_state.pending_question = (
            "What are the eligibility criteria?"
        )

with qcol2:

    if st.button(
        "Which documents are required?",
        use_container_width=True
    ):
        st.session_state.pending_question = (
            "Which documents are required?"
        )

    if st.button(
        "Explain the admission procedure.",
        use_container_width=True
    ):
        st.session_state.pending_question = (
            "Explain the admission procedure."
        )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📂 View Source Documents"):

                for source in message["sources"]:

                    st.write(
                        f"📄 **{source['source']}** "
                        f"— Page {source['page']}"
                    )


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

typed_question = st.chat_input(
    "Ask anything about Indala College admissions..."
)

question = (
    typed_question
    if typed_question
    else st.session_state.pending_question
)

st.session_state.pending_question = None


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "🔍 Searching official college documents..."
            ):

                matches = retrieve_context(
                    question,
                    top_k=TOP_K
                )

            relevant_matches = [
                item
                for item in matches
                if item["distance"] <= MAX_DISTANCE
            ]

            if not relevant_matches:

                answer = UNAVAILABLE
                sources = []

            else:

                with st.spinner(
                    "🤖 Groq AI is preparing your answer..."
                ):

                    answer = generate_answer(
                        question,
                        relevant_matches
                    )

                unique_sources = []
                seen = set()

                for item in relevant_matches:

                    key = (
                        item["source"],
                        item["page"]
                    )

                    if key not in seen:

                        seen.add(key)

                        unique_sources.append({
                            "source": item["source"],
                            "page": item["page"]
                        })

                sources = unique_sources

            st.markdown(answer)

            if sources:

                with st.expander(
                    "📂 View Source Documents"
                ):

                    for source in sources:

                        st.write(
                            f"📄 **{source['source']}** "
                            f"— Page {source['page']}"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as exc:

            error_message = (
                f"Unable to generate an answer: {exc}"
            )

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
<b>INDALA COLLEGE | AI ADMISSION ASSISTANT</b>
<br><br>
Powered by Groq AI · ChromaDB · Streamlit
<br>
For official confirmation, please contact the
Indala College Admission Office.
</div>
""", unsafe_allow_html=True)
