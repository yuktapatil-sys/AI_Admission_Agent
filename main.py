
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


from ingest import (
    BASE_DIR,
    DOCUMENTS_DIR,
    get_collection,
    ingest_documents,
    retrieve_context,
)

# Load environment variables
load_dotenv(
    dotenv_path=BASE_DIR / ".env",
    override=True
)

# Streamlit configuration
st.set_page_config(
    page_title="Indala College | Admission Assistant",
    page_icon="🎓",
    layout="wide"
)

# System instructions
SYSTEM_PROMPT = """
You are the admission assistance chatbot for Indala College.

Answer using only the supplied college document excerpts.

If the information is unavailable in the excerpts, clearly say:
"Sorry, this information is not available in the supplied college documents.
Please contact the Indala College Admission Office for accurate information."

Do not invent fees, eligibility rules, admission dates, course details
or contact information.

Do not use outside knowledge or assumptions.
Use simple, polite, student-friendly language.
If the excerpts do not support an answer, say the information is unavailable.
Do not create or guess page numbers or document names.
"""

# Configuration
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.75"))
TOP_K = int(os.getenv("TOP_K", "5"))

UNAVAILABLE = (
    "Sorry, this information is not available in the supplied college "
    "documents. Please contact the Indala College Admission Office "
    "for accurate information."
)


# --------------------------------------------------
# CUSTOM UI DESIGN
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #142d4e, #203f66);
}

section[data-testid="stSidebar"] * {
    color: white;
}

section[data-testid="stSidebar"] .stButton button {
    background: #294c74;
    color: white;
    border: 1px solid #52749b;
    border-radius: 9px;
}

/* MAIN HEADER */

.header-title {
    font-size: 32px;
    font-weight: 800;
    color: #193b61;
    margin-bottom: 4px;
}

.header-subtitle {
    font-size: 14px;
    color: #718096;
}

/* WELCOME BANNER */

.welcome-banner {
    background: linear-gradient(120deg, #173b63, #3979b9);
    padding: 30px;
    border-radius: 17px;
    color: white;
    margin-top: 22px;
    margin-bottom: 25px;
}

.welcome-banner h2 {
    color: white;
    font-size: 27px;
    font-weight: 700;
}

.welcome-banner p {
    color: #e3edf8;
    font-size: 14px;
    line-height: 1.8;
}

/* INFO CARDS */

.info-card {
    background: white;
    padding: 20px;
    border-radius: 13px;
    border: 1px solid #e2eaf3;
    min-height: 140px;
    margin-bottom: 12px;
}

.info-card h4 {
    color: #244d77;
    font-size: 16px;
}

.info-card p {
    color: #77869a;
    font-size: 13px;
    line-height: 1.6;
}

/* CHAT MESSAGES */

[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid #e1e9f2;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 12px;
}

/* INPUTS */

.stTextInput input {
    border-radius: 9px;
}

.stChatInput textarea {
    border-radius: 12px;
}

/* BUTTONS */

.stButton button {
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
}

.stButton button[kind="primary"] {
    background-color: #28669f;
    color: white;
    border: none;
}

/* METRICS */

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1e9f2;
    padding: 15px;
    border-radius: 12px;
}

/* FOOTER */

.footer {
    text-align: center;
    padding: 22px;
    color: #8391a3;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# GROQ API ANSWER GENERATION
# --------------------------------------------------

def generate_answer(question, matches):

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Please add your Groq API key to the .env file."
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
                    f"Supplied college document excerpts:\n"
                    f"{context}\n\n"
                    "Answer only if the excerpts contain enough evidence. "
                    "Do not use external knowledge."
                )
            }
        ]
    )

    return response.choices[0].message.content.strip()


# --------------------------------------------------
# SAVE UPLOADED PDF DOCUMENTS
# --------------------------------------------------

def save_uploaded_files(uploaded_files):

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

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
# PAGE HEADER
# --------------------------------------------------

st.markdown("""
<div class="header-title">
    🎓 Indala College
</div>

<div class="header-subtitle">
    Smart AI-Powered Admission Assistance Portal
</div>

<div class="welcome-banner">

    <h2>Welcome to Indala College!</h2>

    <p>
        Your intelligent admission assistant is here to help.
        Ask questions about courses, eligibility, admission
        procedures, fees and required documents.
    </p>

    <p>
        ✨ Get information quickly using our AI-powered
        college knowledge base.
    </p>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR - DOCUMENT ADMINISTRATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:15px 5px;">

        <div style="font-size:48px;">🎓</div>

        <h2 style="color:white;font-size:23px;">
            INDALA COLLEGE
        </h2>

        <p style="font-size:11px;color:#d3e4f5;">
            AI ADMISSION ASSISTANT
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.header("📁 Document Administration")

    st.caption(
        "Upload official college PDF documents "
        "and add them to the knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "💾 Save Uploaded PDFs",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning("Please select at least one PDF.")

        else:

            saved = save_uploaded_files(uploaded_files)

            st.success(
                f"Saved {len(saved)} PDF file(s)."
            )

    if st.button(
        "⚙️ Process Documents",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Extracting and processing PDF documents..."
            ):

                result = ingest_documents()

            st.success(
                f"Processing complete! "
                f"{result['stored_chunks']} chunks stored. "
                f"Database contains "
                f"{result['collection_count']} chunks."
            )

            if result["skipped_files"]:

                st.warning(
                    "No selectable text found in: "
                    + ", ".join(result["skipped_files"])
                    + ". These may be scanned PDFs and need OCR."
                )

        except Exception as exc:

            st.error(f"Processing failed: {exc}")

    st.divider()

    st.subheader("📊 Database Status")

    try:

        st.metric(
            "Stored Text Chunks",
            get_collection().count()
        )

    except Exception:

        st.metric("Stored Text Chunks", 0)

    st.caption(
        "Upload only official college documents."
    )


# --------------------------------------------------
# ADMISSION INFORMATION CARDS
# --------------------------------------------------

st.markdown("### Explore Admission Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="info-card">

        <div style="font-size:32px;">📚</div>

        <h4>Course Information</h4>

        <p>
            Discover available courses and
            academic programme details.
        </p>

    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="info-card">

        <div style="font-size:32px;">🎯</div>

        <h4>Eligibility Criteria</h4>

        <p>
            Understand course eligibility
            using official college documents.
        </p>

    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="info-card">

        <div style="font-size:32px;">📝</div>

        <h4>Admission Process</h4>

        <p>
            Learn about application procedures
            and required documents.
        </p>

    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


st.divider()

st.markdown("### 💬 Chat with AI Admission Assistant")

st.caption(
    "Ask your admission questions below. "
    "Answers are generated using official college documents."
)


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📄 View Source Documents"):

                for source in message["sources"]:

                    st.write(
                        f"**{source['source']}** "
                        f"— Page {source['page']}"
                    )


# --------------------------------------------------
# USER QUESTION AND ANSWER
# --------------------------------------------------

question = st.chat_input(
    "Ask about courses, fees, eligibility or admission..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

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
                    "🤖 Generating your answer..."
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

                        unique_sources.append(
                            {
                                "source": item["source"],
                                "page": item["page"]
                            }
                        )

                sources = unique_sources

            st.markdown(answer)

            if sources:

                with st.expander("📄 View Source Documents"):

                    for source in sources:

                        st.write(
                            f"**{source['source']}** "
                            f"— Page {source['page']}"
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

        except Exception as exc:

            error_message = (
                f"Unable to generate an answer: {exc}"
            )

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                }
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">

    🎓 <b>Indala College</b>

    <br><br>

    AI-Powered Admission Assistant

    <br>

    Powered by RAG Technology, ChromaDB and Groq AI

    <br><br>

    For official confirmation of admission details,
    please contact the Indala College Admission Office.

</div>
""", unsafe_allow_html=True)
