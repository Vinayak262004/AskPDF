import streamlit as st
import requests
from io import BytesIO
import base64

API_URL = "http://127.0.0.1:8000"

# -----------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------
st.set_page_config(page_title="Ask Your PDF", layout="wide", initial_sidebar_state="collapsed")

# -----------------------------------------------------------
# Custom CSS (Dark Theme + Chat Bubbles + Sticky Input)
# -----------------------------------------------------------
st.markdown("""
    <style>

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark Theme Background */
    .main {
        background-color: #0f1117;
        color: white;
    }

    /* Chat container */
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 14px;
        margin-bottom: 10px;
        max-width: 85%;
        line-height: 1.5;
        font-size: 16px;
    }

    /* User bubble */
    .user-bubble {
        background-color: #1f6feb;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        color: white;
    }

    /* AI bubble */
    .ai-bubble {
        background-color: #161b22;
        border: 1px solid #30363d;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        color: #d0d7de;
    }

    /* Rounded avatar images */
    .avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        margin-right: 8px;
        object-fit: cover;
    }

    /* Chat row layout */
    .chat-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 12px;
    }

    /* Sticky Input Box */
    .sticky-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 15px 25px;
        background: #0f1117;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.4);
    }

    /* PDF viewer panel */
    .pdf-box {
        background: #161b22;
        border-radius: 10px;
        padding: 15px;
        height: 88vh;
        overflow-y: auto;
        border: 1px solid #30363d;
    }

    /* AI typing animation */
    @keyframes blink {
        0% { opacity: 0.2; }
        50% { opacity: 1; }
        100% { opacity: 0.2; }
    }
    .typing {
        display: inline-block;
    }
    .typing span {
        animation: blink 1.4s infinite both;
    }
    .typing span:nth-child(2) {animation-delay: 0.2s;}
    .typing span:nth-child(3) {animation-delay: 0.4s;}

    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Session State
# -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

# -----------------------------------------------------------
# LAYOUT → Two Columns: PDF Preview (left) + Chat (right)
# -----------------------------------------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("📄 PDF Preview")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        st.session_state.pdf_bytes = uploaded_pdf.getvalue()

        # Display PDF inside Streamlit
        base64_pdf = base64.b64encode(uploaded_pdf.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="750px"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

        if st.button("Process PDF"):
            files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
            response = requests.post(f"{API_URL}/upload-pdf", files=files)

            if response.ok:
                st.success("PDF successfully processed!")
            else:
                st.error("Processing failed. Check backend logs.")

with right_col:
    st.subheader("💬 Chat With Your PDF")

    # Display chat messages
    for msg in st.session_state.messages:
        avatar = "https://cdn-icons-png.flaticon.com/512/847/847969.png" if msg["role"] == "user" else "https://cdn-icons-png.flaticon.com/512/4712/4712104.png"
        bubble_class = "user-bubble" if msg["role"] == "user" else "ai-bubble"

        st.markdown(
            f"""
            <div class="chat-row">
                <img class="avatar" src="{avatar}"/>
                <div class="chat-bubble {bubble_class}">
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    # Input Box (sticky bottom)
    st.markdown("<div class='sticky-input'>", unsafe_allow_html=True)

    user_input = st.text_input("Ask something about the PDF...", key="chat_input")

    if st.button("Send"):
        if user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.spinner("AI is thinking..."):
                # Call backend
                response = requests.post(f"{API_URL}/ask", json={"question": user_input})

            if response.ok:
                answer = response.json().get("answer", "No answer returned.")
            else:
                answer = "⚠️ Backend error."

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()


    st.markdown("</div>", unsafe_allow_html=True)
