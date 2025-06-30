import streamlit as st
import requests

st.title("PDF QA with AstraDB & LangChain (FastAPI Backend)")

# PDF upload section
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Uploading and processing PDF..."):
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        try:
            resp = requests.post("http://localhost:7000/upload_pdf", files=files)
            if resp.status_code == 200:
                st.success("PDF uploaded and processed successfully!")
            else:
                st.error(f"Upload failed: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.error(f"Could not upload PDF: {e}")

# Fetch and display PDF summary
summary = None
try:
    resp = requests.get("http://localhost:7000/summary")
    if resp.status_code == 200:
        summary = resp.json().get("summary", "")
except Exception as e:
    summary = None
    st.warning(f"Could not fetch summary: {e}")

if summary:
    st.markdown("### PDF Summary")
    st.info(summary)

st.markdown("""
Ask questions about your PDF.  
The backend is powered by FastAPI and LangChain.
""")

question = st.text_input("Enter your question:")

if st.button("Ask") and question.strip():
    with st.spinner("Getting answer..."):
        try:
            response = requests.post(
                "http://localhost:7000/ask",
                json={"question": question}
            )
            if response.status_code == 200:
                data = response.json()
                st.subheader("Answer")
                st.write(data.get("answer", "No answer returned."))
                st.subheader("Top Documents")
                for idx, doc in enumerate(data.get("documents", []), 1):
                    st.markdown(f"**{idx}. (Score: {doc['score']:.4f})**")
                    st.code(doc["content"])
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
