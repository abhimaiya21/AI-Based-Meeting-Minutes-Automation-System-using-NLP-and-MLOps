import streamlit as st
import requests
import json

st.set_page_config(page_title="AI Meeting System", layout="wide")
st.title("AI Meeting System — Frontend")

st.markdown("Upload an AMI XML transcript file to analyze meeting minutes using the backend API.")

backend_url = st.text_input("Backend Analyze URL", "http://127.0.0.1:8001/analyze")

uploaded = st.file_uploader("Upload AMI XML file", type=["xml"])

if uploaded:
    st.write(f"Selected file: {uploaded.name} ({uploaded.size} bytes)")
    if st.button("Analyze"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/xml")}
        try:
            with st.spinner("Sending file to backend..."):
                resp = requests.post(backend_url, files=files, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            st.success("Analysis completed")

            st.subheader("Summary")
            st.write(data.get("gemini_summary", "(no summary)"))

            st.subheader("Topics")
            st.write(data.get("gemini_topics", []))

            st.subheader("Action Items Detected")
            st.write(data.get("action_items_detected", []))

            st.subheader("Transcript Preview")
            st.code(data.get("full_text_reference", ""))

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except json.JSONDecodeError:
            st.error("Backend returned non-JSON response.")

st.markdown("---")
st.markdown("If the backend is not running, start it with `python app.py` (or `python main.py` for the other service).")
