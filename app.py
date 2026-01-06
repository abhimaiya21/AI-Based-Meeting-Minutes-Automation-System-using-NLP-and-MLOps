import os
import streamlit as st
import requests
import pandas as pd
import json

# --- 1. CONFIGURATION ---
# Default to localhost for local dev, override with env var in Docker
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="AI Meeting Manager", 
    layout="wide", 
    page_icon="🧠"
)

# Initialize Session State
if 'meeting_context' not in st.session_state:
    st.session_state['meeting_context'] = ""

# --- 2. MAIN APP UI ---
st.title("🧠 Hybrid AI Meeting Dashboard")

# Update status to reflect 2025 technology stack
st.markdown(f"**System Status:** Connected to [`{API_URL}`]({API_URL}/docs) | **Models:** Gemini 3 Flash + AIF360 Governance")

# Sidebar for Upload & Real-time Governance Watchtower
with st.sidebar:
    st.header("📁 Upload Meeting Data")
    uploaded_file = st.file_uploader("Choose an XML transcript...", type=['xml'])
    analyze_button = st.button("Analyze Meeting", type="primary")
    
    st.divider()
    st.header("⚖️ Governance Watchtower")
    
    # Fetch real governance metrics from backend API
    try:
        gov_response = requests.get(f"{API_URL}/governance")
        if gov_response.status_code == 200:
            gov_stats = gov_response.json()
        else:
            gov_stats = None
    except Exception:
        gov_stats = None
    
    if gov_stats and "error" not in gov_stats:
        # Metrics
        di = gov_stats.get("disparate_impact", 0.0)
        parity = gov_stats.get("statistical_parity_difference", 0.0)
        drift_detected = gov_stats.get("drift_detected", False)
        
        # Strict Threshold Logic
        if di < 0.8:
            status_color = "red"
            status_msg = "CRITICAL: Data Drift"
        elif di > 1.2:
            status_color = "red"
            status_msg = "CRITICAL: Data Leakage"
        else:
            status_color = "green"
            status_msg = "Governance Compatible"
            
        st.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold'>{status_msg}</span>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        c1.metric("Disparate Impact", f"{di:.3f}")
        c2.metric("Stat. Parity", f"{parity:.3f}")
        
        if di < 0.8:
            st.error("⚠️ **Drift Warning**: DI < 0.8 indicates feature distribution mismatch. Model predictions may be unreliable.")
        elif di > 1.2:
            st.error("⚠️ **Leakage Warning**: DI > 1.2 indicates potential overfitting or data leakage. Evaluation is unrealistic.")
        else:
            st.success("✅ System is Fair and Stable.")
            
        if drift_detected:
            st.warning("📉 Data Drift Detected via Evidently")
            
    else:
        st.warning("[*] Governance data unavailable. Run monitor pipeline.")

# Main Content Logic
if uploaded_file and analyze_button:
    with st.spinner("Executing Hybrid Pipeline (ML + Generative AI)..."):
        try:
            files = {'file': (uploaded_file.name, uploaded_file, 'text/xml')}
            response = requests.post(f"{API_URL}/analyze", files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.success("Analysis Complete!")
                
                # Update context for the chatbot
                st.session_state['meeting_context'] = data.get('full_text_reference', data.get('full_text', ''))

                # --- DASHBOARD TABS ---
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Executive Summary", 
                    "📈 ML Analytics", 
                    "⚖️ Governance Audit", # Integrated Phase 2 Tab
                    "✅ Action Items", 
                    "💾 Export", 
                    "💬 Chat with Data"
                ])
                
                # --- TAB 1: GEMINI SUMMARY ---
                with tab1:
                    st.subheader("📝 Executive Summary (Gemini 3 Flash)")
                    st.info(data.get('gemini_summary', "No summary available."))
                    
                    st.subheader("🏷️ Key Topics")
                    topics = data.get('gemini_topics', [])
                    if topics:
                        tags_html = "".join([f"<span style='background-color:#e1ecf4; padding:5px 10px; border-radius:15px; margin-right:5px; color:#333; display:inline-block; margin-bottom:5px;'>{topic}</span>" for topic in topics])
                        st.markdown(tags_html, unsafe_allow_html=True)

                # --- TAB 2: ML ANALYTICS ---
                with tab2:
                    st.subheader("🤖 Machine Learning Model Predictions")
                    preds = data.get('ml_predictions', {})
                    if preds:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Linear Regression", preds.get('Linear_Regression', 0))
                        c2.metric("Random Forest", preds.get('Random_Forest', 0))
                        c3.metric("XGBoost", preds.get('XGBoost', 0))
                    
                    st.divider()
                    st.subheader("🔍 Explainability Features")
                    feats = data.get('ml_features', {})
                    if feats:
                        c4, c5, c6 = st.columns(3)
                        c4.info(f"Words: {feats.get('word_count', 0)}")
                        c5.success(f"Positive: {feats.get('positive_count', 0)}")
                        c6.error(f"Negative: {feats.get('negative_count', 0)}")

                # --- TAB 3: GOVERNANCE AUDIT (NEW PHASE 2) ---
                with tab3:
                    st.subheader("⚖️ Real-Time Governance Audit")
                    
                    if gov_stats and "error" not in gov_stats:
                        di = gov_stats.get("disparate_impact", 0.0)
                        parity = gov_stats.get("statistical_parity_difference", 0.0)
                        
                        # Visuals
                        col_a, col_b = st.columns(2)
                        
                        is_safe = 0.8 <= di <= 1.2
                        
                        col_a.metric(
                            label="Disparate Impact (Real)",
                            value=f"{di:.4f}",
                            delta="Pass" if is_safe else "Fail",
                            delta_color="normal" if is_safe else "inverse"
                        )
                        
                        col_b.metric(
                            label="Statistical Parity",
                            value=f"{parity:.4f}"
                        )
                        
                        st.divider()
                        
                        # Detailed Interpretation
                        st.markdown("### 🔍 Diagnostic Report")
                        if di < 0.8:
                            st.error(f"""
                            **Violation: Data Drift / Bias (DI = {di:.4f})**
                            
                            The Disparate Impact is below 0.8. This suggests that the model is treating the unprivileged group significantly worse than the privileged group, or that the input data distribution has drifted significantly from the training set.
                            
                            **Risk:** Unreliable predictions, potential discrimination.
                            **Action:** Retrain model with balanced data.
                            """)
                        elif di > 1.2:
                            st.error(f"""
                            **Violation: Data Leakage / Overfitting (DI = {di:.4f})**
                            
                            The Disparate Impact is abnormally high (> 1.2). In a governance context, this often indicates **Data Leakage** — where the model has accidentally "memorized" the target variable or protected attributes, leading to unrealistically perfect (or inverted) fairness scores.
                            
                            **Risk:** Model is overfitted and will fail in production.
                            **Action:** Investigate feature engineering for leakage.
                            """)
                        else:
                            st.success(f"""
                            **Governance Passed (DI = {di:.4f})**
                            
                            The metrics are within the safe trust boundary (0.8 - 1.2).
                            ✓ No significant Drift
                            ✓ No Evidence of Leakage
                            """)
                            
                        # Show raw JSON for audit
                        with st.expander("View Raw Audit JSON"):
                            st.json(gov_stats)
                            
                    else:
                         st.warning("Governance data not loaded. Check backend connection.")

                # --- TAB 4: ACTION ITEMS ---
                with tab4:
                    st.subheader("✅ Detected Action Items (spaCy NER)")
                    actions = data.get('action_items', data.get('action_items_detected', []))
                    if actions:
                        st.table(pd.DataFrame(actions, columns=["Meeting Context / Tasks"]))
                    else:
                        st.info("No specific action items detected.")

                # --- TAB 5: EXPORT ---
                with tab5:
                    st.subheader("💾 Data Portability")
                    st.download_button("Download JSON Audit Report", data=json.dumps(data, indent=2), file_name="meeting_report.json")

                # --- TAB 6: CHATBOT ---
                with tab6:
                    st.subheader("💬 Query Meeting Context")
                    user_question = st.text_input("Example: 'What were the software tools mentioned?'")
                    if st.button("Query Gemini 3 Flash"):
                        if st.session_state['meeting_context']:
                            chat_payload = {"context_text": st.session_state['meeting_context'], "question": user_question}
                            with st.spinner("AI is thinking..."):
                                chat_res = requests.post(f"{API_URL}/chat", json=chat_payload)
                                if chat_res.status_code == 200:
                                    st.markdown(f"**🤖 Answer:** {chat_res.json().get('bot_answer')}")
                        else:
                            st.error("Analyze a file first to provide context.")
            else:
                st.error(f"Backend API Error: {response.status_code}")
        except Exception as e:
            st.error(f"System Error: {e}")

# Persistence check
elif st.session_state['meeting_context'] and not analyze_button:
    st.info("Previous session active. Access the Chat tab to continue.")