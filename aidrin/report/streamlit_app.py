import streamlit as st
import json
import pandas as pd
import sys

# Set page config for a premium wide layout
st.set_page_config(
    page_title="AIDRIN Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

def render_dashboard(report_path: str):
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except Exception as e:
        st.error(f"Failed to load report from {report_path}: {e}")
        return
        
    st.title("AIDRIN: AI Data Readiness Inspector")
    st.caption(f"**Source Dataset:** `{report.get('source', 'Unknown')}`")
    
    st.divider()
    
    # Premium hero metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Rows", value=f"{report.get('row_count', 0):,}")
    with col2:
        st.metric(label="Columns", value=f"{report.get('column_count', 0):,}")
    
    # Compute overall score if possible, else show static
    traditional_score = 100
    if "missing_values_ratio" in report.get("traditional_metrics", {}):
        missing_total = sum(report.get("traditional_metrics", {}).get("missing_values_ratio", {}).values())
        if missing_total > 0:
            traditional_score -= 10
            
    with col3:
        st.metric(label="Data Quality Score", value=f"{traditional_score}%", delta="High Confidence")
        
    with col4:
        fair_score = report.get("fair_compliance", {}).get("overall_compliance_score", 0)
        st.metric(label="FAIR Compliance", value=f"{fair_score*100:.1f}%")

    st.divider()
    
    # Interactive tabs with emojis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Traditional Quality", 
        "AI Readiness", 
        "Privacy & PII", 
        "FAIR Compliance", 
        "LLM Insights"
    ])
    
    with tab1:
        st.subheader("Data Quality Summary")
        t_metrics = report.get("traditional_metrics", {})
        
        # summary stats
        stats = t_metrics.get("summary_statistics", {})
        if stats:
            st.dataframe(pd.DataFrame(stats).T, use_container_width=True)
            
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("**Missing Values Ratio**")
            st.json(t_metrics.get("missing_values_ratio", {}))
        with col_right:
            st.write("**Outliers IQR**")
            st.json(t_metrics.get("outliers_iqr", {}))
        
    with tab2:
        st.subheader("AI-Specific Readiness")
        ai_metrics = report.get("ai_readiness_metrics", {})
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("**Feature Importance (SHAP)**")
            feat_imp = ai_metrics.get("feature_importance", {})
            if feat_imp:
                df_imp = pd.DataFrame(list(feat_imp.items()), columns=['Feature', 'SHAP Value'])
                st.bar_chart(df_imp.set_index('Feature'))
            else:
                st.info("No feature importance data found.")
                
            st.write("**Target Bias (TSD)**")
            st.json(ai_metrics.get("fairness_bias", {}))
                    
        with col_right:
            st.write("**Feature Correlations (Theil's U / Pearson)**")
            st.json(ai_metrics.get("feature_correlations", {}))
                
            st.write("**Class Imbalance (ID)**")
            st.json(ai_metrics.get("class_imbalance", {}))

    with tab3:
        st.subheader("Privacy & PII Detection")
        pii_metrics = report.get("privacy_metrics", {}).get("pii_detected", {})
        if pii_metrics:
            st.warning("PII detected in the dataset. Please review the details below.")
            df_pii = pd.DataFrame(list(pii_metrics.items()), columns=['Column', 'Detected Entities'])
            st.dataframe(df_pii, use_container_width=True)
        else:
             st.success("No PII detected.")
             st.json(report.get("privacy_metrics", {}))
        
    with tab4:
        st.subheader("FAIR Metadata Compliance")
        fair = report.get("fair_compliance", {})
        if "overall_compliance_score" in fair:
            st.progress(fair["overall_compliance_score"], text=f"Overall Compliance: {fair['overall_compliance_score']*100:.1f}%")
        st.json(fair)

    with tab5:
        st.subheader("Actionable Intelligence (LLM)")
        insights = report.get("llm_insights", "No insights available.")
        if "Skipped" in insights or "Error" in insights or "not set" in insights:
            st.error(insights)
        else:
            st.info("The following insights were generated asynchronously by your configured LLM (GPT-4o).")
            # Present insights in a nice container
            with st.container(border=True):
                st.markdown(insights)

if __name__ == "__main__":
    report_file = sys.argv[1] if len(sys.argv) > 1 else "aidrin_report.json"
    render_dashboard(report_file)
