"""
Streamlit Web Application: Autonomous Self-Healing Anomaly Detection Platform.
"""

import json
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.duckdb_manager import DuckDBManager
from src.dataset import CustomerTabularPreprocessor
from src.model import ContrastiveAnomalyDetector
from src.self_healing import SelfHealingPipeline
from inference import CustomerAnomalyInferenceEngine


st.set_page_config(
    page_title="Self-Healing Anomaly Detection Studio",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/contrastive_anomaly_detector.pt"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
DB_PATH = "data/anomaly_detection.duckdb"


@st.cache_resource
def get_duckdb_manager():
    return DuckDBManager(DB_PATH)


@st.cache_resource
def load_pytorch_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = CustomerTabularPreprocessor.load(PREPROCESSOR_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    feature_names = checkpoint.get("feature_names", preprocessor.feature_names)
    input_dim = checkpoint.get("input_dim", len(feature_names))

    model = ContrastiveAnomalyDetector(
        input_dim=input_dim,
        hidden_dim=128,
        latent_dim=64,
        proj_dim=32,
        num_layers=3
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, preprocessor, feature_names


def main():
    db = get_duckdb_manager()
    model, preprocessor, feature_names = load_pytorch_model()

    st.sidebar.title("🛠️ Self-Healing Studio")
    st.sidebar.caption("PyTorch Contrastive Learning + DuckDB + Autonomous Remediation")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "⚡ Autonomous Self-Healing",
            "📊 System Dashboard",
            "🔍 Live Inference & Inspection",
            "⚠️ Schema Violations & Quality",
            "📈 Data Drift Monitoring",
            "🗄️ DuckDB Database Explorer",
            "✍️ User Feedback & Audit"
        ]
    )

    if model is None:
        st.error("⚠️ Model or Preprocessor not found! Please run `python train.py` first.")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training PyTorch Contrastive Model on Reference Data..."):
                from train import train_customer_model
                train_customer_model()
            st.success("Model trained successfully! Please reload the page.")
            st.rerun()
        return

    default_threshold = float(model.threshold.item())

    # -------------------------------------------------------------
    # 1. Autonomous Self-Healing (THE CORE WORKFLOW)
    # -------------------------------------------------------------
    if menu == "⚡ Autonomous Self-Healing":
        st.title("⚡ Autonomous Self-Healing Data Pipeline")
        st.markdown(
            "**Workflow:** Enter/upload an uncleaned dataset $\\to$ Identify anomalies & violations $\\to$ "
            "Execute automated self-healing $\\to$ Re-score and download pristine data."
        )

        st.subheader("Step 1: Enter or Upload Dataset")
        input_source = st.radio(
            "Select Dataset Source:",
            ["Use Injected Evaluation Batch (5,020 rows)", "Upload Custom CSV / Excel File"],
            horizontal=True
        )

        input_df = None
        if input_source == "Use Injected Evaluation Batch (5,020 rows)":
            if os.path.exists("data/incoming_data.csv"):
                input_df = pd.read_csv("data/incoming_data.csv")
                st.info(f"Loaded evaluation dataset: {len(input_df)} records with intentional anomalies, typos, nulls, and duplicates.")
            else:
                st.warning("data/incoming_data.csv not found. Run evaluate_incoming.py first.")
        else:
            uploaded = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])
            if uploaded is not None:
                try:
                    if uploaded.name.endswith(".csv"):
                        input_df = pd.read_csv(uploaded)
                    else:
                        xls = pd.ExcelFile(uploaded)
                        default_sheet_idx = 0
                        if "Incoming_Data" in xls.sheet_names:
                            default_sheet_idx = xls.sheet_names.index("Incoming_Data")
                        elif "Reference_Data" in xls.sheet_names:
                            default_sheet_idx = xls.sheet_names.index("Reference_Data")

                        chosen_sheet = st.selectbox(
                            "Select Sheet to Evaluate:",
                            xls.sheet_names,
                            index=default_sheet_idx
                        )
                        input_df = pd.read_excel(xls, sheet_name=chosen_sheet)

                    from src.validator import DataQualityValidator
                    input_df = DataQualityValidator().standardize_dataframe(input_df)
                    st.write(f"Loaded **{len(input_df)} rows** with columns: `{list(input_df.columns)[:8]}...`")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        if input_df is not None:
            # Store in session state
            if "raw_dataset" not in st.session_state or len(st.session_state.raw_dataset) != len(input_df):
                st.session_state.raw_dataset = input_df
                st.session_state.healed_dataset = None
                st.session_state.pre_eval = None
                st.session_state.post_eval = None

            st.dataframe(input_df.head(5), use_container_width=True)

            st.divider()
            st.subheader("Step 2: Anomaly & Violation Identification")

            if st.button("🔎 Scan & Identify Anomalies in Raw Dataset", type="secondary"):
                with st.spinner("Running PyTorch contrastive anomaly scoring and schema audit..."):
                    engine = CustomerAnomalyInferenceEngine(
                        model_path=MODEL_PATH,
                        preprocessor_path=PREPROCESSOR_PATH,
                        db_path=DB_PATH,
                        db=db
                    )
                    ref_df = pd.read_csv("data/reference_data.csv") if os.path.exists("data/reference_data.csv") else None
                    pre_eval = engine.evaluate_batch(
                        input_df,
                        reference_df=ref_df,
                        source="ui_pre_healing",
                        log_to_db=False
                    )
                    st.session_state.pre_eval = pre_eval

            if st.session_state.get("pre_eval") is not None:
                pre_eval = st.session_state.pre_eval
                audit = pre_eval["audit_report"]

                # Metric badges
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Flagged Anomalies", f"{pre_eval['anomalies_flagged']} ({pre_eval['anomaly_rate_percent']}%)")
                m2.metric("Duplicates", audit["duplicates_detected"])
                m3.metric("Missing Value Cells", sum(audit["missing_values"].values()))
                m4.metric("Category/Syntax Issues", len(audit["schema_violations"]))
                m5.metric("Date Errors", len(audit["date_violations"]))

                # Show detected issues
                st.write("**Identified Anomaly Details:**")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Category & Casing Typos Caught:**")
                    cat_issues = pd.DataFrame(audit["schema_violations"])
                    if not cat_issues.empty:
                        st.dataframe(cat_issues[["column", "value", "issue"]].drop_duplicates(), use_container_width=True)
                    else:
                        st.success("No category violations detected.")

                with c2:
                    st.markdown("**Domain Boundary & Numerical Outliers:**")
                    num_issues = pd.DataFrame(audit["domain_violations"])
                    if not num_issues.empty:
                        st.dataframe(num_issues[["column", "value", "issue"]].drop_duplicates(), use_container_width=True)
                    else:
                        st.success("No numerical boundary violations detected.")

                st.divider()
                st.subheader("Step 3: Autonomous Self-Healing")

                if st.button("🚀 Trigger Autonomous Self-Healing", type="primary"):
                    with st.spinner("Executing autonomous remediation pipeline (deduplication, imputation, domain clipping, typo repair)..."):
                        healer = SelfHealingPipeline(preprocessor, db=db)
                        ref_df = pd.read_csv("data/reference_data.csv") if os.path.exists("data/reference_data.csv") else None
                        heal_res = healer.heal_dataset(input_df, reference_df=ref_df)

                        healed_df = heal_res["healed_df"]
                        st.session_state.healed_dataset = healed_df
                        st.session_state.healing_report = heal_res["healing_report"]

                        # Re-score healed dataset
                        engine = CustomerAnomalyInferenceEngine(
                            model_path=MODEL_PATH,
                            preprocessor_path=PREPROCESSOR_PATH,
                            db_path=DB_PATH,
                            db=db
                        )
                        post_eval = engine.evaluate_batch(
                            healed_df,
                            reference_df=ref_df,
                            source="ui_post_healing",
                            log_to_db=True
                        )
                        st.session_state.post_eval = post_eval

                if st.session_state.get("healed_dataset") is not None:
                    healed_df = st.session_state.healed_dataset
                    report = st.session_state.healing_report
                    post_eval = st.session_state.post_eval

                    st.success(f"✅ Self-Healing Complete! {report['total_cells_healed']} cells repaired across {report['duplicates_removed']} duplicates and 10 attributes.")

                    # Before vs After Health Score Comparison
                    col_h1, col_h2, col_h3 = st.columns(3)
                    col_h1.metric("Data Health Before", f"{report['health_score_before']}%")
                    col_h2.metric("Data Health After Healing", f"{report['health_score_after']}%", delta=f"+{round(report['health_score_after'] - report['health_score_before'], 2)}%")
                    col_h3.metric("Remaining Anomalies", f"{post_eval['anomalies_flagged']} ({post_eval['anomaly_rate_percent']}%)")

                    # Side-by-side inspection
                    st.subheader("Step 4: Verified Edge Cases (Before vs After Healing)")
                    test_cases = ["CUST102824", "CUST103415", "CUST101451", "CUST102977", "CUST101887", "CUST102999", "CUST104233", "CUST102785"]
                    
                    diff_rows = []
                    for cid in test_cases:
                        r_orig = input_df[input_df["Customer_ID"] == cid]
                        r_heal = healed_df[healed_df["Customer_ID"] == cid]
                        if not r_orig.empty and not r_heal.empty:
                            score_after = post_eval["results_df"][post_eval["results_df"]["Customer_ID"] == cid]["anomaly_score"].values[0]
                            diff_rows.append({
                                "Customer_ID": cid,
                                "Raw Age": r_orig.iloc[0]["Age"],
                                "Healed Age": r_heal.iloc[0]["Age"],
                                "Raw Balance": r_orig.iloc[0]["Balance"],
                                "Healed Balance": r_heal.iloc[0]["Balance"],
                                "Raw Credit Score": r_orig.iloc[0]["Credit_Score"],
                                "Healed Credit Score": r_heal.iloc[0]["Credit_Score"],
                                "Raw Monthly Income": r_orig.iloc[0]["Monthly_Income"],
                                "Healed Income": r_heal.iloc[0]["Monthly_Income"],
                                "Post Anomaly Score": round(float(score_after), 4),
                                "Healthy": bool(score_after < default_threshold)
                            })

                    st.dataframe(pd.DataFrame(diff_rows), use_container_width=True)

                    # Detailed Healing Actions Log Table
                    st.subheader("Self-Healing Action Log (Sample)")
                    actions_df = pd.DataFrame(report["healing_actions"])
                    if not actions_df.empty:
                        st.dataframe(actions_df.head(100), use_container_width=True)

                    # Download clean healed dataset
                    csv_data = healed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Pristine Healed Dataset (CSV)",
                        data=csv_data,
                        file_name="customer_dataset_self_healed.csv",
                        mime="text/csv",
                        type="primary"
                    )

    # -------------------------------------------------------------
    # 2. System Dashboard
    # -------------------------------------------------------------
    elif menu == "📊 System Dashboard":
        st.title("📊 System Overview & Health Status")
        summary = db.get_anomaly_summary()
        healing_sum = db.get_healing_summary()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Inferences Logged", summary["total_evaluated"])
        col2.metric("Flagged Outliers", summary["total_anomalies"])
        col3.metric("Anomaly Rate", f"{summary['anomaly_rate']:.2f}%")
        col4.metric("Self-Healed Cells", healing_sum["total_healed_actions"])

        st.divider()
        st.subheader("Model Performance & Pretraining Baseline")
        perf_df = db.get_latest_metrics()
        if not perf_df.empty:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(perf_df, use_container_width=True)
            with c2:
                fig = px.bar(perf_df, x="metric_name", y="metric_value", color="metric_name", title="Model Calibration Parameters")
                st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # 3. Live Inference & Inspection
    # -------------------------------------------------------------
    elif menu == "🔍 Live Inference & Inspection":
        st.title("🔍 Single Profile Anomaly Scorer")
        st.markdown("Inspect anomaly scores and attribution explanations on a custom customer profile.")

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", value=35, min_value=-10, max_value=1000)
            job = st.selectbox("Job", CustomerTabularPreprocessor.JOB_CATEGORIES + ["unknown_job"])
            education = st.selectbox("Education", CustomerTabularPreprocessor.EDUCATION_CATEGORIES + ["tertiery"])
            housing = st.selectbox("Housing Loan", ["no", "yes", "YES"])
        with c2:
            balance = st.number_input("Balance ($)", value=45000, step=1000)
            income = st.number_input("Monthly Income ($)", value=50000, step=1000)
            credit_score = st.number_input("Credit Score", value=710, min_value=0, max_value=1500)
            loan = st.selectbox("Personal Loan", ["no", "yes"])
        with c3:
            tx_count = st.number_input("Transaction Count", value=18, min_value=-50, max_value=1000)
            tenure = st.number_input("Account Tenure (Years)", value=5, min_value=0, max_value=50)
            campaign = st.number_input("Campaign", value=2)
            pdays = st.number_input("Previous Contact Days", value=14)

        if st.button("Evaluate Profile", type="primary"):
            engine = CustomerAnomalyInferenceEngine(
                model_path=MODEL_PATH,
                preprocessor_path=PREPROCESSOR_PATH,
                db_path=DB_PATH,
                db=db
            )
            sample = pd.DataFrame([{
                "Customer_ID": "TEST_CUST", "Age": age, "Job": job, "Education": education,
                "Balance": balance, "Housing": housing, "Loan": loan, "Campaign": campaign,
                "Previous_Contact_Days": pdays, "Transaction_Count": tx_count,
                "Monthly_Income": income, "Account_Tenure_Years": tenure,
                "Credit_Score": credit_score, "Last_Contact": "2026-06-15 00:00:00",
                "Subscription_Status": "not_subscribed"
            }])
            res = engine.evaluate_batch(sample, log_to_db=False)
            res_df = res["results_df"]
            score = float(res_df.iloc[0]["anomaly_score"])
            is_anom = bool(res_df.iloc[0]["is_anomaly"])
            top_cause = res_df.iloc[0]["top_contributing_feature"]

            st.metric("Anomaly Probability Score", f"{score:.4f}", delta="ANOMALOUS" if is_anom else "NORMAL", delta_color="inverse" if is_anom else "normal")
            st.info(f"Primary Anomaly Trigger: **{top_cause}** (Threshold: {default_threshold:.3f})")

    # -------------------------------------------------------------
    # 4. Schema Violations & Quality
    # -------------------------------------------------------------
    elif menu == "⚠️ Schema Violations & Quality":
        st.title("⚠️ Schema Violations & Data Quality Log")
        st.markdown("Records of illegal categories (`tertiery`, `unknown_job`, `YES`), malformed dates, and impossible domain bounds.")
        v_df = db.get_schema_violations_df()
        st.metric("Total Schema Violations Logged", len(v_df))
        if not v_df.empty:
            st.dataframe(v_df, use_container_width=True)

    # -------------------------------------------------------------
    # 5. Data Drift Monitoring
    # -------------------------------------------------------------
    elif menu == "📈 Data Drift Monitoring":
        st.title("📈 Feature Distribution Drift Monitoring")
        drift_df = db.get_data_drift_df()
        if not drift_df.empty:
            st.dataframe(drift_df, use_container_width=True)
            fig = px.bar(
                drift_df, x="feature_name", y="metric_value", color="is_drifted",
                title="Wasserstein Distance vs Reference Baseline (Drift Alert)",
                color_discrete_map={True: "crimson", False: "seagreen"}
            )
            fig.add_hline(y=0.15, line_dash="dash", line_color="black", annotation_text="Drift Threshold (0.15)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No drift records found.")

    # -------------------------------------------------------------
    # 6. DuckDB Database Explorer
    # -------------------------------------------------------------
    elif menu == "🗄️ DuckDB Database Explorer":
        st.title("🗄️ DuckDB Database Explorer")
        tbl = st.selectbox("Select Table:", ["self_healing_log", "anomalies", "schema_violations", "data_drift", "raw_data"])
        query_df = db.conn.execute(f"SELECT * FROM {tbl} ORDER BY timestamp DESC LIMIT 300").fetchdf()
        st.dataframe(query_df, use_container_width=True)

    # -------------------------------------------------------------
    # 7. User Feedback & Audit
    # -------------------------------------------------------------
    elif menu == "✍️ User Feedback & Audit":
        st.title("✍️ User Feedback & Retraining Audit")
        fb_df = db.get_feedback_df()
        st.dataframe(fb_df, use_container_width=True)


if __name__ == "__main__":
    main()
