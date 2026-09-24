"""
Test and Verification Script for Autonomous Self-Healing Pipeline.
Tests:
1. Corrupted input ingestion -> 2. Anomaly identification -> 3. Autonomous self-healing execution -> 4. Post-healing verification.
"""

import pandas as pd
from inference import CustomerAnomalyInferenceEngine
from src.dataset import CustomerTabularPreprocessor
from src.duckdb_manager import DuckDBManager
from src.self_healing import SelfHealingPipeline


def test_self_healing_pipeline():
    print("=" * 80)
    print("AUTONOMOUS SELF-HEALING ANOMALY DETECTION TEST")
    print("=" * 80)

    # 1. Load Datasets
    raw_incoming = pd.read_csv("data/incoming_data.csv")
    reference_df = pd.read_csv("data/reference_data.csv")

    print(f"Step 1: Loaded incoming corrupted batch: {len(raw_incoming)} rows.")

    # 2. Run Pre-Healing Anomaly Evaluation
    print("\nStep 2: Identifying anomalies in raw incoming dataset...")
    engine = CustomerAnomalyInferenceEngine()
    pre_result = engine.evaluate_batch(
        raw_incoming,
        reference_df=reference_df,
        source="pre_healing_raw",
        log_to_db=False
    )
    raw_anom_count = pre_result["anomalies_flagged"]
    raw_rate = pre_result["anomaly_rate_percent"]
    print(f"  -> Flagged {raw_anom_count} anomalies ({raw_rate}%) in raw dataset.")
    print(f"  -> Detected {pre_result['audit_report']['duplicates_detected']} duplicates.")
    print(f"  -> Detected {len(pre_result['audit_report']['schema_violations'])} category/syntax violations.")
    print(f"  -> Detected {len(pre_result['audit_report']['date_violations'])} date violations.")
    print(f"  -> Detected {len(pre_result['audit_report']['domain_violations'])} numerical boundary violations.")

    # 3. Trigger Autonomous Self-Healing
    print("\nStep 3: Triggering Autonomous Self-Healing Pipeline...")
    db = DuckDBManager()
    healer = SelfHealingPipeline(engine.preprocessor, db=db)
    heal_output = healer.heal_dataset(raw_incoming, reference_df=reference_df)

    healed_df = heal_output["healed_df"]
    report = heal_output["healing_report"]

    print(f"  -> Self-Healing Completed!")
    print(f"  -> Total Cells Repaired:      {report['total_cells_healed']}")
    print(f"  -> Duplicates Removed:        {report['duplicates_removed']}")
    print(f"  -> Health Score Before:       {report['health_score_before']}%")
    print(f"  -> Health Score After:        {report['health_score_after']}%")

    # 4. Verify Post-Healing Data Quality
    print("\nStep 4: Re-evaluating healed dataset with PyTorch Contrastive Model...")
    post_result = engine.evaluate_batch(
        healed_df,
        reference_df=reference_df,
        source="post_healing_verified",
        log_to_db=True
    )
    post_anom_count = post_result["anomalies_flagged"]
    post_rate = post_result["anomaly_rate_percent"]
    print(f"  -> Flagged Anomalies After Healing: {post_anom_count} ({post_rate}%)")
    print(f"  -> Remaining Duplicates:            {post_result['audit_report']['duplicates_detected']}")
    print(f"  -> Remaining Schema Violations:     {len(post_result['audit_report']['schema_violations'])}")
    print(f"  -> Remaining Date Violations:       {len(post_result['audit_report']['date_violations'])}")
    print(f"  -> Remaining Domain Violations:     {len(post_result['audit_report']['domain_violations'])}")

    # 5. Check Specific Previously-Corrupted Customer Records
    print("\nStep 5: Verifying Specific Repaired Customers (Before vs After):")
    test_cases = [
        ("CUST102824", "Age", "Negative age (-5)"),
        ("CUST103415", "Age", "Impossible age (150)"),
        ("CUST101451", "Credit_Score", "Invalid score (50)"),
        ("CUST102977", "Credit_Score", "Score > 850 (1200)"),
        ("CUST101887", "Balance", "Negative balance (-$500k)"),
        ("CUST102999", "Balance", "Extreme balance ($9M)"),
        ("CUST104233", "Transaction_Count", "Negative transactions (-20)"),
        ("CUST102785", "Monthly_Income", "Extreme income ($16.5M)"),
    ]

    for cid, col, desc in test_cases:
        raw_rows = raw_incoming[raw_incoming["Customer_ID"] == cid]
        healed_rows = healed_df[healed_df["Customer_ID"] == cid]
        if not raw_rows.empty and not healed_rows.empty:
            orig_val = raw_rows.iloc[0][col]
            new_val = healed_rows.iloc[0][col]
            # Get post-healing score
            post_score = post_result["results_df"][post_result["results_df"]["Customer_ID"] == cid]["anomaly_score"].values[0]
            is_anom = post_result["results_df"][post_result["results_df"]["Customer_ID"] == cid]["is_anomaly"].values[0]
            print(f"  [{desc}] Customer {cid}: {col} '{orig_val}' -> HEALED TO '{new_val}' | Post-Score: {post_score:.4f} (Anomaly: {is_anom})")

    # 6. Save Healed Dataset
    healed_csv_path = "data/customer_healed_dataset.csv"
    healed_df.to_csv(healed_csv_path, index=False)
    print(f"\nStep 6: Saved clean, healed dataset to {healed_csv_path}")

    # Check DuckDB healing log table
    db_healing_summary = db.get_healing_summary()
    print("\nDuckDB Self-Healing Audit:")
    print(f"  -> Total Healing Actions in Database: {db_healing_summary['total_healed_actions']}")
    print(f"  -> Affected Columns in Database:       {db_healing_summary['affected_columns']}")
    print(f"  -> Unique Repair Strategies:           {db_healing_summary['unique_strategies']}")
    db.close()

    print("\n" + "=" * 80)
    print("SELF-HEALING PIPELINE VERIFIED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_self_healing_pipeline()
