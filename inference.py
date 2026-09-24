"""
Customer Anomaly Detection Inference Engine.
Integrates:
- CustomerTabularPreprocessor
- DataQualityValidator (schema, categories, dates, drift, duplicates)
- PyTorch Contrastive Model Anomaly Scoring & Attribution
- DuckDB Persistence
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from src.dataset import CustomerTabularPreprocessor
from src.duckdb_manager import DuckDBManager
from src.model import ContrastiveAnomalyDetector
from src.validator import DataQualityValidator


class CustomerAnomalyInferenceEngine:
    """Production inference engine for Customer dataset."""

    def __init__(
        self,
        model_path: str = "models/contrastive_anomaly_detector.pt",
        preprocessor_path: str = "models/preprocessor.joblib",
        db_path: str = "data/anomaly_detection.duckdb",
        db: Optional[DuckDBManager] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.db = db if db is not None else DuckDBManager(db_path)
        self.validator = DataQualityValidator(self.db)

        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run train.py first.")
        self.preprocessor = CustomerTabularPreprocessor.load(preprocessor_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run train.py first.")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.feature_names = checkpoint.get("feature_names", self.preprocessor.feature_names)
        input_dim = checkpoint.get("input_dim", len(self.feature_names))

        self.model = ContrastiveAnomalyDetector(
            input_dim=input_dim,
            hidden_dim=128,
            latent_dim=64,
            proj_dim=32,
            num_layers=3
        ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.default_threshold = float(checkpoint.get("threshold", 0.5))

    def evaluate_batch(
        self,
        df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        custom_threshold: Optional[float] = None,
        source: str = "incoming_batch",
        log_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation on the dataset:
        1. Validates duplicates, missing values, categories, dates, and domain boundaries.
        2. Detects feature distribution drift against reference data.
        3. Preprocesses data and runs PyTorch model inference.
        4. Calculates anomaly scores and feature attributions.
        5. Logs results to DuckDB tables.
        """
        # Step 1: Data Quality & Schema Audit
        validation_output = self.validator.validate_and_clean_batch(df, reference_df=reference_df)
        clean_df = validation_output["clean_df"]
        audit_report = validation_output["audit_report"]

        # Step 2: Transform with Preprocessor
        X_scaled = self.preprocessor.transform(clean_df)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # Step 3: PyTorch Model Anomaly Scoring
        threshold_to_use = custom_threshold if custom_threshold is not None else self.default_threshold
        scoring = self.model.compute_anomaly_scores(tensor_x, custom_threshold=threshold_to_use)

        scores = scoring["anomaly_score"]
        raw_dists = scoring["raw_distance"]
        is_anom = scoring["is_anomaly"]

        # Step 4: Explain Flagged Anomalies
        top_features = []
        for i in range(len(clean_df)):
            if is_anom[i]:
                exp = self.model.explain_anomaly(tensor_x[i], self.feature_names)
                # Map one-hot feature back to primary column name
                top_feat = list(exp.keys())[0]
                col_name = top_feat.split("_")[0] if "_" in top_feat else top_feat
                top_features.append(col_name)
            else:
                top_features.append("None")

        results_df = clean_df.copy()
        results_df["anomaly_score"] = scores
        results_df["raw_distance"] = raw_dists
        results_df["is_anomaly"] = is_anom
        results_df["top_contributing_feature"] = top_features

        # Step 5: Persist to DuckDB
        if log_to_db:
            raw_ids = self.db.insert_raw_records(clean_df, source=source)
            scaled_dicts = [
                {feat: float(val) for feat, val in zip(self.feature_names, X_scaled[i])}
                for i in range(len(clean_df))
            ]
            self.db.insert_preprocessed_records(raw_ids, scaled_dicts)

            for i in range(len(clean_df)):
                self.db.log_anomaly(
                    data_id=raw_ids[i],
                    anomaly_score=float(scores[i]),
                    threshold_used=float(threshold_to_use),
                    is_anomaly=bool(is_anom[i]),
                    features=clean_df.iloc[i].to_dict(),
                    top_contributing_feature=top_features[i]
                )

        return {
            "results_df": results_df,
            "audit_report": audit_report,
            "total_evaluated": len(results_df),
            "anomalies_flagged": int(is_anom.sum()),
            "anomaly_rate_percent": round(float(is_anom.sum() / len(results_df) * 100.0), 2)
        }
