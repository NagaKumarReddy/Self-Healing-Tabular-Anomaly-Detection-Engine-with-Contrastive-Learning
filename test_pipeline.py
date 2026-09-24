"""
Integration and Unit Tests for Anomaly Detection Pipeline.
Covers:
- Task 4.3: Comprehensive Testing & Debugging
- Model forward pass, loss calculation, representations
- Data preprocessing & augmentations
- DuckDB persistence & query verification
- Anomaly scoring and explanation
"""

import os
import shutil
import unittest
import numpy as np
import pandas as pd
import torch

from src.dataset import DataPreprocessor, TabularContrastiveDataset, create_dataloaders
from src.duckdb_manager import DuckDBManager
from src.model import (
    ContrastiveAnomalyDetector,
    NTXentLoss,
    ProjectionHead,
    TabularAugmentor,
    TabularEncoder,
)


class TestContrastivePipeline(unittest.TestCase):

    def setUp(self):
        self.test_db_dir = "test_data"
        self.test_db_path = os.path.join(self.test_db_dir, "test.duckdb")
        os.makedirs(self.test_db_dir, exist_ok=True)
        self.input_dim = 8
        self.batch_size = 16

    def tearDown(self):
        if os.path.exists(self.test_db_dir):
            try:
                shutil.rmtree(self.test_db_dir)
            except Exception:
                pass

    def test_tabular_augmentor(self):
        """Test data augmentor generates perturbed views without altering shapes."""
        x = torch.randn(self.batch_size, self.input_dim)
        augmentor = TabularAugmentor(corruption_rate=0.3, noise_std=0.05, mask_rate=0.1)
        augmentor.fit_marginal(x)

        x_aug = augmentor.augment(x)
        self.assertEqual(x_aug.shape, x.shape)
        # Should not be strictly identical due to noise and corruption
        self.assertFalse(torch.equal(x, x_aug))

    def test_model_architecture(self):
        """Test encoder and projection head output shapes and normalization."""
        encoder = TabularEncoder(input_dim=self.input_dim, hidden_dim=32, latent_dim=16, num_layers=2)
        proj_head = ProjectionHead(latent_dim=16, proj_dim=8)

        x = torch.randn(self.batch_size, self.input_dim)
        h = encoder(x)
        self.assertEqual(h.shape, (self.batch_size, 16))

        z = proj_head(h)
        self.assertEqual(z.shape, (self.batch_size, 8))

        # Check L2 unit sphere normalization
        norms = torch.norm(z, p=2, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

    def test_ntxent_loss(self):
        """Test NT-Xent loss produces positive scalar."""
        loss_fn = NTXentLoss(temperature=0.1)
        z_i = torch.nn.functional.normalize(torch.randn(self.batch_size, 16), p=2, dim=1)
        z_j = torch.nn.functional.normalize(torch.randn(self.batch_size, 16), p=2, dim=1)

        loss = loss_fn(z_i, z_j)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_anomaly_scoring_and_explanation(self):
        """Test centroid distance computation, score bounds [0, 1], and gradient attribution."""
        model = ContrastiveAnomalyDetector(input_dim=self.input_dim, hidden_dim=32, latent_dim=16, proj_dim=8)

        # Fake normal data for calibration
        dummy_data = np.random.randn(50, self.input_dim).astype(np.float32)
        loader = create_dataloaders(dummy_data, batch_size=16, shuffle=False, is_train=False)

        device = torch.device("cpu")
        model.compute_centroid(loader, device)
        model.calibrate_threshold(loader, device, percentile=90.0)

        # Test scores
        test_x = torch.randn(10, self.input_dim)
        res = model.compute_anomaly_scores(test_x)

        scores = res["anomaly_score"]
        self.assertEqual(len(scores), 10)
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))

        # Test explanation
        feat_names = [f"f_{i}" for i in range(self.input_dim)]
        explanation = model.explain_anomaly(test_x[0], feat_names)
        self.assertEqual(len(explanation), self.input_dim)
        self.assertAlmostEqual(sum(explanation.values()), 100.0, places=2)

    def test_preprocessor(self):
        """Test preprocessing imputation, scaling, and serialization."""
        df = pd.DataFrame([{
            'Age': 40.0, 'Balance': 50000.0, 'Campaign': 2, 'Previous_Contact_Days': 10,
            'Transaction_Count': 18, 'Monthly_Income': 45000.0, 'Account_Tenure_Years': 5,
            'Credit_Score': 710.0, 'Job': 'admin', 'Education': 'secondary',
            'Housing': 'yes', 'Loan': 'no', 'Subscription_Status': 'subscribed'
        }])
        prep = DataPreprocessor()
        scaled = prep.fit_transform(df)

        self.assertEqual(scaled.shape, (1, 27))
        self.assertFalse(np.isnan(scaled).any())

        save_path = os.path.join(self.test_db_dir, "prep.joblib")
        prep.save(save_path)
        loaded_prep = DataPreprocessor.load(save_path)
        self.assertEqual(loaded_prep.feature_names, prep.feature_names)

    def test_duckdb_integration(self):
        """Test DuckDB table storage and analytical query retrieval."""
        db = DuckDBManager(self.test_db_path)

        # Test raw and preprocessed records
        df = pd.DataFrame({"cpu": [80.0], "mem": [4096.0]})
        raw_ids = db.insert_raw_records(df, source="unit_test")
        self.assertEqual(len(raw_ids), 1)

        db.insert_preprocessed_records(raw_ids, [{"cpu": 0.5, "mem": -0.2}])

        # Test anomaly logging
        anom_id = db.log_anomaly(
            data_id=raw_ids[0],
            anomaly_score=0.88,
            threshold_used=0.65,
            is_anomaly=True,
            features={"cpu": 80.0, "mem": 4096.0},
            top_contributing_feature="cpu"
        )
        self.assertIsNotNone(anom_id)

        # Test summary query
        summary = db.get_anomaly_summary()
        self.assertEqual(summary["total_evaluated"], 1)
        self.assertEqual(summary["total_anomalies"], 1)

        db.close()

    def test_self_healing(self):
        """Test autonomous self-healing on corrupted customer rows."""
        from src.dataset import CustomerTabularPreprocessor
        from src.self_healing import SelfHealingPipeline

        # Mock reference df
        ref_df = pd.DataFrame([{
            'Age': 35.0, 'Balance': 40000.0, 'Campaign': 2, 'Previous_Contact_Days': 10,
            'Transaction_Count': 15, 'Monthly_Income': 45000.0, 'Account_Tenure_Years': 5,
            'Credit_Score': 700.0, 'Job': 'admin', 'Education': 'tertiary',
            'Housing': 'yes', 'Loan': 'no', 'Subscription_Status': 'not_subscribed'
        }])
        prep = CustomerTabularPreprocessor().fit(ref_df)

        # Mock corrupted batch
        corrupted = pd.DataFrame([
            {
                'Customer_ID': 'CUST1', 'Age': -5.0, 'Balance': -100.0, 'Campaign': 1,
                'Previous_Contact_Days': 0, 'Transaction_Count': -10, 'Monthly_Income': np.nan,
                'Account_Tenure_Years': 2, 'Credit_Score': 1200.0, 'Job': 'unknown_job',
                'Education': 'tertiery', 'Housing': 'YES', 'Loan': 'no',
                'Last_Contact': 'not_a_date', 'Subscription_Status': 'not_subscribed'
            },
            {
                'Customer_ID': 'CUST1', 'Age': -5.0, 'Balance': -100.0, 'Campaign': 1,
                'Previous_Contact_Days': 0, 'Transaction_Count': -10, 'Monthly_Income': np.nan,
                'Account_Tenure_Years': 2, 'Credit_Score': 1200.0, 'Job': 'unknown_job',
                'Education': 'tertiery', 'Housing': 'YES', 'Loan': 'no',
                'Last_Contact': 'not_a_date', 'Subscription_Status': 'not_subscribed'
            }
        ])

        healer = SelfHealingPipeline(prep)
        res = healer.heal_dataset(corrupted)
        healed_df = res['healed_df']
        report = res['healing_report']

        # 1. Duplicates removed (from 2 down to 1)
        self.assertEqual(len(healed_df), 1)
        self.assertEqual(report['duplicates_removed'], 1)

        # 2. Typos repaired
        self.assertEqual(healed_df.iloc[0]['Education'], 'tertiary')
        self.assertEqual(healed_df.iloc[0]['Housing'], 'yes')

        # 3. Impossible values repaired
        self.assertGreaterEqual(healed_df.iloc[0]['Age'], 18.0)
        self.assertLessEqual(healed_df.iloc[0]['Credit_Score'], 850.0)
        self.assertGreaterEqual(healed_df.iloc[0]['Transaction_Count'], 0)
        self.assertGreaterEqual(healed_df.iloc[0]['Balance'], 0)
        self.assertFalse(pd.isna(healed_df.iloc[0]['Monthly_Income']))


if __name__ == "__main__":
    unittest.main()
