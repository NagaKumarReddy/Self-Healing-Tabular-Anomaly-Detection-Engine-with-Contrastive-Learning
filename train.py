"""
Training Pipeline for Customer Anomaly Detection using Self-Supervised Contrastive Learning.
Trains on Reference_Data sheet from uploaded dataset.
Computes normal manifold centroid and calibrates anomaly detection threshold.
"""

import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.dataset import CustomerTabularPreprocessor, create_dataloaders
from src.duckdb_manager import DuckDBManager
from src.model import ContrastiveAnomalyDetector, NTXentLoss, TabularAugmentor


EXCEL_DATASET_PATH = r"C:/Users/nagak/.gemini/antigravity/brain/ec534b71-3f39-4fdd-b019-9db15d71be12/.user_uploaded/media_1790233722382.xlsx"


def load_and_cache_datasets(excel_path: str = EXCEL_DATASET_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Reference_Data and Incoming_Data sheets and cache to data/ directory."""
    os.makedirs("data", exist_ok=True)
    ref_df = pd.read_excel(excel_path, sheet_name="Reference_Data")
    inc_df = pd.read_excel(excel_path, sheet_name="Incoming_Data")

    ref_df.to_csv("data/reference_data.csv", index=False)
    inc_df.to_csv("data/incoming_data.csv", index=False)
    print(f"Loaded {len(ref_df)} reference rows and {len(inc_df)} incoming rows.")
    return ref_df, inc_df


def train_customer_model(
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    temperature: float = 0.1,
    save_dir: str = "models",
    db_path: str = "data/anomaly_detection.duckdb"
):
    print("=" * 70)
    print("TRAINING CONTRASTIVE ANOMALY DETECTOR ON CUSTOMER DATASET")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device}")

    # Step 1: Load and Cache Reference Data
    ref_df, inc_df = load_and_cache_datasets()

    # Step 2: Fit Preprocessor
    print("\n[Data Pipeline] Fitting CustomerTabularPreprocessor on reference records...")
    preprocessor = CustomerTabularPreprocessor()
    X_train = preprocessor.fit_transform(ref_df)
    feature_names = preprocessor.feature_names
    input_dim = len(feature_names)
    print(f"Engineered input representation: {input_dim} features.")

    os.makedirs(save_dir, exist_ok=True)
    prep_path = os.path.join(save_dir, "preprocessor.joblib")
    preprocessor.save(prep_path)
    print(f"Saved preprocessor to {prep_path}")

    # Step 3: PyTorch DataLoader Setup
    augmentor = TabularAugmentor(corruption_rate=0.25, noise_std=0.04, mask_rate=0.1)
    train_loader = create_dataloaders(
        features=X_train,
        batch_size=batch_size,
        shuffle=True,
        augmentor=augmentor,
        is_train=True
    )
    calib_loader = create_dataloaders(
        features=X_train,
        batch_size=batch_size,
        shuffle=False,
        augmentor=None,
        is_train=False
    )

    # Step 4: Model Initialization
    model = ContrastiveAnomalyDetector(
        input_dim=input_dim,
        hidden_dim=128,
        latent_dim=64,
        proj_dim=32,
        num_layers=3,
        dropout_rate=0.1
    ).to(device)

    criterion = NTXentLoss(temperature=temperature)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Step 5: Training Loop
    print("\nStarting self-supervised contrastive pretraining...")
    model.train()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0

        for x_orig, x_aug in train_loader:
            x_orig = x_orig.to(device)
            x_aug = x_aug.to(device)

            optimizer.zero_grad()
            _, z_i = model(x_orig)
            _, z_j = model(x_aug)

            loss = criterion(z_i, z_j)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - NT-Xent Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

    # Step 6: Centroid & Threshold Calibration
    print("\n[Scoring Calibration] Computing normal centroid & threshold...")
    model.compute_centroid(calib_loader, device)
    model.calibrate_threshold(calib_loader, device, percentile=95.0)

    print(f"Computed Centroid Norm: {torch.norm(model.centroid).item():.4f}")
    print(f"Calibrated Normal Threshold (95th percentile): {model.threshold.item():.4f}")

    # Step 7: Save Model Checkpoint
    model_save_path = os.path.join(save_dir, "contrastive_anomaly_detector.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "feature_names": feature_names,
        "centroid": model.centroid.cpu(),
        "threshold": model.threshold.item(),
        "score_mean": model.score_mean.item(),
        "score_std": model.score_std.item(),
    }, model_save_path)
    print(f"Saved PyTorch model checkpoint to {model_save_path}")

    # Step 8: Initialize DuckDB
    db = DuckDBManager(db_path)
    metrics = {
        "Pretrain_NTXent_Loss": avg_loss,
        "Calibrated_Threshold": model.threshold.item(),
        "Num_Training_Samples": len(ref_df),
        "Input_Dimensions": input_dim
    }
    db.log_model_metrics("Customer-Contrastive-ResMLP-v1.0", metrics, dataset_used="Reference_Data")
    db.close()
    print("Logged baseline training parameters to DuckDB.")
    print("=" * 70)


if __name__ == "__main__":
    train_customer_model()
