from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = PROJECT_ROOT / "outputs/esm2_cyp2c19_full"
SPLIT_CSV = PROJECT_ROOT / "outputs/cyp2c19_abundance/position_split.csv"


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(42)

    split_assignments = pd.read_csv(SPLIT_CSV)
    embedding_metadata = pd.read_csv(
        EMBEDDING_DIR / "cyp2c19_embedding_metadata.csv"
    )
    embeddings = np.load(
        EMBEDDING_DIR / "cyp2c19_esm2_embeddings.npy", allow_pickle=False
    )

    identity_columns = ["embedding_row", "variant_id"]
    if not split_assignments[identity_columns].equals(
        embedding_metadata[identity_columns]
    ):
        raise ValueError("Split and embedding metadata are not aligned")
    if embeddings.shape != (len(embedding_metadata), 320):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    def load_partition(split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        rows = split_assignments["split"].eq(split_name).to_numpy()
        features = torch.from_numpy(embeddings[rows]).float()
        scores = torch.from_numpy(
            embedding_metadata.loc[rows, "experimental_abundance_score"]
            .to_numpy(dtype=np.float32)
        )
        return features, scores

    train_features, train_scores = load_partition("train")
    validation_features, validation_scores = load_partition("validation")

    # Scale both partitions using statistics from training only.
    feature_mean = train_features.mean(dim=0)
    feature_scale = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_features = (train_features - feature_mean) / feature_scale
    validation_features = (validation_features - feature_mean) / feature_scale

    train_loader = DataLoader(
        TensorDataset(train_features, train_scores),
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    training_median = float(np.median(train_scores.numpy()))
    model = nn.Linear(320, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(training_median)

    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def evaluate_validation_mae() -> float:
        model.eval()
        with torch.no_grad():
            predictions = model(validation_features).squeeze(-1)
            return loss_function(predictions, validation_scores).item()

    baseline_mae = evaluate_validation_mae()
    print(f"Epoch 0 | median baseline validation MAE: {baseline_mae:.6f}")
    
    best_validation_mae = float("inf")
    best_epoch = 0

    checkpoint_path = SPLIT_CSV.with_name("best_abundance_predictor.pt")

    for epoch in range(1, 21):
        model.train()

        for batch_features, batch_scores in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features).squeeze(-1)
            loss = loss_function(predictions, batch_scores)
            loss.backward()
            optimizer.step()

        validation_mae = evaluate_validation_mae()
        print(f"Epoch {epoch:2d} | validation MAE: {validation_mae:.6f}")
        
        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": feature_mean,
                    "feature_scale": feature_scale,
                    "epoch": best_epoch,
                    "validation_mae": best_validation_mae,
                },
                checkpoint_path,
            )
        
    print(f"Best validation MAE: {best_validation_mae:.6f} (epoch {best_epoch})")
    print(f"Baseline − best MAE: {baseline_mae - best_validation_mae:+.6f}")


if __name__ == "__main__":
    main()
