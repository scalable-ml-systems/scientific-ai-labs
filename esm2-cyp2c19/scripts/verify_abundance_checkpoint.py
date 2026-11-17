from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABUNDANCE_DIR = PROJECT_ROOT / "outputs/cyp2c19_abundance"
EMBEDDING_DIR = PROJECT_ROOT / "outputs/esm2_cyp2c19_full"


def main() -> None:
    torch.set_num_threads(2)

    split_assignments = pd.read_csv(ABUNDANCE_DIR / "position_split.csv")
    embedding_metadata = pd.read_csv(
        EMBEDDING_DIR / "cyp2c19_embedding_metadata.csv"
    )
    identity_columns = ["embedding_row", "variant_id"]
    if not split_assignments[identity_columns].equals(
        embedding_metadata[identity_columns]
    ):
        raise ValueError("Split and embedding metadata are not aligned")

    validation_mask = split_assignments["split"].eq("validation").to_numpy()
    embeddings = np.load(
        EMBEDDING_DIR / "cyp2c19_esm2_embeddings.npy", allow_pickle=False
    )
    if embeddings.shape != (len(embedding_metadata), 320):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    validation_features = torch.from_numpy(embeddings[validation_mask]).float()
    validation_scores = torch.from_numpy(
        embedding_metadata.loc[validation_mask, "experimental_abundance_score"]
        .to_numpy(dtype=np.float32)
    )

    checkpoint = torch.load(
        ABUNDANCE_DIR / "best_abundance_predictor.pt",
        map_location="cpu",
        weights_only=True,
    )

    # Use the exact training statistics saved with the model.
    validation_features = (
        validation_features - checkpoint["feature_mean"]
    ) / checkpoint["feature_scale"]

    model = nn.Linear(320, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        predictions = model(validation_features).squeeze(-1)
        reloaded_mae = (predictions - validation_scores).abs().mean().item()

    saved_mae = checkpoint["validation_mae"]
    difference = abs(reloaded_mae - saved_mae)

    print(f"Saved epoch:              {checkpoint['epoch']}")
    print(f"Validation variants:      {len(validation_scores):,}")
    print(f"Saved validation MAE:    {saved_mae:.6f}")
    print(f"Reloaded validation MAE:  {reloaded_mae:.6f}")

    if not difference <= 1e-6:
        raise RuntimeError(f"Checkpoint MAE mismatch: {difference:.9g}")

    print("Checkpoint reload: PASS")


if __name__ == "__main__":
    main()
