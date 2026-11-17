from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = PROJECT_ROOT / "outputs/cyp2c19_abundance/position_split.csv"
METADATA_CSV = (
    PROJECT_ROOT / "outputs/esm2_cyp2c19_full/cyp2c19_embedding_metadata.csv"
)
SCORE_COLUMN = "experimental_abundance_score"


def main() -> None:
    split_assignments = pd.read_csv(SPLIT_CSV)
    embedding_metadata = pd.read_csv(METADATA_CSV)

    identity_columns = ["embedding_row", "variant_id"]
    if not split_assignments[identity_columns].equals(
        embedding_metadata[identity_columns]
    ):
        raise ValueError("Split and embedding metadata rows do not match")

    training_scores = embedding_metadata.loc[
        split_assignments["split"].eq("train"), SCORE_COLUMN
    ]
    validation_scores = embedding_metadata.loc[
        split_assignments["split"].eq("validation"), SCORE_COLUMN
    ]

    training_median = training_scores.median()
    validation_mae = (validation_scores - training_median).abs().mean()

    print(f"Training variants:   {len(training_scores):,}")
    print(f"Validation variants: {len(validation_scores):,}")
    print(f"Training median:     {training_median:.6f}")
    print(f"Validation MAE:      {validation_mae:.6f}")


if __name__ == "__main__":
    main()
