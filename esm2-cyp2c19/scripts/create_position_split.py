import hashlib
import json
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "outputs/esm2_cyp2c19_full"
VALIDATED_CSV = PROJECT_ROOT / "data/processed/cyp2c19_validated_missense.csv"
METADATA_CSV = ARTIFACT_DIR / "cyp2c19_embedding_metadata.csv"
SPLIT_CSV = PROJECT_ROOT / "outputs/cyp2c19_abundance/position_split.csv"
SPLIT_SEED = 42
EXPECTED_VARIANTS = 7830


def sha256_file(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = ARTIFACT_DIR / "cyp2c19_embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if sha256_file(VALIDATED_CSV) != manifest["validated_csv_sha256"]:
        raise ValueError("Validated CSV does not match the Release 1 manifest")
    if sha256_file(METADATA_CSV) != manifest["artifact_sha256"][METADATA_CSV.name]:
        raise ValueError("Embedding metadata does not match the Release 1 manifest")

    variant_positions = pd.read_csv(
        VALIDATED_CSV,
        usecols=["accession", "position_1based"],
        dtype={"position_1based": "int64"},
    )
    embedding_metadata = pd.read_csv(
        METADATA_CSV, usecols=["embedding_row", "variant_id"]
    )

    if embedding_metadata["embedding_row"].tolist() != list(range(EXPECTED_VARIANTS)):
        raise ValueError("Missing or out-of-order embedding rows")
    if not embedding_metadata["variant_id"].is_unique:
        raise ValueError("Duplicate variant IDs in embedding metadata")
    if embedding_metadata["variant_id"].tolist() != variant_positions["accession"].tolist():
        raise ValueError("Variant IDs do not match in embedding order")

    residue_positions = sorted(variant_positions["position_1based"].unique())
    random.Random(SPLIT_SEED).shuffle(residue_positions)

    train_end = int(0.70 * len(residue_positions))
    validation_end = train_end + int(0.15 * len(residue_positions))

    positions_by_split = {
        "train": residue_positions[:train_end],
        "validation": residue_positions[train_end:validation_end],
        "test": residue_positions[validation_end:],
    }
    split_by_position = {
        position: split_name
        for split_name, positions in positions_by_split.items()
        for position in positions
    }

    split_assignments = embedding_metadata.copy()
    split_assignments["position_1based"] = variant_positions["position_1based"]
    split_assignments["split"] = split_assignments["position_1based"].map(
        split_by_position
    )

    if set(split_assignments["split"]) != {"train", "validation", "test"}:
        raise ValueError("A partition is empty or a variant has no assignment")
    splits_per_position = split_assignments.groupby("position_1based")["split"].nunique()
    if not splits_per_position.eq(1).all():
        raise ValueError("A residue position appears in multiple partitions")

    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    split_assignments.to_csv(SPLIT_CSV, index=False, mode="x")

    print(f"Split seed: {SPLIT_SEED}")
    for split_name, positions in positions_by_split.items():
        variant_count = split_assignments["split"].eq(split_name).sum()
        print(f"{split_name}: {variant_count:,} variants, {len(positions)} positions")
    print(f"Split validation: PASS — {len(split_assignments):,} variants, no position overlap")
    print(f"Saved: {SPLIT_CSV}")


if __name__ == "__main__":
    main()
