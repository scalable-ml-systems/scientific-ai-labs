from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from scientific_ml.mavedb import get_reference_protein
from scientific_ml.variants import validate_protein_variant

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_SCORE_CSV = PROJECT_DIR / "data/raw/cyp2c19_scores.csv"
PROCESSED_DIR = PROJECT_DIR / "data/processed"


def sha256_file(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def main() -> None:
    if not RAW_SCORE_CSV.is_file():
        raise FileNotFoundError(f"Missing score file: {RAW_SCORE_CSV}")

    variant_scores_df = pd.read_csv(
        RAW_SCORE_CSV, dtype={"accession": "string", "hgvs_pro": "string"}
    )
    required_columns = {"accession", "hgvs_pro", "score"}
    missing_columns = required_columns - set(variant_scores_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if len(variant_scores_df) != 8149:
        raise ValueError(f"Expected 8149 source records; found {len(variant_scores_df)}")

    reference_protein_sequence = get_reference_protein()
    scores_numeric = pd.to_numeric(variant_scores_df["score"], errors="coerce")
    duplicate_accessions = (
        variant_scores_df["accession"].notna()
        & variant_scores_df["accession"].duplicated(keep=False)
    )
    duplicate_annotations = (
        variant_scores_df["hgvs_pro"].notna()
        & variant_scores_df["hgvs_pro"].duplicated(keep=False)
    )

    validation_results = []
    for source_row_index, record in variant_scores_df.iterrows():
        accession = record["accession"]
        measured_score = scores_numeric.iloc[source_row_index]

        if pd.isna(accession) or not str(accession).strip():
            result = {"validation_status": "invalid", "validation_reason": "missing_accession"}
        elif duplicate_accessions.iloc[source_row_index]:
            result = {"validation_status": "invalid", "validation_reason": "duplicate_accession"}
        elif pd.isna(measured_score) or not math.isfinite(float(measured_score)):
            result = {"validation_status": "invalid", "validation_reason": "missing_or_nonfinite_score"}
        else:
            result = validate_protein_variant(
                record["hgvs_pro"], reference_protein_sequence
            )

        validation_results.append({
            "source_row_index": source_row_index,
            **result,
            "duplicate_protein_annotation": bool(duplicate_annotations.iloc[source_row_index]),
        })

    validated_df = pd.concat(
        [variant_scores_df.reset_index(drop=True), pd.DataFrame(validation_results)],
        axis=1,
    )
    validated_df["position_1based"] = validated_df["position_1based"].astype("Int64")

    accepted_variants_df = validated_df.loc[
        validated_df["validation_status"] == "accepted"
    ].copy()
    review_variants_df = validated_df.loc[
        validated_df["validation_status"] != "accepted"
    ].copy()

    assert len(accepted_variants_df) + len(review_variants_df) == len(variant_scores_df)
    assert accepted_variants_df["accession"].is_unique
    assert accepted_variants_df["variant_protein_sequence"].str.len().eq(
        len(reference_protein_sequence)
    ).all()

    reason_counts = Counter(review_variants_df["validation_reason"])
    report = {
        "dataset_accession": "urn:mavedb:00001199-a-1",
        "raw_csv_sha256": sha256_file(RAW_SCORE_CSV),
        "reference_protein_sha256": hashlib.sha256(
            reference_protein_sequence.encode("ascii")
        ).hexdigest(),
        "reference_protein_length": len(reference_protein_sequence),
        "total_records": len(variant_scores_df),
        "accepted": len(accepted_variants_df),
        "excluded": int((validated_df["validation_status"] == "excluded").sum()),
        "invalid": int((validated_df["validation_status"] == "invalid").sum()),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows_with_duplicate_protein_annotations": int(duplicate_annotations.sum()),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    accepted_variants_df.to_csv(PROCESSED_DIR / "cyp2c19_validated_missense.csv", index=False)
    review_variants_df.to_csv(PROCESSED_DIR / "cyp2c19_variants_for_review.csv", index=False)
    (PROCESSED_DIR / "cyp2c19_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print(f"\nOutputs: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

