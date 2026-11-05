
from pathlib import Path
import re

import pandas as pd
import torch
from Bio.SeqUtils import seq1
from transformers import AutoTokenizer, AutoModel

from scientific_ml.mavedb import get_reference_protein

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cyp2c19_scores.csv"
)

def build_mutant_sequence(reference, mutation):
    pattern = r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})"
    match = re.fullmatch(pattern, mutation)

    if match is None:
        raise ValueError(f"Unsupported mutation: {mutation}")

    original, position, replacement = match.groups()
    position = int(position)

    original_aa = seq1(original)
    replacement_aa = seq1(replacement)

    if original_aa not in "ACDEFGHIKLMNPQRSTVWY":
        raise ValueError(f"Unsupported original residue: {original}")

    if replacement_aa not in "ACDEFGHIKLMNPQRSTVWY":
        raise ValueError(f"Unsupported replacement residue: {replacement}")

    index = position - 1

    if not 0 <= index < len(reference):
        raise ValueError(f"Invalid position: {position}")

    if reference[index] != original_aa:
        raise ValueError(f"Reference mismatch: {mutation}")

    return reference[:index] + replacement_aa + reference[index + 1:]


def main():
    # 1. Load existing experimental data and reference protein
    df = pd.read_csv(CSV_PATH)
    reference = get_reference_protein()

    print("Dataset rows:", len(df))
    print("Reference length:", len(reference))

    # 2. Select 10 supported real variants
    selected = []

    for row in df.itertuples(index=False):
        if pd.isna(row.hgvs_pro) or pd.isna(row.score):
            continue

        try:
            sequence = build_mutant_sequence(reference, row.hgvs_pro)
        except ValueError as error:
            print("Skipped:", row.hgvs_pro, error)
            continue

        selected.append({
            "id": row.accession,
            "variant": row.hgvs_pro,
            "score": row.score,
            "sequence": sequence,
        })

        if len(selected) == 10:
            break

    if len(selected) != 10:
        raise RuntimeError("Could not find 10 valid variants")

    # 3. Load pretrained ESM-2
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        add_pooling_layer=False,
    )
    model.eval()

    # 4. Convert protein sequences into model inputs
    sequences = [item["sequence"] for item in selected]

    batch = tokenizer(
        sequences,
        padding=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    # 5. Execute model forward pass
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        hidden_states = outputs.last_hidden_state

        # Exclude padding and ESM special tokens
        valid_mask = (
            batch["attention_mask"].bool()
            & ~batch["special_tokens_mask"].bool()
        )

        assert valid_mask.sum(dim=1).tolist() == [
            len(s) for s in sequences
        ]

        mask = valid_mask.unsqueeze(-1)

        embeddings = (hidden_states * mask).sum(dim=1)
        embeddings = embeddings / mask.sum(dim=1)

    # 6. Inspect results
    print("\nHidden states:", tuple(hidden_states.shape))
    print("Embeddings:", tuple(embeddings.shape))
    print("All finite:", torch.isfinite(embeddings).all().item())

    for i, item in enumerate(selected):
        print(
            item["id"],
            item["variant"],
            "score:", round(item["score"], 4),
            "sequence length:", len(item["sequence"]),
            "embedding shape:", tuple(embeddings[i].shape),
        )


if __name__ == "__main__":
    main()

