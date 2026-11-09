import re
from functools import partial
from pathlib import Path

import torch
from Bio.SeqUtils import seq1
from scientific_ml.cyp2c19_dataset import (
    CYP2C19MissenseDataset,
    collate_cyp2c19_variants,
)
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATED_CSV_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cyp2c19_validated_missense.csv"
)

BATCH_SIZE = 8


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

    return reference[:index] + replacement_aa + reference[index + 1 :]


def main():
    # 1. Load the validated scientific dataset
    variant_dataset = CYP2C19MissenseDataset(VALIDATED_CSV_PATH)

    print("Validated variants:", len(variant_dataset))

    # 2. Load the pretrained tokenizer
    esm2_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Construct the PyTorch DataLoader
    variant_data_loader = DataLoader(
        dataset=variant_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=partial(
            collate_cyp2c19_variants,
            tokenizer=esm2_tokenizer,
        ),
    )

    print("DataLoader batches:", len(variant_data_loader))

    # 4. Inspect the entire DataLoader
    processed_variant_ids = []
    first_variant_batch = None
    final_batch_size = 0

    for variant_batch in variant_data_loader:
        batch_variant_ids = variant_batch["variant_ids"]

        processed_variant_ids.extend(batch_variant_ids)
        final_batch_size = len(batch_variant_ids)

        if first_variant_batch is None:
            first_variant_batch = variant_batch

    if processed_variant_ids != variant_dataset.variant_ids:
        raise RuntimeError("DataLoader variant ordering or coverage mismatch")

    print("Processed variants:", len(processed_variant_ids))
    print("Final batch size:", final_batch_size)
    print("Dataset coverage: PASS")

    # 5. Inspect tensors from the first batch
    print(
        "Input IDs:",
        tuple(first_variant_batch["input_ids"].shape),
    )
    print(
        "Abundance scores:",
        tuple(first_variant_batch["experimental_abundance_scores"].shape),
    )

    # 6. Execute ESM-2 on the first batch
    esm2_model = AutoModel.from_pretrained(
        MODEL_NAME,
        add_pooling_layer=False,
    )
    esm2_model.eval()

    with torch.inference_mode():
        esm2_outputs = esm2_model(
            input_ids=first_variant_batch["input_ids"],
            attention_mask=first_variant_batch["attention_mask"],
        )

        #residue_hiddne_states contain vectors
        residue_hidden_states = esm2_outputs.last_hidden_state

        valid_residue_mask = (
            first_variant_batch["attention_mask"].bool()
            & ~first_variant_batch["special_tokens_mask"].bool()
        )

        pooling_mask = valid_residue_mask.unsqueeze(-1).to(residue_hidden_states.dtype)

        variant_embeddings = (residue_hidden_states * pooling_mask).sum(
            dim=1
        ) / pooling_mask.sum(dim=1)

    # 7. Verify output shapes and numerical validity
    if variant_embeddings.shape != (BATCH_SIZE, 320):
        raise RuntimeError("Unexpected embedding shape")

    if not torch.isfinite(variant_embeddings).all():
        raise RuntimeError("Non-finite embeddings detected")

    print(
        "Hidden states:",
        tuple(residue_hidden_states.shape),
    )
    print(
        "Embeddings:",
        tuple(variant_embeddings.shape),
    )
    print("Embedding validation: PASS")

    for variant_id, mutation, score in zip(
        first_variant_batch["variant_ids"],
        first_variant_batch["protein_mutations"],
        first_variant_batch["experimental_abundance_scores"],
    ):
        print(variant_id, mutation, "score:", round(score.item(), 4))


if __name__ == "__main__":
    main()
