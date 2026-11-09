from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

def collate_cyp2c19_variants(variant_samples, tokenizer):
        protein_sequences = [sample["protein_sequence"] for sample in variant_samples]

        tokenized_sequences = tokenizer(
            protein_sequences,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

        experimental_abundance_scores = torch.tensor(
            [sample["experimental_abundance_score"] for sample in variant_samples],
            dtype=torch.float32,
        )

        protein_sequence_lengths = torch.tensor(
            [len(sequence) for sequence in protein_sequences],
            dtype=torch.long,
        )

        residue_mask = (
            tokenized_sequences["attention_mask"].bool()
            & ~tokenized_sequences["special_tokens_mask"].bool()
        )

        if not torch.equal(
            residue_mask.sum(dim=1),
            protein_sequence_lengths,
        ):
            raise ValueError(
                "Tokenized residue counts do not match protein sequence lengths"
            )

        return {
            "variant_ids": [sample["variant_id"] for sample in variant_samples],
            "protein_mutations": [sample["protein_mutation"] for sample in variant_samples],
            "experimental_abundance_scores": (experimental_abundance_scores),
            "protein_sequence_lengths": protein_sequence_lengths,
            "input_ids": tokenized_sequences["input_ids"],
            "attention_mask": tokenized_sequences["attention_mask"],
            "special_tokens_mask": (tokenized_sequences["special_tokens_mask"]),
        }

class CYP2C19MissenseDataset(Dataset):
    REQUIRED_COLUMNS = {
        "accession",
        "hgvs_pro",
        "variant_protein_sequence",
        "score",
    }
   
    def __init__(self, validated_csv_path: str | Path):
        self.validated_csv_path = Path(validated_csv_path)

        self.variant_records_df = pd.read_csv(self.validated_csv_path)

        missing_columns = self.REQUIRED_COLUMNS - set(self.variant_records_df.columns)

        if missing_columns:
            raise ValueError(f"Missing dataset columns: {missing_columns}")

        if self.variant_records_df.empty:
            raise ValueError("Validated dataset is empty")

        required_data = self.variant_records_df[list(self.REQUIRED_COLUMNS)]

        if required_data.isna().any().any():
            raise ValueError("Missing required variant data")

        if self.variant_records_df["accession"].duplicated().any():
            raise ValueError("Duplicate variant accessions detected")

        sequence_lengths = self.variant_records_df["variant_protein_sequence"].str.len()

        if not sequence_lengths.eq(490).all():
            raise ValueError("Unexpected protein sequence length")

        self.variant_records = self.variant_records_df.to_dict(orient="records")

        self.variant_ids = self.variant_records_df["accession"].tolist()

    def __len__(self):
        return len(self.variant_records)

    def __getitem__(self, index):
        variant_record = self.variant_records[index]

        return {
            "variant_id": variant_record["accession"],
            "protein_mutation": variant_record["hgvs_pro"],
            "protein_sequence": (variant_record["variant_protein_sequence"]),
            "experimental_abundance_score": float(variant_record["score"]),
        }
