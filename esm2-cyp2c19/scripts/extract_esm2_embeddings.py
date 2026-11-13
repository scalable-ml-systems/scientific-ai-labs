import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer

from scientific_ml.cyp2c19_dataset import (
    CYP2C19MissenseDataset,
    collate_cyp2c19_variants,
)

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MODEL_REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
BATCH_SIZE = 8

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATED_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cyp2c19_validated_missense.csv"
)


def select_execution_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available")

    return torch.device(requested_device)


def create_variant_data_loader(
    variant_dataset,
    esm2_tokenizer,
    batch_size,
):
    return DataLoader(
        dataset=variant_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=partial(
            collate_cyp2c19_variants,
            tokenizer=esm2_tokenizer,
        ),
    )


def extract_esm2_embeddings(
    variant_data_loader,
    esm2_model,
    execution_device,
):
    processed_variant_ids = []
    abundance_score_batches = []
    embedding_batches = []
    final_batch_size = 0

    total_batches = len(variant_data_loader)
    expected_embedding_size = esm2_model.config.hidden_size

    for batch_number, variant_batch in enumerate(
        variant_data_loader, start=1
    ):
        # Move the tokenized inputs to the model's device.
        input_ids = variant_batch["input_ids"].to(execution_device)

        attention_mask = variant_batch["attention_mask"].to(
            execution_device
        )

        special_tokens_mask = variant_batch["special_tokens_mask"].to(
            execution_device
        )

        # Run ESM-2 without recording gradients.
        with torch.inference_mode():
            esm2_outputs = esm2_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            residue_hidden_states = esm2_outputs.last_hidden_state

            # Select actual amino-acid tokens.
            valid_residue_mask = (
                attention_mask.bool()
                & ~special_tokens_mask.bool()
            )

            residue_counts = valid_residue_mask.sum(dim=1)

            if (residue_counts == 0).any().item():
                raise RuntimeError(
                    f"Batch {batch_number} contains an empty sequence"
                )

            expected_residue_counts = variant_batch[
                "protein_sequence_lengths"
            ].to(execution_device)

            if not torch.equal(
                residue_counts, expected_residue_counts
            ):
                raise RuntimeError(
                    f"Batch {batch_number}: residue count mismatch"
                )

            # Average only actual amino-acid representations.
            pooling_mask = valid_residue_mask.unsqueeze(-1).to(
                residue_hidden_states.dtype
            )

            variant_embeddings = (
                (residue_hidden_states * pooling_mask).sum(dim=1)
                / residue_counts.unsqueeze(-1)
            )

        current_batch_size = len(variant_batch["variant_ids"])

        if variant_embeddings.shape != (
            current_batch_size,
            expected_embedding_size,
        ):
            raise RuntimeError(
                f"Batch {batch_number}: unexpected embedding shape"
            )

        if not torch.isfinite(variant_embeddings).all().item():
            raise RuntimeError(
                f"Batch {batch_number}: non-finite embeddings"
            )

        # Keep finished results on CPU.
        # clone() outside inference_mode creates an ordinary tensor.
        embedding_batches.append(
            variant_embeddings.detach().cpu().clone()
        )

        abundance_score_batches.append(
            variant_batch["experimental_abundance_scores"].clone()
        )

        processed_variant_ids.extend(variant_batch["variant_ids"])
        final_batch_size = current_batch_size

        if (
            batch_number == 1
            or batch_number % 100 == 0
            or batch_number == total_batches
        ):
            print(
                f"Batch {batch_number}/{total_batches} | "
                f"Variants processed: {len(processed_variant_ids)}"
            )

    complete_variant_embeddings = torch.cat(
        embedding_batches, dim=0
    )

    complete_abundance_scores = torch.cat(
        abundance_score_batches, dim=0
    )

    return (
        processed_variant_ids,
        complete_abundance_scores,
        complete_variant_embeddings,
        final_batch_size,
    )

def sha256_file(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def save_embedding_artifacts(
    output_dir: Path,
    variant_ids: list[str],
    abundance_scores: torch.Tensor,
    variant_embeddings: torch.Tensor,
    input_csv_sha256: str,
    execution_device: torch.device,
    batch_size: int,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Publish the directory only after all three files are written.
    with TemporaryDirectory(dir=output_dir.parent) as temporary_dir:
        staging_dir = Path(temporary_dir) / "artifacts"
        staging_dir.mkdir()

        embeddings_path = staging_dir / "cyp2c19_esm2_embeddings.npy"
        metadata_path = staging_dir / "cyp2c19_embedding_metadata.csv"
        manifest_path = staging_dir / "cyp2c19_embedding_manifest.json"

        np.save(embeddings_path, variant_embeddings.numpy(), allow_pickle=False)

        with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
            writer = csv.writer(metadata_file)
            writer.writerow(["embedding_row", "variant_id", "experimental_abundance_score"])
            for embedding_row, (variant_id, abundance_score) in enumerate(
                zip(variant_ids, abundance_scores.tolist(), strict=True)
            ):
                writer.writerow([embedding_row, variant_id, abundance_score])

        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_accession": "urn:mavedb:00001199-a-1",
            "validated_csv_sha256": input_csv_sha256,
            "model_id": MODEL_NAME,
            "model_and_tokenizer_revision": MODEL_REVISION,
            "pooling": "final_hidden_state_mean_over_residue_tokens",
            "embedding_shape": list(variant_embeddings.shape),
            "embedding_dtype": str(variant_embeddings.numpy().dtype),
            "batch_size": batch_size,
            "device": str(execution_device),
            "gpu_name": (
                torch.cuda.get_device_name(execution_device)
                if execution_device.type == "cuda"
                else None
            ),
            "pytorch_version": str(torch.__version__),
            "pytorch_cuda_build": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "artifact_sha256": {
                embeddings_path.name: sha256_file(embeddings_path),
                metadata_path.name: sha256_file(metadata_path),
            },
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit-variants", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--check-batch-consistency",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Save validated embeddings and metadata to a new directory",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    
    if args.output_dir is not None and args.output_dir.exists():
        parser.error(f"--output-dir already exists: {args.output_dir}")

    input_csv_sha256 = (
        sha256_file(VALIDATED_CSV_PATH)
        if args.output_dir is not None
        else None
    )

    # Load the full validated dataset.
    variant_dataset = CYP2C19MissenseDataset(VALIDATED_CSV_PATH)

    if args.limit_variants is not None:
        if not 1 <= args.limit_variants <= len(variant_dataset):
            parser.error("--limit-variants is outside dataset bounds")

        active_dataset = Subset(
            variant_dataset,
            range(args.limit_variants),
        )
        expected_variant_ids = variant_dataset.variant_ids[
            :args.limit_variants
        ]
    else:
        active_dataset = variant_dataset
        expected_variant_ids = variant_dataset.variant_ids

    execution_device = select_execution_device(args.device)

    print("Validated variants:", len(variant_dataset))
    print("Selected for inference:", len(active_dataset))
    print("Execution device:", execution_device)

    # Load pretrained ESM-2.
    esm2_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    esm2_model = AutoModel.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        add_pooling_layer=False,
    ).to(execution_device)

    esm2_model.eval()

    # Build the DataLoader.
    variant_data_loader = create_variant_data_loader(
        active_dataset,
        esm2_tokenizer,
        args.batch_size,
    )

    print("DataLoader batches:", len(variant_data_loader))

    # Execute all selected batches.
    (
        processed_variant_ids,
        experimental_abundance_scores,
        complete_variant_embeddings,
        final_batch_size,
    ) = extract_esm2_embeddings(
        variant_data_loader,
        esm2_model,
        execution_device,
    )

    # Verify completeness and ordering.
    if processed_variant_ids != expected_variant_ids:
        raise RuntimeError("Variant coverage or ordering mismatch")

    if len(set(processed_variant_ids)) != len(processed_variant_ids):
        raise RuntimeError("Duplicate variant identifiers found")

    expected_shape = (
        len(active_dataset),
        esm2_model.config.hidden_size,
    )

    if complete_variant_embeddings.shape != expected_shape:
        raise RuntimeError("Incorrect final embedding matrix shape")

    if experimental_abundance_scores.shape != (
        len(active_dataset),
    ):
        raise RuntimeError("Experimental score alignment failure")

    if not torch.isfinite(complete_variant_embeddings).all().item():
        raise RuntimeError("Non-finite values in final embeddings")

    print("\n--- WP4 Validation ---")
    print("Processed variants:", len(processed_variant_ids))
    print("Final batch size:", final_batch_size)
    print("Embedding matrix:", tuple(complete_variant_embeddings.shape))
    print("All finite: True")
    print("Variant coverage and alignment: PASS")

    print(
        "First variant:",
        processed_variant_ids[0],
        "score:",
        round(experimental_abundance_scores[0].item(), 4),
    )
    print(
        "Last variant:",
        processed_variant_ids[-1],
        "score:",
        round(experimental_abundance_scores[-1].item(), 4),
    )

    # Optional consistency check using the first eight variants.
    if args.check_batch_consistency:
        check_dataset = Subset(
            variant_dataset,
            range(8),
        )

        check_results = []

        for check_batch_size in (1, 8):
            check_loader = create_variant_data_loader(
                check_dataset,
                esm2_tokenizer,
                check_batch_size,
            )

            check_ids, _, check_embeddings, _ = extract_esm2_embeddings(
                check_loader,
                esm2_model,
                execution_device,
            )

            if check_ids != variant_dataset.variant_ids[:8]:
                raise RuntimeError(
                    "Consistency check: variant ordering mismatch"
                )

            check_results.append(check_embeddings)

        max_difference = (
            check_results[0] - check_results[1]
        ).abs().max().item()

        print("\nMaximum batch-size difference:", max_difference)

        torch.testing.assert_close(
            check_results[0],
            check_results[1],
            atol=1e-4,
            rtol=1e-4,
        )

        print("Batch-size consistency: PASS")
    
    if args.output_dir is not None:
        save_embedding_artifacts(
            output_dir=args.output_dir,
            variant_ids=processed_variant_ids,
            abundance_scores=experimental_abundance_scores,
            variant_embeddings=complete_variant_embeddings,
            input_csv_sha256=input_csv_sha256,
            execution_device=execution_device,
            batch_size=args.batch_size,
        )
        print("Saved embedding artifacts:", args.output_dir)

if __name__ == "__main__":
    main()
