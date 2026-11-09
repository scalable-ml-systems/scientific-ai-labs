# Implementation Plan — Release 1

**Project:** ESM-2 / CYP2C19 Scientific ML Lifecycle Engineering  
**Release:** R1 — Data Engineering and Model Execution  
**Status:** Planned  
**Scope:** Single dataset, single pretrained model, single-machine execution

## 1. Definition of Success

At the end of Release 1, we must be able to execute one command that:

1. Loads a versioned CYP2C19 experimental dataset.
2. Validates variants against the reference protein sequence.
3. Constructs mutant protein sequences.
4. Loads samples through a custom PyTorch Dataset and DataLoader.
5. Tokenizes and batches those sequences.
6. Executes pretrained ESM-2 without updating its weights.
7. Produces one fixed-length embedding per accepted variant.
8. Saves the embeddings with variant identifiers and a reproducibility manifest.

**Final output:** A verified dataset-to-embedding pipeline, not a trained abundance predictor.

## 2. Fixed Technical Decisions

| Component | Decision |
|---|---|
| Dataset | Existing CYP2C19 MaveDB score set |
| Reference | CYP2C19, UniProt P33261, 490 amino acids |
| Model | `facebook/esm2_t6_8M_UR50D` |
| Model interface | Hugging Face `AutoTokenizer` and `AutoModel` |
| Data handling | Pandas, PyTorch `Dataset` and `DataLoader` |
| Batching | Custom CPU-side collate function |
| Initial representation | Mean-pooled final-layer residue embeddings |
| Inference mode | `model.eval()` and `torch.inference_mode()` |
| Execution | CPU smoke test, then a single GPU |
| Output | Embedding array, variant metadata, run manifest |
| Tests | pytest |
| Training | Excluded from R1 |

We will preserve the original experimental score, replicate scores and uncertainty columns for Release 2, but they will not influence embedding extraction.

## 3. Implementation Work Packages

### WP1 — First Working Vertical Slice

**Goal:** Prove that real scientific data can pass through ESM-2 and produce an embedding.

Start with 10 valid records from the existing CYP2C19 dataset.

Implementation:

- Reuse the existing MaveDB loading code.
- Read each variant's mutation notation.
- Construct the mutated amino-acid sequence.
- Load the pretrained ESM-2 model and tokenizer.
- Execute inference on the sequences.
- Extract fixed-size sequence representations.
- Print variant identifier, score, sequence length and embedding shape.

Expected result:

- 10 successfully processed variants.
- One 320-dimensional vector per variant.
- No training or gradient updates.

**Exit gate:** We have a real data-to-model-to-output path.

Do this before building additional abstractions.

### WP2 — Make the Scientific Data Reliable

**Goal:** Convert the exploratory dataset loading code into a validated input pipeline.

Reuse the current work that has already established:

- 8,149 variant records.
- A 490-amino-acid CYP2C19 reference sequence.
- Available score and replicate columns.
- Matching DNA translation and reference protein sequence.

Implement:

1. Save and identify a fixed source-data snapshot.
2. Preserve the original row identifiers and variant notation.
3. Parse supported single-amino-acid substitutions.
4. Convert the biological position from 1-based notation to Python's 0-based index.
5. Verify the expected wild-type residue.
6. Substitute the mutant residue.
7. Check resulting sequence length and valid amino-acid characters.
8. Detect missing scores, duplicate records, conflicting records and unsupported variant types.
9. Write a data-validation summary.

For example:

`p.Ala103Cys` means validate alanine at position 103, then replace it with cysteine.

Invalid or unsupported variants must be explicitly reported rather than silently discarded.

**Output contract per accepted record:**

| Field | Purpose |
|---|---|
| `variant_id` | Stable record identifier |
| `variant` | Original protein mutation notation |
| `position` | Mutated residue position |
| `sequence` | Full mutant protein sequence |
| `score` | Experimental abundance measurement |
| `sd`, `se` | Experimental uncertainty if available |
| `source_row` | Link to the original dataset record |

**Exit gate:** Every input record is accounted for as accepted, rejected or requiring investigation. The accepted set contains valid sequences with traceable experimental records.

### WP3 — Build the PyTorch Data Pipeline

**Goal:** Replace direct loops over a DataFrame with the standard data abstractions used by ML workloads.

Implement a custom `CYP2C19Dataset`.

Responsibilities:

- Store or reference validated records.
- Implement `__len__`.
- Implement `__getitem__`.
- Return the sequence, variant identifier, position and associated experimental metadata.

Implement a custom collate function.

Responsibilities:

- Collect multiple samples into a batch.
- Tokenize sequences using the ESM-2 tokenizer.
- Apply appropriate padding.
- Construct `input_ids`, `attention_mask` and special-token masks.
- Preserve the mapping between model inputs and variant identifiers.

Use `DataLoader` to control:

- Batch size.
- Number of workers.
- Input ordering.
- Memory pinning on supported GPU setups.

**Initial configuration:**

- Batch size: 8.
- Workers: 0.
- Shuffle: disabled.
- Drop last batch: disabled.
- Precision: float32.

These are baseline settings, not optimized values.

**Exit gate:** All accepted variants pass through the DataLoader exactly once. Changing batch size does not change sample membership or variant-to-output alignment.

### WP4 — Implement the ESM-2 Execution Layer

**Goal:** Execute the pretrained model correctly and extract reproducible representations.

Implement:

1. Load the exact model and tokenizer revisions.
2. Select the execution device.
3. Move model parameters to the selected device.
4. Enter evaluation/inference mode.
5. Move batch tensors to the same device.
6. Run the model forward pass.
7. Inspect `last_hidden_state`.
8. Exclude padding and special tokens.
9. Mean-pool valid amino-acid token representations.
10. Return one vector per variant.

Expected tensor behavior:

| Stage | Expected shape |
|---|---|
| Tokenized input | `[batch, sequence_length]` |
| Final hidden states | `[batch, sequence_length, 320]` |
| Pooled embeddings | `[batch, 320]` |

The exact tokenized sequence length includes model-specific special tokens and possibly padding.

Tests must verify:

- Correct device placement.
- Correct output dimensions.
- No gradients are stored during inference.
- No NaN or infinite embedding values.
- Identical inputs produce consistent outputs within a numerical tolerance.
- Single-sample and batched inference agree within the chosen tolerance.

**Exit gate:** ESM-2 processes real variants in batches, preserving correct input/output alignment.

### WP5 — Cache Embeddings and Record Provenance

**Goal:** Avoid repeating expensive model execution whenever we begin training or evaluating experiments in later releases.

Save:

- `embeddings.npz` — numeric embeddings.
- `variants.csv` — ordered variant identifiers and metadata.
- `manifest.json` — execution provenance.
- `validation_report.json` — accepted and rejected records.

The manifest must capture:

- MaveDB accession and source-file checksum.
- Reference sequence checksum.
- Model identifier and exact revision.
- Tokenizer revision.
- Embedding extraction and pooling strategy.
- PyTorch and Transformers versions.
- Hardware/device information.
- Precision and batch size.
- Git commit.
- Execution timestamp.
- Record counts.
- Output checksums.

Write outputs safely so interrupted runs do not leave a partial file appearing complete.

**Exit gate:** Saved embeddings can be reloaded and correctly matched to their original variants. A second execution using identical inputs produces results within documented tolerances.

## 4. Repository Organization

Continue using the existing `scientific-ai-labs` repository. Do not create a monorepo or introduce an independent nested project.

Proposed organization:

```text
scientific-ai-labs/
├── README.md
├── pyproject.toml
├── configs/
│   └── release1.yaml
├── src/
│   └── scientific_ml/
│       ├── mavedb.py
│       ├── variants.py
│       ├── dataset.py
│       ├── esm2.py
│       ├── extract.py
│       └── artifacts.py
├── scripts/
│   └── inspect_dataset.py
├── tests/
│   ├── test_variants.py
│   ├── test_dataset.py
│   ├── test_collate.py
│   └── test_embeddings.py
├── data/
│   └── raw/
└── outputs/
    └── release1/
```

Existing working scripts should be reused or refactored carefully; we will not rewrite functioning code simply to match the folder diagram.

Generated artifacts and large data files stay out of Git. Configuration, tests, provenance schemas and documentation are version-controlled.

## 5. Proposed Execution Interface

Implement one command-line entry point with configurable execution.

**First validation:**

`python -m scientific_ml.extract --config configs/release1.yaml --limit 10`

**Full execution:**

`python -m scientific_ml.extract --config configs/release1.yaml --all`

These are target commands to implement, not commands expected to work before the code exists.

The configuration controls model revision, data source, device, batch size, workers, precision and output directory.

No web service, dashboard or orchestration framework is required.

## 6. Verification and Acceptance Tests

| Area | Required verification |
|---|---|
| Input integrity | Source and reference checksums recorded |
| Variant correctness | Mutation matches expected reference residue |
| Dataset | All accepted records returned exactly once |
| Collation | Correct token masks, metadata alignment and batch shapes |
| Model | Expected output dimensions, finite values |
| Execution | Evaluation mode, no gradient tracking |
| Batching | Batch-size-independent outputs within tolerance |
| Data integrity | Output rows map correctly to original variants |
| Artifact loading | Saved embeddings can be reloaded |
| Repeatability | Repeat execution meets numerical tolerance |
| Auditability | Rejected records and processing counts are documented |

**Release 1 passes only when the complete accepted dataset can be processed—not just the initial 10-record smoke test.**

The validated number may differ from the original 8,149 records if unsupported or invalid cases are discovered. All exclusions must be explained.

## 7. Explicitly Out of Scope

Release 1 will not implement:

- Training loops or optimizers.
- Regression heads.
- Fine-tuning.
- Scientific accuracy evaluation.
- Train/validation/test splits.
- MLflow experiment tracking.
- Model registries or promotion gates.
- ONNX or quantization.
- Deployment APIs.
- Prometheus or service monitoring.
- Distributed training.
- Kubernetes.
- GPU kernel profiling or extensive performance optimization.

Those are part of subsequent approved releases.

## 8. Completion Deliverables

Release 1 is complete with five concrete artifacts:

1. **Working pipeline** — Real variants processed by a PyTorch Dataset/DataLoader and ESM-2.
2. **Validated dataset** — Input integrity report with accepted and rejected records.
3. **Embeddings** — Fixed-size vectors matched to variant identifiers.
4. **Reproducibility manifest** — Sufficient configuration and provenance to repeat execution.
5. **Engineering runbook** — Setup, execution, validation and known limitations.

No separate tutorial series or additional demonstration project.

## 9. Handoff to Release 2

Release 2 consumes the validated records and cached embeddings.

It will introduce scientific baselines, training/validation/test splits, a PyTorch prediction head, optimizer loops, fine-tuning, checkpoints and experiment tracking.

The cached representations allow us to train a lightweight prediction head without executing the ESM-2 encoder on every training batch.

If the encoder itself is fine-tuned, embeddings must be regenerated from the updated model rather than reusing the frozen-encoder cache.

**Release 1 is complete when the data-to-embedding pipeline is trustworthy. Release 2 begins when we ask whether those embeddings can predict experimental abundance.**
