# Scientific ML Lifecycle Engineering — ESM-2/CYP2C19

Build a reproducible scientific ML workflow that starts with an existing pretrained protein model and experimental data, then progresses through model adaptation, evaluation, benchmarking, governance and deployment.

**Scientific question:**

Can representations from ESM-2 predict experimentally measured CYP2C19 protein variant abundance, and how reliable are those predictions on held-out variants?

**Engineering question:**

Can we build a reliable, reproducible workflow for preparing data, running a pretrained model, training and evaluating a predictor, versioning artifacts, and deploying approved model versions?

Scientific workload	ESM-2 + CYP2C19 VAMP-seq

Current milestone	Release 1 — Data Engineering & Model Execution

## What we have demonstrated

From the code and execution output:

- The experimental dataset loaded successfully: 8,149 rows.
- Ten supported variants were selected and applied to the 490-residue reference, with checks on mutation position and the original residue.
- ESM-2 generated token-level representations of shape `(10, 492, 320)`.
- Masked mean pooling produced one 320-dimensional embedding per variant: `(10, 320)`.
- The residue-count assertion passed, and all final embedding values were finite.
- Each embedding remained aligned with its variant identifier and experimental score in the current run.

What we have not yet demonstrated
- Full dataset validation: Only 10 variants were processed. We haven't established how many of the 8,149 records are valid, unsupported or require special handling.
- Reusable data pipeline: We haven't implemented a PyTorch Dataset, DataLoader or custom collate function.
- Reproducibility: The execution succeeded, but we haven't verified repeatability across runs, environments or batch sizes.
- Scientific predictive value: ESM-2 generated embeddings, but we haven't established whether they contain enough information to predict experimental CYP2C19 abundance.
- Model training: No prediction head, loss function, optimizer or training loop has been implemented.
- Artifact persistence: Embeddings were generated and inspected, but not yet saved as reusable, versioned artifacts.

### References 

- Scientific sources: Boyle et al. (2024), Rubin et al. (2025), Lin et al. (2023), UniProt and HGVS.
- Implementation references: MaveDB API, Hugging Face ESM and PyTorch documentation.