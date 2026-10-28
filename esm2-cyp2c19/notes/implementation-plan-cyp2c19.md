# ESM-2 + CYP2C19 — Implementation plan

**Repository:** `scientific-ai-labs/esm2-cyp2c19`
**Scientific dataset:** Boyle et al. (2024), CYP2C19 VAMP-seq
**Experimental accession:** `urn:mavedb:00001199-a-1`
**Model:** `facebook/esm2_t12_35M_UR50D`
**Project status:** Not implemented or experimentally validated

## Project objective

Evaluate whether a pretrained ESM-2 protein language model can rank experimentally measured changes in CYP2C19 protein abundance following single-amino-acid substitutions.

Develop the implementation incrementally while learning scientific data analysis, protein language modeling, and GPU inference.

We will not train a new model. We will not claim to predict drug response or clinical outcomes.

## Implementation stages

### Phase A — Understand the biological data

**Stage 0 — Establish the project**

- Create a single, clean repository.
- Set up a Python virtual environment.
- Read the paper's abstract, experimental methodology, and data availability section.
- Document the research question in `README.md`.

**Python concepts:** Virtual environments, imports, packages, paths and basic project organization.

**Checkpoint:** Explain what CYP2C19 is, what VAMP-seq measures, and what scientific question we're investigating.

**Stage 1 — Retrieve the real experimental metadata**

- Query the official MaveDB API using the selected accession.
- Inspect the returned metadata without making assumptions about its structure.
- Identify the experimental target, assay, score set, and available data files.
- Record the source accession and retrieval information.

**Python concepts:** HTTP requests, dictionaries, JSON, strings, error handling.

**Checkpoint:** Inspect the actual metadata object and explain where the experimental measurements come from.

**Stage 2 — Inspect the real measurements**

- Download the experimental score CSV.
- Read it with pandas.
- Identify the actual variant identifiers, measurement columns and replicate-related fields.
- Count rows, missing values and different variant types.
- Visualize the distribution of experimental scores.
- Retrieve the reference CYP2C19 protein sequence.

**Python concepts:** CSV files, pandas DataFrames, columns, filtering, data types and descriptive statistics.

**Checkpoint:** Explain what one experimental row means and distinguish protein-abundance measurements from enzyme activity.

**Stage 3 — Implement mutation validation**

- Identify single-amino-acid substitutions.
- Interpret the dataset's actual variant notation.
- Convert biological residue positions to Python indices.
- Verify every reference amino acid against the protein sequence.
- Exclude unsupported variant types with explicit reasons.
- Detect duplicates and resolve repeated measurements using a documented policy.

**Python concepts:** Functions, indexing, regular expressions, tuples, dictionaries, loops, sets and exceptions.

**Checkpoint:** Produce a validated dataset with an auditable record of included and excluded mutations.

### Phase B — Understand ESM-2

**Stage 4 — Load and inspect the pretrained model**

- Load the official ESM-2 35M checkpoint.
- Tokenize the real CYP2C19 sequence.
- Examine token IDs, special tokens and sequence length.
- Inspect input/output tensor shapes.
- Confirm correct CPU/GPU execution and numerical outputs.

**Python concepts:** Objects, PyTorch tensors, shapes, indexing, devices and inference mode.

**Checkpoint:** Explain what the model accepts and what its output logits represent.

**Stage 5 — Score one actual mutation**

- Select one valid measured missense variant.
- Mask its wild-type amino acid in the reference sequence.
- Execute the pretrained model.
- Extract scores for the wild-type and alternate amino acids.
- Calculate the masked-marginal log-likelihood ratio.
- Compare the predicted score with the experimental abundance measurement.

**Python concepts:** Tensor indexing, scalar values, functions, mathematical operations and debugging.

**Checkpoint:** Explain the scoring equation, the direction of the model score, and why sequence plausibility is not the same as protein abundance.

**Stage 6 — Score the complete valid dataset**

- Group variants by mutated residue position.
- Avoid redundant model executions for substitutions sharing a masked context.
- Batch masked protein sequences when justified.
- Save predictions alongside validated variant identifiers.
- Check score completeness and finite numerical values.
- Verify representative scores using an independent implementation.

**Python concepts:** Grouping, iterators, batching, DataFrame joins, tensor indexing and file output.

**Checkpoint:** Produce real ESM-2 predictions for every eligible experimental single-amino-acid substitution.

### Phase C — Evaluate scientific predictions

**Stage 7 — Compare model predictions with experiments**

- Join predictions and measured abundance scores.
- Calculate Spearman rank correlation.
- Plot predicted scores against experimental measurements.
- Investigate mutations with large prediction–experiment disagreement.
- Analyze regional patterns only when supported by actual data and published biological context.
- Compare against an appropriate baseline.

**Python concepts:** SciPy, matplotlib, statistical functions and data visualization.

**Checkpoint:** Explain where ESM-2 agrees or disagrees with the abundance assay and identify plausible limitations.

### Phase D — Apply infrastructure engineering

**Stage 8 — Measure and improve GPU inference**

- Record model load time, warm inference time and full-dataset runtime.
- Measure peak GPU memory and throughput.
- Compare simple execution with position-based batching.
- Validate that optimization does not change mutation scores beyond numerical tolerance.
- Save hardware, software and model-version information.

**Python concepts:** Performance measurement, CUDA synchronization, memory profiling, configuration and JSON artifacts.

**Checkpoint:** Produce actual GPU measurements and explain the performance tradeoffs.

**Stage 9 — Publish the scientific engineering case study**

- Finalize reproducible execution instructions.
- Publish real evaluation plots and concise results.
- Document assumptions, exclusion rules and model limitations.
- Summarize scientific findings and engineering decisions.
- Explain what you would improve or investigate next.

**Checkpoint:** A reviewer can understand the experiment, inspect the evidence, and reproduce the workflow without relying on undocumented manual steps.

## Data structures in Python

| Structure             | Used for                                                          |
| --------------------- | ----------------------------------------------------------------- |
| `str`                 | Protein sequences, mutation identifiers, accessions               |
| `dict`                | MaveDB metadata, configuration and run manifests                  |
| `list`                | Ordered records, positions and predictions                        |
| `set`                 | Unique positions and duplicate detection                          |
| `tuple` / `dataclass` | Validated mutation records and experiment objects                 |
| `pandas.DataFrame`    | Experimental measurements, validated variants and predictions     |
| `torch.Tensor`        | GPU inputs, masked sequences, logits and numerical scoring arrays |
| JSON files            | Reproducibility metadata and evaluation summaries                 |

## Repository layout

Create files when we reach the stage that needs them.

```text
esm2-cyp2c19/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── experiment.yaml
├── scripts/
│   ├── inspect_dataset.py
│   ├── run_inference.py
│   └── evaluate.py
├── src/
│   └── esm2_cyp2c19/
│       ├── data.py
│       ├── variants.py
│       └── scoring.py
├── tests/
│   ├── test_variants.py
│   └── test_scoring.py
├── data/
│   ├── raw/
│   └── processed/
├── results/
└── docs/
    └── findings.md
```

Raw data and downloaded model weights remain outside version control. Reproducible configurations, source links, code, tests and selected genuine findings are retained.

## Scientific correctness requirements

- Never invent mutation data or biological measurements.
- Record exactly which score set, sequence and dataset version were used.
- Do not assume MaveDB score columnDeliberately es are standardized across studies.
- Distinguish missense substitutions, synonymous variants, nonsense variants and multi-substitution records.
- Validate sequence coordinates before scoring.
- Do not use experimental scores as input to the zero-shot model.
- Preserve the experimental score direction and documented normalization.
- Report dataset coverage and exclusion counts.
- Do not equate predicted sequence likelihood with protein abundance or drug-metabolizing activity.
- A poor correlation is an acceptable result if the evaluation is correct and reproducible.

## xcluded from Version 1

No Kubernetes, API server, distributed orchestration, training from scratch, fine-tuning, elaborate package framework, or unrelated scientific models.

**Definition of success:** Explain and reproduce every transformation from published experimental data through protein-sequence validation, ESM-2 inference, statistical comparison, and GPU profiling.

