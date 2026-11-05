# Project Scope v1.0 — Scientific ML Lifecycle Engineering

**Project:** ESM-2 / CYP2C19 Scientific ML Lifecycle  
**Repository:** `scientific-ai-labs` (existing repository, project at root)  
**Project type:** End-to-end ML engineering  
**Primary role alignment:** ML Infrastructure Engineer, AI Platform Engineer, Scientific AI Platform Engineer  
**Scope:** One dataset, one pretrained model family, one scientific prediction task, one evolving engineering system.

---

## 1. Project Objective

Build a reproducible scientific ML workflow that starts with an existing pretrained protein model and experimental data, then progresses through model adaptation, evaluation, benchmarking, governance and deployment.

**Scientific question:**

Can representations from ESM-2 predict experimentally measured CYP2C19 protein variant abundance, and how reliable are those predictions on held-out variants?

**Engineering question:**

Can we build a reliable, reproducible workflow for preparing data, running a pretrained model, training and evaluating a predictor, versioning artifacts, and deploying approved model versions?

The objective is **engineering competency demonstrated through a real scientific ML workload**, not developing a novel biological model.

## 2. Fixed Project Inputs

| Item | Decision |
|---|---|
| Scientific target | Human CYP2C19 |
| Prediction target | Experimental protein abundance (VAMP-seq score) |
| Dataset | CYP2C19 VAMP-seq, MaveDB `urn:mavedb:00001199-a-1` |
| Initial data | 8,149 recorded variant measurements |
| Pretrained model | Meta ESM-2, `esm2_t6_8M_UR50D` |
| ML framework | PyTorch |
| Model integration | Hugging Face Transformers |
| Compute | Single GPU; CPU for preprocessing |
| Language | Python |
| Experiment tracking | MLflow |
| Interface | Batch prediction CLI; one minimal inference API |
| Deployment | Single-host execution |

No additional datasets, scientific targets, model families or distributed infrastructure are required for v1.

## 3. Required Functional Scope

### Release 1 — Data Engineering and Model Execution

**Problem:** Convert experimental variant records into verified model inputs and execute the pretrained model correctly.

Build:

- Load and validate experimental variant records.
- Verify mutation positions and amino acid substitutions against the reference sequence.
- Construct variant protein sequences.
- Implement PyTorch `Dataset` and `DataLoader`.
- Implement a custom collate function for sequence tensors, masks, variant identifiers and labels.
- Support configurable batch sizes.
- Load ESM-2 through Hugging Face.
- Generate and cache sequence representations.
- Measure basic data loading and inference behavior.

**Deliverable:** A reproducible data-to-embedding pipeline.

**Acceptance:** All usable records are validated or explicitly rejected; embedding outputs have the expected shapes; batching produces consistent results; a separate rerun can reproduce the generated outputs within documented numerical tolerances.

### Release 2 — Training, Evaluation and Reproducibility

**Problem:** Determine whether ESM-2 representations can predict experimental abundance.

Build:

- Establish a simple statistical prediction baseline.
- Train a regression head on frozen ESM-2 representations.
- Implement a PyTorch training/validation loop.
- Perform one limited encoder fine-tuning experiment.
- Save and reload model and optimizer checkpoints.
- Resume an interrupted training run.
- Record model configuration, data splits, random seeds and environment.
- Track experiments in MLflow.
- Compare training runs using scientific evaluation metrics.

Evaluate:

- Spearman correlation.
- Mean absolute error (MAE).
- Performance on held-out variants.
- Position-held-out evaluation to assess generalization to unseen residue positions.
- Variability across repeated runs.
- Prediction failures and limitations.

**Deliverable:** A trained predictor, baseline comparison and reproducible evaluation report.

**Acceptance:** We can independently reproduce the evaluation procedure, reload a trained model, resume training, and explain whether ESM-2 adaptation improves prediction over the baseline.

A successful engineering result does not require ESM-2 to outperform the baseline. A well-supported negative result is valid.

### Release 3 — Performance Benchmarking

**Problem:** Characterize the computational cost of the workflow.

Build a repeatable benchmark runner that records:

- DataLoader throughput.
- Model cold-start time.
- Warm inference latency.
- Batch throughput (sequences/second).
- GPU peak memory.
- Model and preprocessing time separately.
- Hardware, software and model configuration.

Compare a small, fixed matrix of batch sizes.

**Deliverable:** A benchmark report identifying the dominant performance costs.

**Acceptance:** Measurements can be reproduced under documented conditions, and any optimization claim is supported by before/after data.

No broad GPU performance tuning investigation is required.

### Release 4 — Model Governance and Artifact Lifecycle

**Problem:** Ensure only validated model versions can be used for approved inference workflows.

Implement:

- Versioned training and evaluation artifacts.
- Dataset and checkpoint lineage.
- Model registration using MLflow.
- Configurable quality-gate rules.
- Candidate, staging, production and archived lifecycle states through metadata and deployment aliases.
- Promotion and rejection records.
- Model rollback.

Quality gates must cover:

1. Scientific evaluation against the agreed baseline.
2. Reproducibility and required metadata.
3. Model loading and inference compatibility.
4. Documented performance requirements.

Scientific metric thresholds will be selected after baseline evaluation, not invented beforehand.

**Deliverable:** A quality-gated, versioned model lifecycle.

**Acceptance:** A failing candidate is rejected, an approved candidate can be promoted, and a prior model version can be restored.

### Release 5 — Deployment and Observability

**Problem:** Allow another engineer to use an approved model without manually running research scripts.

Build:

- Batch prediction CLI.
- One minimal FastAPI prediction endpoint.
- Model version loading through the registry.
- Input validation and structured error reporting.
- Health and readiness checks.
- Structured logs.
- Prometheus-compatible metrics.

Track:

- Prediction count.
- Request latency.
- Error count and error categories.
- Active model version.
- GPU memory consumption where applicable.

**Deliverable:** An operable single-host inference service.

**Acceptance:** Another engineer can load the approved model, submit valid inputs, inspect the resulting predictions and diagnose common failures.

No Kubernetes deployment is required.

### Release 6 — Export and Runtime Comparison

**Problem:** Determine whether an alternative model representation or runtime improves inference efficiency.

Build:

- A PyTorch reference inference path.
- One ONNX export path, where supported.
- Numerical compatibility testing between PyTorch and the exported model.
- A benchmark comparison of the supported runtimes.
- One INT8 quantization experiment if operator support and hardware make it meaningful.

**Deliverable:** A runtime comparison with measured costs, benefits and compatibility limitations.

**Acceptance:** We can demonstrate equivalent behavior within defined tolerances, or document why a proposed export or optimization is unsuitable.

A faster result is not required. INT8 deployment is not a mandatory outcome.

---

## 4. Cross-Cutting Engineering Requirements

These apply throughout the project rather than becoming separate workstreams.

**Reproducibility**
- Pin dataset, model and dependency revisions.
- Preserve split manifests, configurations and random seeds.
- Distinguish numerical reproducibility from run-to-run training variability.

**Testing**
- Unit tests for data transformations.
- Integration tests for model execution and artifact loading.
- Regression tests for predictions and model compatibility.
- Failure tests for checkpoint interruption and invalid inputs.

**Artifact lineage**
- Connect each model version to its source data, code revision, configuration, training run and evaluation report.

**Documentation**
- Maintain one architecture diagram.
- Maintain one reproducible runbook.
- Record important engineering decisions and unexpected results.

No separate documentation project.

## 5. Explicitly Out of Scope

The following will not be added to v1:

- New neural-network architecture design or foundation-model pretraining.
- Multiple biological targets or datasets.
- Boltz-2, Evo 2, Cellpose or other scientific models.
- Distributed training, DDP/FSDP or multi-node GPU clusters.
- Kubernetes, Kueue, Ray, Kubeflow or custom job schedulers.
- Custom GPU kernels or low-level CUDA development.
- Large-scale hyperparameter optimization.
- Full-featured feature stores or data lakes.
- Building a custom model registry or experiment-tracking framework.
- Multi-user dashboards, authentication platforms or elaborate UIs.
- Production cloud infrastructure or high-availability deployment.
- Complex agent frameworks or LLM applications.
- Extensive comparisons across many model sizes or inference runtimes.
- Biological claims beyond what the experimental dataset and evaluation support.

These may be separate future investigations. They are not dependencies for completing this project.

## 6. Definition of Done

The project is complete when:

1. A real CYP2C19 dataset can be validated and processed using a PyTorch data pipeline.
2. ESM-2 can be executed and adapted to produce abundance predictions.
3. Training, evaluation, checkpoints and limited fine-tuning are implemented and understood.
4. Model results are compared against a baseline using reproducible scientific evaluation.
5. Runtime behavior is measured using a repeatable benchmark.
6. Model versions and their data/code lineage are traceable.
7. Quality gates govern model promotion and rejection.
8. Approved models can be loaded through a usable inference interface.
9. Inference behavior and failures can be observed.
10. An ONNX comparison and, where feasible, INT8 experiment document numerical and runtime tradeoffs.
11. The code, configurations, runbooks and results are sufficient for another engineer to reproduce the workflow.

**The project does not require a successful new scientific discovery or state-of-the-art prediction accuracy.**

## 7. Scope Control Rules

**Rule 1 — One scientific problem.** Every addition must contribute to understanding or operating the CYP2C19 abundance-prediction workflow.

**Rule 2 — Build before abstracting.** Implement a working vertical slice before introducing reusable infrastructure.

**Rule 3 — Adopt existing tools.** Use established PyTorch, Hugging Face and MLflow capabilities rather than building replacements.

**Rule 4 — No unnecessary infrastructure.** Prefer local or single-GPU execution unless an observed limitation requires more.

**Rule 5 — Require evidence.** Every performance optimization, model improvement or promotion decision needs measurable supporting results.

**Rule 6 — Control dependencies.** A later release must not prevent an earlier release from producing a usable result.

**Rule 7 — Record, don't chase, tangents.** Interesting discoveries outside the agreed scope go into a backlog rather than the implementation.

**Rule 8 — Freeze v1 at six releases.** No additional subsystem, model family or infrastructure platform is added without an explicit scope decision.

---

## 8. Intended Interview Evidence

At completion, the project should support concrete discussions about:

- Designing PyTorch data pipelines and batching.
- Working with pretrained models and fine-tuning.
- Training loops, evaluation and checkpoint recovery.
- Scientific data quality and validation.
- Experiment tracking and reproducibility.
- ML benchmarking and performance analysis.
- Model registries, promotion and governance.
- Model export and runtime compatibility.
- Deployment, observability and operational failure handling.

The final artifact is **one cohesive scientific ML system with a documented engineering history**, not a collection of tutorials.

**Project positioning:** Model-development and lifecycle engineering, complementing existing systems and GPU infrastructure experience.

**Scope principle:** Learn the model lifecycle by owning a real scientific ML problem from data to reproducible deployment.
