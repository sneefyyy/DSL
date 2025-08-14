
---
license: mit
task_categories:
- text-generation
language:
- en
size_categories:
- 10K<n<100K
---

# DSL ARC Dataset

Dataset for training models on ARC-like tasks using a Domain Specific Language.

## Dataset Structure

Each example contains:
- `train_input1`, `train_output1`: First training example
- `train_input2`, `train_output2`: Second training example  
- `test_input`, `test_output`: Test example
- `solution`: DSL code to solve the task
- `task_type`: Type of transformation

## Task Types

- Connect
- MoveShape
- RotateShape
- CreateShape
- FloodFill
- MirrorShape
- SymmetryComplete
- ExtractPattern
---
license: mit
task_categories:
	- text-generation
language:
	- en
size_categories:
	- 10K<n<100K
---

# DSL ARC — end-to-end README

This repository contains everything required to: generate DSL few-shot examples, build a deduplicated Hugging Face dataset, fine-tune a causal language model on those examples, and evaluate models both by exact text-match and by executing generated DSL in a sandboxed interpreter.

This README documents the pipeline, important implementation details, and how to run each step locally or in Colab (PEFT/LoRA + W&B supported).

## High-level pipeline

1. Generate many few-shot JSON examples using the generators in the `generators/` package.
2. Deduplicate and split the examples into train/validation/test and save a HF Dataset to disk (`dsl_dataset_v0.0.1/`) or push to the Hub.
3. Fine-tune a causal language model with `src/dsl/finetune_hf.py`. Training uses masked labels so the model conditions on a prompt and is only penalized for the completion.
4. Evaluate the model with two signals:
	 - Exact-match: string equality of the generated completion vs the reference completion.
	 - Functional (execution) match: run the generated DSL in a sandboxed process and compare produced output to expected output.

## Repository layout (important files)

- `generators/` — generator classes that synthesize few-shot JSON examples per task. Each generator implements a common interface and helper methods.
- `JSON_training/` — output from running the generation scripts (one JSON file per task).
- `src/dsl/create_hf_dataset.py` — loads JSON files, deduplicates, shuffles and splits (default 90/5/5), and saves a HF Dataset to disk (parquet + arrow shards).
- `src/dsl/finetune_hf.py` — main training script using Hugging Face `Trainer`. Supports optional PEFT/LoRA, W&B logging, prediction logging and functional evaluation callbacks.
- `src/dsl/evaluate_model.py` — evaluation runner that can generate many candidates and execute them in a sandboxed subprocess to compute functional metrics and pass@k.
- `src/dsl/interpreter.py` — the lightweight interpreter used to build the execution namespace (imports `generators` and exposes test input) and to validate examples locally.
- `upload_to_hf.py` / `src/dsl/upload_to_hf.py` — helpers to push the dataset to the Hub.

## Generators (how example JSON is produced)

Location: `generators/`

Design
- A base class (in `generators/base.py`) provides utilities for building and saving few-shot examples. Typical methods:
	- `create_fewshot_examples(num_examples)` — create N examples for a task. Each example is a dict with keys described in "Dataset fields" below.
	- `save_fewshot_examples(path, examples)` — helper to write a JSON file.

Subclasses
- Each task (e.g., `connect.py`, `move_shape.py`, `flood_fill.py`) implements a small class that generates examples for that task. Some generator subclasses include multiple helper functions (for example, both `create_shape` and `create_shape_helper` or generation helpers that compute candidate shapes). The important contract is that `create_fewshot_examples()` returns canonical examples.

Example output
- Generated JSON files live in `JSON_training/` (one file per task). Each line or object contains fields described below.

## Dataset fields (example format)

Each example in the dataset has these fields:
- `train_input1`, `train_output1` — first training example (grids or structured arrays).
- `train_input2`, `train_output2` — second training example.
- `test_input`, `test_output` — test example input and the expected output used for functional evaluation.
- `solution` — the canonical DSL solution as a string (or list of solution lines).
- `task_type` — name of the task/generator used (e.g., `Connect`, `MoveShape`).

The generator code produces natural-language-ish prompts and DSL solutions that the model is trained to produce.

## Creating the Hugging Face dataset

Script: `src/dsl/create_hf_dataset.py`

What it does
- Loads the JSON files from `JSON_training/`.
- Deduplicates examples by stable JSON serialization.
- Shuffles and splits into train/validation/test (default 90/5/5).
- Saves a Dataset to disk (arrow/parquet shards) under a versioned directory (e.g., `dsl_dataset_v0.0.1/`) and optionally uploads to the Hub.

Notes
- When the script detects a local directory path, `finetune_hf.py` will use `datasets.load_from_disk()` to correctly load the saved dataset and its features.

## How the interpreter works (execution for functional checks)

Files: `src/dsl/interpreter.py`, `src/dsl/evaluate_model.py`

Design
- The interpreter builds an execution namespace containing helpful classes and functions from `generators/` and sets `test_input` to the example's input.
- Candidate solutions produced by the model are executed in a subprocess or isolated namespace. There are two main execution modes in this repository:
	1. Local interpreter (`interpreter.py`) used for smoke-testing generator outputs. It imports `generators`, constructs the expected environment, executes solution code with `exec()`, and inspects known names (for example `output_grid` or `grid`) for produced output.
	2. Sandboxed subprocess executor (used by `evaluate_model.py` and `FunctionalEvaluatorCallback`) which runs each candidate in a separate OS process with a hard timeout. This prevents hangs and provides basic isolation.

Execution contract
- When executing, the runner expects the solution code to write the result into a known variable (for example `output_grid`), or to mutate a known object the interpreter can read back. The runner compares that output with `test_output` to decide functional success.

Security note
- Running generated code is inherently risky. The sandboxing used here (subprocess + timeout) reduces risk of hangs but is not a full security sandbox. For untrusted models run evaluations on isolated VMs/containers.

## Fine-tuning script internals (`src/dsl/finetune_hf.py`)

Overview
- The script fine-tunes a decoder-only (causal) LM using Hugging Face `Trainer`. Key features:
	- Formatting of examples into `prompt` and `completion` using `format_example_for_training_local()`.
	- Tokenization and label construction so loss is only computed on the completion portion.
	- A data collator that pads inputs and labels for batching.
	- `ExactMatchTrainer` (Trainer subclass) that computes exact-match percentages on validation and sampled training examples.
	- `PredictionLogger` callback: logs a small sample table of model predictions (prompt, target, prediction, exact) to W&B if enabled.
	- `FunctionalEvaluatorCallback` callback: at evaluation time it samples examples, generates candidates for each prompt, executes the candidates using the sandboxed runner, computes functional success and pass@k, logs scalars to the Trainer (and to W&B as a table) and prints a compact summary to the console.

Important implementation details

- Prompt/label construction
	- `format_example_for_training_local(example)` builds a text prompt containing the task name, two training examples (input/output), and the test input followed by the literal `"Test Output: "` marker.
	- The `completion` is the expected test output string (the DSL solution line(s)).
	- `tokenize_and_build_labels(example, tokenizer, max_length)` tokenizes prompt and completion separately. It builds `input_ids = prompt_ids + completion_ids + [eos?]` and `labels = [-100]*len(prompt_ids) + completion_ids + [eos?]`.
	- `-100` is the PyTorch/HF ignore index. Positions with `-100` are omitted from loss computation, so the model sees the prompt but loss is computed only on the completion tokens. This is standard teacher-forcing for decoder-only models.

- Data collator
	- `DataCollatorForCausalLMWithPadding` pads `input_ids` and `labels` to the batch max length and builds an `attention_mask`. Label padding uses `-100` so padded positions don't affect training loss.

- PEFT / LoRA support
	- The script optionally wraps the model with PEFT/LoRA when `--use_peft` is set. The import and instantiation are guarded so the script still runs without the `peft` package; if `peft` is unavailable the script logs a warning and continues with full fine-tuning.

- Logging & W&B
	- If `--use_wandb` is passed and `WANDB_API_KEY` is set, the script initializes W&B and logs training & evaluation metrics.
	- The following scalar metrics are logged (Trainer + W&B):
		- `eval_loss`, `train_loss`, `eval_exact_match_pct`, `train_exact_match_pct` (sampled), `functional_exact_pct`, `pass_at_N` (functional pass@k), `learning_rate`, `global_step`, `epoch`.
	- Two table artifacts are logged to W&B when enabled:
		- `predictions` (via `PredictionLogger`): columns `prompt`, `target`, `prediction`, `exact`, `step`.
		- `functional_predictions` (via `FunctionalEvaluatorCallback`): columns `prompt`, `target`, `prediction`, `functional`.

- SMOKE_TEST
	- Setting `SMOKE_TEST=1` in the environment causes the script to select a tiny sample from each split for a very fast smoke run useful in Colab or debugging.

## Evaluation & pass@k

- `src/dsl/evaluate_model.py` can generate many candidates per prompt and run them through the same sandboxed executor to compute exact and functional metrics and approximate pass@k.
- The `FunctionalEvaluatorCallback` computes a sampled functional success rate and a simple pass@k where `k` is the number of returned sequences per prompt. For more rigorous pass@k estimators (over large candidate pools) you can extend the evaluator to draw larger candidate sets.

## How to run (commands)

- Generate JSON few-shot examples (from repository root):
	- `python generate_examples.py`  — uses `generators/` to create JSON files under `JSON_training/`.

- Create the HF dataset and save to disk:
	- `python src/dsl/create_hf_dataset.py --save_dir dsl_dataset_v0.0.1`  — dedupes, splits 90/5/5 and writes parquet/arrow shards.

- (Optional) Upload the dataset to HF Hub:
	- `python src/dsl/upload_to_hf.py --dataset_dir dsl_dataset_v0.0.1 --repo_id middles/dsl-arc-dataset-v0.0.1`

- Quick smoke finetune (local; runs small sample when `SMOKE_TEST=1`):
	- `SMOKE_TEST=1 python3 src/dsl/finetune_hf.py --model_name_or_path gpt2 --dataset_id dsl_dataset_v0.0.1 --output_dir ./tmp-finetune --num_train_epochs 1 --per_device_train_batch_size 1 --use_wandb`.

- Full finetune (with PEFT + W&B):
	- `python3 src/dsl/finetune_hf.py --model_name_or_path gpt2 --dataset_id middles/dsl-arc-dataset-v0.0.1 --output_dir ./dsl-finetuned --num_train_epochs 3 --per_device_train_batch_size 4 --use_peft --use_wandb --wandb_project my-dsl-run`.

## Inspecting W&B output

- Online: open your W&B project page at `https://wandb.ai/<entity>/<project>` to see scalars, charts and Tables (`predictions` and `functional_predictions`).
- Offline: run the script with `WANDB_MODE=offline` (or set in the environment). Offline runs are stored under `./wandb/offline-run-*` and can be inspected in the notebook or re-uploaded later.

## Safety & performance notes

- Execution safety: the built-in sandbox is process-based with a short timeout. It's sufficient to avoid hangs during evaluation, but it does not protect against malicious or harmful code. For untrusted models run evaluation in an isolated VM or container and remove any network/file access.
- GPU & memory: use `--use_peft` on limited GPUs (Colab) to reduce VRAM usage. Consider `bitsandbytes` + `accelerate` for larger models and mixed precision.

## Developer tips

- When adding a new generator, implement `create_fewshot_examples()` returning examples obeying the dataset schema. Add a short test in `src/dsl/interpreter.py` to smoke-run a few examples.
- To extend functional evaluation (more rigorous pass@k), modify `src/dsl/evaluate_model.py` to generate larger candidate pools and use the standard pass@k estimator.

---
If you'd like, I can also:
- produce a Colab notebook pre-configured to run a PEFT finetune (already added as `colab_finetune_dsl.ipynb`), or
- add a CLI flag to `finetune_hf.py` to control `FunctionalEvaluatorCallback` sample size and candidate count for easier tuning.
