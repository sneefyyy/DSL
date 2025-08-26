#!/usr/bin/env python3
"""
Fine-tune a causal language model on the `middles/dsl-arc-dataset` dataset.

Behavior:
- Loads the dataset from the Hub
- Formats examples into `prompt` and `completion` (uses the repo's formatting)
- Tokenizes prompt and completion separately, concatenates them and sets labels
  so that loss is only computed on the completion tokens
- Trains with Hugging Face Trainer API

Usage (example):
  python finetune_hf.py --model_name_or_path gpt2 --output_dir ./dsl-finetuned --num_train_epochs 3 --per_device_train_batch_size 4

Be cautious: running full training requires GPU and enough disk space.
"""

# ============================================================================
# READING GUIDE (advanced user focused)
# ============================================================================
# High-level pipeline flow:
#   1. CLI args parsed (see main()).
#   2. Dataset loaded (local path or hub) -> mapped to prompt/completion format.
#   3. (Optional) curriculum filtering & smoke test sub-sampling.
#   4. Tokenizer & model loaded (with optional 4/8-bit quant + FlashAttention + bf16).
#   5. Optional gradient checkpointing and PEFT/LoRA wrapping (k-bit prep first).
#   6. Dataset tokenized with label masking (-100 over prompt tokens) -> HF Datasets object.
#   7. Trainer (ExactMatchTrainer subclass) instantiated with:
#        - custom generation-based exact-match metric (validation + sampled train subset)
#        - optional callbacks: prediction samples, functional execution (runs code), sample printer
#   8. Training via Trainer; evaluation either per epoch or every N steps.
#   9. Optional final test exact-match + functional pass@k evaluation.
#  10. (Optional) push to Hub.
#
# Key design motivations:
#   - PROMPT / COMPLETION SPLIT: We mask prompt loss to focus optimization on solution tokens only.
#   - EXACT-MATCH VIA GENERATE: We generate instead of relying on teacher-forced logits to approximate real inference behavior.
#   - FUNCTIONAL EVAL: Sandbox `exec` of predicted Python to verify produced grid equals target (pragmatic correctness metric beyond string match).
#   - K-BIT + LoRA: QLoRA-style memory efficiency; `prepare_model_for_kbit_training` ensures gradients flow only through LoRA adapters + small norms.
#   - CURRICULUM (static first stage only): Allow external manual multi-stage runs without complicating Trainer loop.
#   - CALLBACK MODULARITY: Each concern (prediction logging / functional exec / train sample) isolated for easy removal.
#
# Recommended path to grok the file:
#   A. Skim CLI args to see feature surface & toggles.
#   B. Read `format_example_for_training_local` (prompt spec) & `tokenize_and_build_labels` (label masking logic).
#   C. Inspect quantization + LoRA block (model loading rationale + memory decisions).
#   D. Examine `ExactMatchTrainer.compute_exact_match` for eval pipeline.
#   E. Review `FunctionalEvaluatorCallback` for sandbox pattern and reasoning about risk/timeouts.
#   F. Finally scan the main() linear orchestration to see assembly ordering.
#
# Debug / validation tips (expert level):
#   - Verify mask correctness: run a tiny batch and assert all pre-completion token positions have label -100.
#   - Sanity-check generation: before training, generate on 1-2 examples to ensure decoding splits prompt from completion correctly.
#   - Confirm LoRA active: print number of trainable params (already logged) & ensure >0 but << total.
#   - When using 4-bit: inspect a base weight dtype (should be int8/4bit quantized module) while LoRA injected weights are fp16/bf16.
#   - Performance profiling: disable functional eval (`--disable_functional_eval`) to isolate pure LM training speed.
#   - Smoke run: set `SMOKE_TEST=1` env var + `--max_steps 20 --eval_every_steps 10` for quick end-to-end verification.
#
# Extension ideas:
#   - Add pass@k mid-training by adjusting FunctionalEvaluatorCallback num_return_sequences.
#   - Introduce dynamic curriculum by subclassing Trainer and overriding training loop (heavier change).
#   - Stream metrics to custom dashboard by adding another callback hooking on `on_log`.
#
# Safety / caveats:
#   - Executing model-generated code is inherently risky; we confine execution to a fresh process + limited timeout, BUT not a hardened sandbox.
#   - For hostile inputs or untrusted models, use a real sandbox (firejail, gvisor) or disable functional eval.
#
# Reading order markers below: look for "==== SECTION" banners for quick navigation.
# ============================================================================

# ---------------------------------------------------------------------------
# Robust temp directory fallback
# Some Colab / restricted environments can surface a FileNotFoundError when
# importing libraries (e.g. dill via datasets/multiprocess) if no writable
# temp directory is detected. We proactively create one and set TMPDIR/TEMP/TMP
# before importing those libs to avoid crashes like:
#   FileNotFoundError: No usable temporary directory found
# ---------------------------------------------------------------------------
import os as _early_os
try:
	_candidate_dirs = [
		_early_os.environ.get('TMPDIR'),
		'/tmp',
		'/content/tmp',  # Colab common path
		'./.tmp'
	]
	_usable = None
	for _d in _candidate_dirs:
		if not _d:
			continue
		try:
			_early_os.makedirs(_d, exist_ok=True)
			# Need write & execute perms
			if _early_os.access(_d, _early_os.W_OK | _early_os.X_OK):
				_usable = _d
				break
		except Exception:
			continue
	if _usable is None:
		# Last resort: attempt to create ./.tmp
		try:
			_early_os.makedirs('./.tmp', exist_ok=True)
			if _early_os.access('./.tmp', _early_os.W_OK | _early_os.X_OK):
				_usable = './.tmp'
		except Exception:
			pass
	if _usable:
		for _v in ('TMPDIR', 'TEMP', 'TMP'):
			if not _early_os.environ.get(_v):
				_early_os.environ[_v] = _usable
except Exception:
	# Silent – fallback not critical if environment already OK
	pass
import argparse
import logging
from datasets import load_dataset
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	Trainer,
	TrainingArguments,
)
from transformers import TrainerCallback
from transformers import EarlyStoppingCallback
import math
import os
import torch
import multiprocessing
import queue
import copy
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_example_for_training_local(example):
	"""Format a single example into prompt/completion.

	There is no task_type field anymore. The prompt shows two training IO pairs and
	then the test input with a trailing 'Test Output: ' cue. The completion is the
	ENTIRE solution (all lines joined by newlines) so the model learns the full
	multi-line program, not only the first line.
	"""
	# NOTE: We intentionally embed exactly two train examples + one test input.
	# Rationale: few-shot pattern induction while controlling prompt length.
	# If you want dynamic k-shot, you'd extend this to sample k examples per task.
	import json as _json
	prompt = (
		"Training Example 1:\n"
		f"Input: {_json.dumps(example['train_input1'])}\n"
		f"Output: {_json.dumps(example['train_output1'])}\n\n"
		"Training Example 2:\n"
		f"Input: {_json.dumps(example['train_input2'])}\n"
		f"Output: {_json.dumps(example['train_output2'])}\n\n"
		f"Test Input: {_json.dumps(example['test_input'])}\n"
		"Test Output: "
	)
	sol = example.get('solution', '')
	if isinstance(sol, (list, tuple)):
		completion = "\n".join(str(line) for line in sol)
	else:
		completion = str(sol)
	return {'prompt': prompt, 'completion': completion, 'full_text': prompt + completion}


def tokenize_and_build_labels(example, tokenizer, max_length=1024):
	# Tokenize prompt and completion separately so we can mask the prompt tokens
	# Advanced note: We avoid packing multiple examples per sequence for simplicity;
	# if throughput is a bottleneck you could implement sequence packing (careful with label masks).
	prompt = example['prompt']
	completion = example['completion']

	prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
	completion_ids = tokenizer.encode(completion, add_special_tokens=False)

	# Ensure there's room for completion; truncate prompt from the left if needed
	# Left-side truncation keeps the most recent (closest) demonstration examples if prompt too long.
	total_needed = len(prompt_ids) + len(completion_ids) + 1  # +1 for eos
	if total_needed > max_length:
		keep = max_length - len(completion_ids) - 1
		if keep <= 0:
			# prompt completely truncated, shorten completion if still needed
			completion_ids = completion_ids[: max(0, max_length - 1)]
			prompt_ids = []
		else:
			prompt_ids = prompt_ids[-keep:]

	input_ids = prompt_ids + completion_ids + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])

	labels = ([-100] * len(prompt_ids)) + completion_ids + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])

	return {'input_ids': input_ids, 'labels': labels}


class DataCollatorForCausalLMWithPadding:
	def __init__(self, tokenizer, label_pad_token_id=-100):
		self.tokenizer = tokenizer
		self.label_pad_token_id = label_pad_token_id

	def __call__(self, features):
		# We perform manual padding (rather than using HF default) to keep explicit control
		# over label padding semantics and avoid inadvertently adding special tokens.
		input_ids = [torch.tensor(f['input_ids'], dtype=torch.long) for f in features]
		labels = [torch.tensor(f['labels'], dtype=torch.long) for f in features]

		input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
		labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)

		attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()

		return {'input_ids': input_ids_padded, 'attention_mask': attention_mask, 'labels': labels_padded}


class ExactMatchTrainer(Trainer):
	"""Trainer subclass that computes exact-match percent on the validation split
	by running generation with the provided tokenizer and comparing the generated
	completion (text after the prompt) to the reference `completion` field.
	"""
	# Design emphasis: evaluate realistic inference path (autoregressive generate) rather than
	# teacher-forced next-token accuracy. This penalizes early hallucination / divergence.
	def __init__(self, *args, tokenizer_for_eval=None, raw_datasets=None, gen_kwargs=None, use_wandb=False, train_eval_samples=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokenizer_for_eval = tokenizer_for_eval
		self.raw_datasets = raw_datasets or {}
		self.gen_kwargs = gen_kwargs or {}
		# Whether to attempt logging to Weights & Biases when available
		self.use_wandb = use_wandb
		# How many training examples to sample when computing train exact-match (None => full)
		self.train_eval_samples = train_eval_samples

	def compute_exact_match(self, split_name, max_samples=None):
		"""Compute exact match percent for a given split in raw_datasets.

		Returns (correct, total, pct)
		"""
		# Perf note: naive loop + generate; if validation set large you can
		#   a) sample (already supported), b) batch generation with padding.
		# We keep it scalar for clarity / debugging introspection.
		ds = self.raw_datasets.get(split_name)
		if ds is None:
			return 0, 0, 0.0

		model = self.model
		model.eval()
		device = next(model.parameters()).device

		gen_kwargs = dict(self.gen_kwargs)

		correct = 0
		total = 0

		# If requested, sample a subset of examples to speed up evaluation
		it = ds
		if max_samples is not None and max_samples < len(ds):
			import random as _random
			indices = _random.sample(range(len(ds)), max_samples)
			it = (ds[i] for i in indices)

		for ex in it:
			# Single-example generate (greedy by default) — deterministic exact-match metric.
			prompt = ex.get('prompt')
			target = ex.get('completion', '').strip()
			if prompt is None:
				continue

			inputs = self.tokenizer_for_eval(prompt, return_tensors='pt')
			inputs = {k: v.to(device) for k, v in inputs.items()}

			with torch.no_grad():
				out_ids = model.generate(**inputs, **gen_kwargs)

			if isinstance(out_ids, torch.Tensor):
				seq = out_ids[0].cpu().tolist()
			else:
				seq = out_ids[0]

			gen_text = self.tokenizer_for_eval.decode(seq, skip_special_tokens=True).strip()
			comp = gen_text[len(prompt):].strip() if gen_text.startswith(prompt) else gen_text

			if comp == target:
				correct += 1
			total += 1

		pct = (correct / total) if total else 0.0
		return correct, total, pct


class PredictionLogger(TrainerCallback):
	"""Callback to log model predictions (and exact-match per example) at evaluation time.

	Logs a small table to W&B and prints a compact summary to console.
	"""
	def __init__(self, tokenizer, raw_datasets, gen_kwargs=None, sample_n=5, use_wandb=False):
		self.tokenizer = tokenizer
		self.raw_datasets = raw_datasets or {}
		self.gen_kwargs = gen_kwargs or {}
		self.sample_n = sample_n
		self.use_wandb = use_wandb

	def on_evaluate(self, args, state, control, **kwargs):
		# Runs AFTER Trainer evaluation; we do lightweight sampling to inspect qualitative drift.
		model = kwargs.get('model')
		if model is None:
			return

		ds = self.raw_datasets.get('validation') or self.raw_datasets.get('test') or self.raw_datasets.get('train')
		if ds is None or len(ds) == 0:
			return

		import random
		indices = random.sample(range(len(ds)), min(self.sample_n, len(ds)))

		rows = []
		correct_count = 0
		total = 0

		device = next(model.parameters()).device

		for i in indices:
			ex = ds[i]
			prompt = ex.get('prompt')
			target = ex.get('completion', '').strip()
			if not prompt:
				continue

			inputs = self.tokenizer(prompt, return_tensors='pt')
			inputs = {k: v.to(device) for k, v in inputs.items()}

			with torch.no_grad():
				out_ids = model.generate(**inputs, **self.gen_kwargs)

			if isinstance(out_ids, torch.Tensor):
				seq = out_ids[0].cpu().tolist()
			else:
				seq = out_ids[0]

			gen_text = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
			pred = gen_text[len(prompt):].strip() if gen_text.startswith(prompt) else gen_text

			is_exact = (pred == target)
			if is_exact:
				correct_count += 1
			total += 1

			rows.append({'prompt': prompt, 'target': target, 'prediction': pred, 'exact': is_exact})

		# Console summary
		print(f"\n[PredictionLogger] sample_eval step={getattr(state, 'global_step', None)} exact={correct_count}/{total}")
		for r in rows:
			print(f"- exact={r['exact']} pred={r['prediction']!r} target={r['target']!r}")

		# W&B logging
		if self.use_wandb:
			try:
				import wandb
				table = wandb.Table(columns=["prompt", "target", "prediction", "exact", "step"])
				step = getattr(state, 'global_step', None)
				for r in rows:
					table.add_data(r['prompt'], r['target'], r['prediction'], r['exact'], step)
				wandb.log({"predictions": table}, step=step)
			except Exception:
				pass

	def evaluate(self, eval_dataset=None, **kwargs):
		# First get the normal HF metrics
		metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)

		# Compute validation exact-match
		c_v, t_v, pct_v = self.compute_exact_match('validation')
		metrics['eval_exact_match_pct'] = pct_v * 100.0
		print(f"Validation exact-match: {c_v}/{t_v} = {pct_v:.4f}")

		# Compute (sampled) training exact-match to track overfitting
		max_train_samples = self.train_eval_samples
		c_tr, t_tr, pct_tr = self.compute_exact_match('train', max_samples=max_train_samples)
		metrics['train_exact_match_pct'] = pct_tr * 100.0
		print(f"Train exact-match (sampled={max_train_samples}): {c_tr}/{t_tr} = {pct_tr:.4f}")

		# Optionally log to Weights & Biases if available
		if self.use_wandb:
			try:
				import wandb as _wandb
				step = getattr(self.state, 'global_step', None)
				# Include additional useful metrics when available
				lr = None
				try:
					if hasattr(self, 'optimizer') and self.optimizer is not None and len(self.optimizer.param_groups) > 0:
						lr = float(self.optimizer.param_groups[0].get('lr', None))
				except Exception:
					lr = None

				log_payload = {
					'eval_exact_match_pct': metrics.get('eval_exact_match_pct'),
					'train_exact_match_pct': metrics.get('train_exact_match_pct'),
					'eval_loss': metrics.get('eval_loss'),
					'learning_rate': lr,
					'global_step': getattr(self.state, 'global_step', None),
					'epoch': getattr(self.state, 'epoch', None),
				}
				if step is not None:
					_wandb.log(log_payload, step=step)
				else:
					_wandb.log(log_payload)
			except Exception:
				# Ignore wandb failures here; we still return metrics
				pass

		return metrics


def main():
	parser = argparse.ArgumentParser()
	# ==== SECTION: CLI ARGUMENTS ==================================================
	# The CLI surface is intentionally verbose to allow iterative experimentation
	# without editing code. Grouped conceptually (model, dataset, PEFT, quant, eval, perf).
	parser.add_argument('--model_name_or_path', type=str, default='gpt2')
	parser.add_argument('--dataset_id', type=str, default='middles/dsl-arc-dataset')
	parser.add_argument('--output_dir', type=str, default='./dsl-finetuned')
	parser.add_argument('--per_device_train_batch_size', type=int, default=4)
	parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
	parser.add_argument('--num_train_epochs', type=int, default=1)
	parser.add_argument('--max_steps', type=int, default=None, help='Override number of training steps (used for smoke tests).')
	parser.add_argument('--learning_rate', type=float, default=5e-5)
	# Scheduler / warmup (helps stabilize early steps and reduce gibberish)
	parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='LR scheduler type (linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, inverse_sqrt, reduce_lr_on_plateau)')
	parser.add_argument('--warmup_ratio', type=float, default=0.0, help='Warmup ratio (fraction of total training steps). Ignored if --warmup_steps is set.')
	parser.add_argument('--warmup_steps', type=int, default=None, help='Number of warmup steps (overrides --warmup_ratio when provided).')
	parser.add_argument('--max_length', type=int, default=1024)
	parser.add_argument('--push_to_hub', action='store_true')
	parser.add_argument('--use_wandb', action='store_true', help='Enable logging to Weights & Biases')
	parser.add_argument('--wandb_project', type=str, default='dsl-finetuning', help='W&B project name')
	parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
	parser.add_argument('--train_eval_samples', type=int, default=100, help='Number of train examples to sample when computing train exact-match (use 0 or negative to use full train set)')
	parser.add_argument('--early_stopping_patience', type=int, default=0, help='Number of evaluation calls with no improvement to wait before stopping. 0 disables early stopping')
	parser.add_argument('--early_stopping_metric', type=str, default='exact_match', choices=['exact_match', 'loss'], help='Metric to monitor for early stopping')
	# PEFT/LoRA options
	parser.add_argument('--use_peft', action='store_true', help='Use PEFT/LoRA for parameter-efficient fine-tuning')
	parser.add_argument('--peft_r', type=int, default=8, help='LoRA rank')
	parser.add_argument('--peft_alpha', type=int, default=32, help='LoRA alpha')
	parser.add_argument('--peft_target_modules', type=str, default=None, help='Comma-separated list of module names for LoRA to target (optional)')
	parser.add_argument('--peft_auto_target', action='store_true', help='Automatically detect common attention/MLP linear modules for LoRA if target modules not provided')
	parser.add_argument('--peft_dropout', type=float, default=0.05, help='LoRA dropout (helps regularize when using higher rank)')
	parser.add_argument('--trust_remote_code', action='store_true', help='Pass trust_remote_code=True when loading tokenizer/model (needed for some repos like Qwen)')
	# Quantization / memory flags for large models
	parser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4-bit (bnb) precision (QLoRA style)')
	parser.add_argument('--load_in_8bit', action='store_true', help='Load model in 8-bit (bnb) precision')
	parser.add_argument('--bnb_compute_dtype', type=str, default='bfloat16', help='Compute dtype for 4-bit/8-bit (e.g. bfloat16, float16)')
	parser.add_argument('--bnb_4bit_quant_type', type=str, default='nf4', choices=['nf4','fp4'], help='4-bit quantization data type')
	parser.add_argument('--bnb_4bit_use_double_quant', action='store_true', help='Enable double quantization in 4-bit mode')
	parser.add_argument('--device_map', type=str, default=None, help="Optional device map for model loading (e.g. 'auto' to spread across GPUs).")
	# Lightweight smoke-test sampling BEFORE tokenization
	parser.add_argument('--limit_train_samples', type=int, default=None, help='If set, limit number of raw train examples before tokenization (smoke test).')
	parser.add_argument('--limit_eval_samples', type=int, default=None, help='If set, limit number of raw validation examples before tokenization (smoke test).')
	parser.add_argument('--tpu_num_cores', type=int, default=None, help='Number of TPU cores (set when running on TPU with torch_xla).')
	# Memory / performance knobs
	parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing to reduce activation memory')
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps (effective batch = per_device_batch * this)')
	parser.add_argument('--fp16', action='store_true', help='Enable fp16 mixed precision training')
	parser.add_argument('--bf16', action='store_true', help='Enable bf16 mixed precision training (ampere+ / TPU)')
	parser.add_argument('--adam8bit', action='store_true', help='Use bitsandbytes 8-bit AdamW optimizer (reduces memory)')
	parser.add_argument('--adam8bit_paged', action='store_true', help='Use bitsandbytes paged 8-bit AdamW (better for long seq)')
	parser.add_argument('--reduce_max_length', type=int, default=None, help='If set, override --max_length with a smaller value for memory relief')
	# Evaluation overhead control
	parser.add_argument('--functional_eval_sample_n', type=int, default=50, help='Sample size for functional evaluator per eval (lower to speed up)')
	parser.add_argument('--disable_functional_eval', action='store_true', help='Disable functional execution evaluation callback entirely')
	parser.add_argument('--disable_prediction_logging', action='store_true', help='Disable lightweight per-eval textual prediction table (does NOT affect functional eval). Speeds eval if you only care about executed code metrics.')
	# Curriculum learning (length-based) options
	parser.add_argument('--curriculum_length_boundaries', type=str, default=None, help='Comma-separated ascending max completion token lengths for curriculum stages (e.g. 200,400,800). If set, training starts with shortest examples and gradually includes longer ones.')
	parser.add_argument('--curriculum_min_epochs_per_stage', type=int, default=1, help='Minimum epochs to spend on each curriculum stage before expanding')
	parser.add_argument('--curriculum_merge_incremental', action='store_true', help='If set, each new stage ADDS longer examples (cumulative). Otherwise the stage REPLACES previous subset.')
	# Evaluation cadence
	parser.add_argument('--eval_every_steps', type=int, default=None, help='If set (>0), run evaluation every N steps instead of each epoch.')
	# Performance / system flags
	parser.add_argument('--tf32', action='store_true', help='Enable TF32 matmul precision (Ampere+ GPUs)')
	parser.add_argument('--dataloader_num_workers', type=int, default=0, help='Number of DataLoader worker processes')
	parser.add_argument('--dataloader_pin_memory', action='store_true', help='Pin host memory in DataLoader for faster HtoD copies')
	parser.add_argument('--use_flash_attention_2', action='store_true', help='Attempt to load model with Flash Attention 2 (requires supported architecture & installed kernels)')
	parser.add_argument('--logging_steps', type=int, default=50, help='Log training loss every N steps')
	parser.add_argument('--skip_final_test', action='store_true', help='Skip final test exact-match/loss evaluation after training (do it manually later)')
	# Final functional test evaluation
	parser.add_argument('--final_functional_test', action='store_true', help='At end of training, run functional execution evaluation on the test split (executes generated code)')
	parser.add_argument('--final_functional_pass_k', type=int, default=1, help='Number of candidates (pass@k) for final functional test evaluation when --final_functional_test is set')
	parser.add_argument('--final_functional_timeout', type=float, default=2.0, help='Per-candidate execution timeout (seconds) for final functional test evaluation')
	parser.add_argument('--final_functional_sample_n', type=int, default=None, help='If set, sample this many test examples for final functional test (omit for full set)')
	parser.add_argument('--eval_temp_enable_cache', action='store_true', help='Temporarily enable model.config.use_cache during generation in evaluations even if disabled for training (speeds up eval generation)')
	# Console sample printing
	parser.add_argument('--print_train_example_every', type=int, default=0, help='If >0, every N optimizer steps print a random training example prompt + target + current model prediction.')
	args = parser.parse_args()

	logger.info('Loading dataset: %s', args.dataset_id)
	# ==== SECTION: DATA LOADING ===================================================
	# Supports either a local saved dataset directory (load_from_disk) or hub dataset id.
	# If the dataset_id is a local path saved via Dataset.save_to_disk, use load_from_disk
	if os.path.isdir(args.dataset_id):
		try:
			from datasets import load_from_disk
			ds = load_from_disk(args.dataset_id)
			logger.info('Loaded dataset from disk: %s', args.dataset_id)
		except Exception:
			logger.info('Falling back to load_dataset for: %s', args.dataset_id)
			ds = load_dataset(args.dataset_id)
	else:
		ds = load_dataset(args.dataset_id)

	# Optional pre-tokenization subsampling for smoke tests
	if args.limit_train_samples is not None and 'train' in ds:
		logger.info('Limiting train split to first %d examples for smoke test', args.limit_train_samples)
		try:
			from datasets import Dataset
			ds['train'] = ds['train'].select(range(min(args.limit_train_samples, len(ds['train']))))
		except Exception:
			pass
	if args.limit_eval_samples is not None and 'validation' in ds:
		logger.info('Limiting validation split to first %d examples for smoke test', args.limit_eval_samples)
		try:
			ds['validation'] = ds['validation'].select(range(min(args.limit_eval_samples, len(ds['validation']))))
		except Exception:
			pass

	# Map formatting to prompt/completion
	logger.info('Formatting examples (prompt/completion)')
	# After this map, each example acquires prompt/completion/full_text fields used downstream.
	ds = ds.map(lambda ex: format_example_for_training_local(ex), batched=False)

	# Optional curriculum: we won't stage dynamically during one run (Trainer lacks native hooks here without custom loop)
	# Instead we implement a static single-stage subset selection for early experimentation OR
	# allow user to manually iterate stages externally (
	#   run with first boundary, then resume/continue with next etc.).
	if args.curriculum_length_boundaries:
		try:
			bounds = [int(b.strip()) for b in args.curriculum_length_boundaries.split(',') if b.strip()]
			bounds = [b for b in bounds if b > 0]
			if bounds:
				first_bound = bounds[0]
				logger.info('Applying curriculum stage 1 filter: keeping examples with completion length <= %d tokens (approx via tokenizer)', first_bound)
				# Approx token length cheaply: tokenize completion only; may slightly under/over count due to special tokens.
				def _len_filter(ex):
					comp = ex.get('completion','')
					# Use simple whitespace split if tokenizer not yet loaded; we have tokenizer earlier but safe fallback
					return len(comp.split())  # coarse; faster than full tokenization of entire dataset
				for split in list(ds.keys()):
					orig_n = len(ds[split])
					filtered = ds[split].filter(lambda ex: _len_filter(ex) <= first_bound)
					logger.info('Curriculum stage 1: %s %d -> %d examples (<= %d tokens)', split, orig_n, len(filtered), first_bound)
					ds[split] = filtered
		except Exception as e:
			logger.warning('Failed to apply curriculum filter: %s', e)

	# Load tokenizer and model
	logger.info('Loading tokenizer and model: %s', args.model_name_or_path)
	# ==== SECTION: TOKENIZER ======================================================
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
	# Ensure pad token exists
	if tokenizer.pad_token is None:
		tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token or tokenizer.sep_token or '<|pad|>'})

	model_kwargs = {}
	# BitsAndBytes quantization config if requested
	# ==== SECTION: QUANTIZATION ===================================================
	# We configure BitsAndBytes BEFORE model load so that `from_pretrained` instantiates
	# quantized Linear modules directly (avoids full fp16/fp32 memory spike).
	if args.load_in_4bit or args.load_in_8bit:
		try:
			import importlib
			from transformers import BitsAndBytesConfig
			import torch as _torch  # local alias to avoid any scope confusion
			if not importlib.util.find_spec('bitsandbytes'):
				raise ImportError('bitsandbytes not installed. Install with: pip install bitsandbytes accelerate')
			# Resolve compute dtype safely
			compute_dtype = getattr(_torch, args.bnb_compute_dtype, _torch.bfloat16)
			bnb_config = BitsAndBytesConfig(
				load_in_4bit=args.load_in_4bit,
				load_in_8bit=(args.load_in_8bit and not args.load_in_4bit),
				bnb_4bit_compute_dtype=compute_dtype,
				bnb_4bit_quant_type=args.bnb_4bit_quant_type,
				bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
			)
			model_kwargs['quantization_config'] = bnb_config
			logger.info('Enabled bitsandbytes quantization: 4bit=%s 8bit=%s type=%s compute=%s double_quant=%s', args.load_in_4bit, args.load_in_8bit, args.bnb_4bit_quant_type, args.bnb_compute_dtype, args.bnb_4bit_use_double_quant)
			# Helpful diagnostics
			if _torch.cuda.is_available():
				try:
					cc = _torch.cuda.get_device_capability(0)
					logger.info('CUDA device capability: %s, torch cuda: %s, bitsandbytes expected >= 7.0 for 4-bit', cc, getattr(_torch.version, 'cuda', 'unknown'))
				except Exception:
					pass
		except Exception as _e:
			logger.warning('Could not set up bitsandbytes quantization (%s). Proceeding without quant. (Ensure compatible CUDA, bitsandbytes version, and GPU capability >=7.0)', _e)
	if args.use_flash_attention_2:
		# Some architectures allow specifying attn_implementation
		model_kwargs['attn_implementation'] = 'flash_attention_2'
	# If bf16 requested, set dtype at load to avoid initial fp32 weights (helps flash attention warning & memory)
	if getattr(args, 'bf16', False):
		try:
			import torch as _torch
			model_kwargs['torch_dtype'] = _torch.bfloat16
			logger.info('Loading model directly in bfloat16 dtype')
		except Exception:
			pass
	# Optional device_map handling (e.g. --device_map auto)
	if args.device_map:
		model_kwargs['device_map'] = args.device_map
	try:
		# Primary load path with advanced kwargs (quantization, flash attn, dtype, device map, etc.)
		model = AutoModelForCausalLM.from_pretrained(
			args.model_name_or_path,
			trust_remote_code=args.trust_remote_code,
			low_cpu_mem_usage=True,
			**model_kwargs,
		)
	except Exception as e:
		# Fallback path: drop advanced knobs except quantization to maximize success chance.
		logger.warning('Model load with provided kwargs failed (%s). Retrying without advanced kwargs.', e)
		fallback_kwargs = {k: v for k, v in model_kwargs.items() if k in ('quantization_config',)}
		model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, **fallback_kwargs)
	model.resize_token_embeddings(len(tokenizer))

	# TF32 enable
	if args.tf32:
		try:
			import torch
			if torch.cuda.is_available():
				torch.backends.cuda.matmul.allow_tf32 = True
				torch.backends.cudnn.allow_tf32 = True
				logger.info('Enabled TF32 matmul/cudnn')
		except Exception as e:
			logger.warning('Could not enable TF32: %s', e)

	# Apply optional sequence length reduction early
	if args.reduce_max_length is not None and args.reduce_max_length > 0 and args.reduce_max_length < args.max_length:
		logger.info('Overriding max_length from %d to %d for memory reduction', args.max_length, args.reduce_max_length)
		args.max_length = args.reduce_max_length

	# Gradient checkpointing (must disable use_cache for some architectures)
	if args.gradient_checkpointing:
		try:
			model.gradient_checkpointing_enable()
			if hasattr(model, 'config') and getattr(model.config, 'use_cache', None) is True:
				model.config.use_cache = False
			logger.info('Enabled gradient checkpointing and disabled use_cache')
		except Exception as e:
			logger.warning('Could not enable gradient checkpointing: %s', e)

	# Optionally apply PEFT/LoRA
	if args.use_peft:
		# ==== SECTION: PEFT / LoRA ==================================================
		# Sequence: (1) prepare model for k-bit (if quantized) -> (2) select target Linear modules -> (3) wrap with LoRA.
		# Any failure keeps base model untouched (logged as warning) so experiments still proceed.
		try:
			from peft import get_peft_model, LoraConfig, TaskType
			# For k-bit training (4/8bit) prepare model (enables input gradients, casts layer norms, etc.)
			if (args.load_in_4bit or args.load_in_8bit):
				try:
					from peft import prepare_model_for_kbit_training
					model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
					logger.info('Prepared model for k-bit training (input gradients & layer norm casting).')
				except Exception as prep_e:
					logger.warning('prepare_model_for_kbit_training failed or unavailable: %s', prep_e)
			# Determine target modules
			target_modules = None
			if args.peft_target_modules:
				target_modules = [m.strip() for m in args.peft_target_modules.split(',') if m.strip()]
			elif args.peft_auto_target:
				# Auto-detect linear submodules typical for attention/MLP
				import torch.nn as nn
				candidate_substrings = ['q_proj','k_proj','v_proj','o_proj','wq','wk','wv','wo','gate_proj','up_proj','down_proj','fc1','fc2','proj']
				seen = set()
				for name, module in model.named_modules():
					if not isinstance(module, nn.Linear):
						continue
					if any(s in name for s in candidate_substrings):
						# Use the final segment as module name reference
						leaf = name.split('.')[-1]
						seen.add(leaf)
						if len(seen) >= 64:  # safety bound
							break
				target_modules = sorted(seen) if seen else None
				logger.info('Auto-detected LoRA target modules: %s', target_modules)
			if target_modules is None:
				logger.warning('LoRA target modules not specified/detected; this will fine-tune all parameters (high memory). Provide --peft_target_modules or --peft_auto_target.')
			lora_config = LoraConfig(
				r=args.peft_r,
				lora_alpha=args.peft_alpha,
				target_modules=target_modules,
				lora_dropout=args.peft_dropout,
				task_type=TaskType.CAUSAL_LM,
			)
			model = get_peft_model(model, lora_config)
			logger.info('Enabled LoRA with r=%d alpha=%d on modules=%s', args.peft_r, args.peft_alpha, target_modules)
			# Sanity check: ensure we actually have trainable parameters
			trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
			total = sum(p.numel() for p in model.parameters())
			logger.info('Trainable parameters: %s (%.2f%% of total %s)', f'{trainable:,}', 100.0*trainable/total if total else 0.0, f'{total:,}')
			if trainable == 0:
				raise ValueError('No trainable parameters detected after LoRA initialization. Specify --peft_target_modules or disable 4bit/8bit quantization.')
		except Exception as e:
			logger.warning('PEFT/LoRA requested but could not be enabled: %s', e)

	# Optionally initialize Weights & Biases for logging
	if args.use_wandb:
		try:
			import wandb
			# Prompt for credentials if not already logged in
			try:
				wandb.login()
			except Exception:
				# Non-interactive environments may fail prompting; proceed and let HF integration handle it
				pass
			wandb.init(project=args.wandb_project, name=args.wandb_run_name)
		except Exception as e:
			logger.warning('Could not initialize wandb: %s', e)

	# Tokenize and build labels
	logger.info('Tokenizing and building labels (this may take a while)')
	# Tokenization is intentionally late (after LoRA) because resizing embeddings must occur before
	# creating batches whose pad token id depends on possibly newly-added pad embedding.
	def _tok(ex):
		return tokenize_and_build_labels(ex, tokenizer, max_length=args.max_length)

	tokenized = {}
	for split in ds.keys():
		tokenized_split = ds[split].map(lambda ex: _tok(ex), remove_columns=ds[split].column_names)
		tokenized[split] = tokenized_split

	# Optional smoke-test: sample a few examples from each split to speed up quick runs
	if os.environ.get('SMOKE_TEST') == '1':
		logger.info('SMOKE_TEST=1 detected — sampling a few examples from each split for a quick run')
		for split in list(tokenized.keys()):
			n = min(5, len(tokenized[split]))
			tokenized[split] = tokenized[split].select(range(n))

	# Convert dataset dict to DatasetDict-like object for Trainer input
	train_dataset = tokenized.get('train')
	eval_dataset = tokenized.get('validation')

	# Collator
	collator = DataCollatorForCausalLMWithPadding(tokenizer)

	# Training arguments
	# Configure `report_to` to include wandb when requested.
	# Use an explicit list; if not using W&B, leave empty to avoid requiring tensorboard.
	report_to = ['wandb'] if args.use_wandb else []

	if args.eval_every_steps and args.eval_every_steps > 0:
		# Step-based eval/save enables overlapped curves (train/eval have same x-axis resolution) for W&B dashboards.
		eval_strategy = 'steps'
		save_strategy = 'steps'
		eval_steps = args.eval_every_steps
		save_steps = args.eval_every_steps
	else:
		eval_strategy = 'epoch' if eval_dataset is not None else 'no'
		save_strategy = 'epoch'
		eval_steps = None
		save_steps = None

	training_kwargs = dict(
		output_dir=args.output_dir,
		overwrite_output_dir=True,
		num_train_epochs=args.num_train_epochs,
		per_device_train_batch_size=args.per_device_train_batch_size,
		per_device_eval_batch_size=args.per_device_eval_batch_size,
		eval_strategy=eval_strategy,
		save_strategy=save_strategy,
		learning_rate=args.learning_rate,
		lr_scheduler_type=args.lr_scheduler_type,
		weight_decay=0.01,
		logging_steps=args.logging_steps,
		report_to=report_to,
		push_to_hub=args.push_to_hub,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		dataloader_num_workers=args.dataloader_num_workers,
		dataloader_pin_memory=args.dataloader_pin_memory,
	)
	if eval_steps:
		training_kwargs['eval_steps'] = eval_steps
	if save_steps:
		training_kwargs['save_steps'] = save_steps

	if args.fp16:
		training_kwargs['fp16'] = True
	if args.bf16:
		training_kwargs['bf16'] = True

	# Warmup configuration
	if args.warmup_steps is not None and args.warmup_steps > 0:
		training_kwargs['warmup_steps'] = args.warmup_steps
	elif args.warmup_ratio and args.warmup_ratio > 0:
		# Let HF handle warmup_ratio directly if supported; fallback to manual compute otherwise
		try:
			training_kwargs['warmup_ratio'] = args.warmup_ratio
		except Exception:
			# Manual approximate compute (may differ slightly from actual due to rounding)
			try:
				if 'max_steps' in training_kwargs:
					total_steps = training_kwargs['max_steps']
				else:
					_world = int(os.environ.get('WORLD_SIZE','1'))
					if train_dataset is not None:
						num_examples = len(train_dataset)
						per_device = args.per_device_train_batch_size
						accum = args.gradient_accumulation_steps
						total_steps = math.ceil(num_examples / (per_device * _world * accum)) * args.num_train_epochs
					training_kwargs['warmup_steps'] = int(total_steps * args.warmup_ratio)
			except Exception:
				pass
	if args.adam8bit or args.adam8bit_paged:
		# Attempt to use bitsandbytes optimizer if installed
		optim_name = 'paged_adamw_8bit' if args.adam8bit_paged else 'adamw_bnb_8bit'
		try:
			import bitsandbytes as bnb  # noqa: F401
			training_kwargs['optim'] = optim_name
			logger.info('Using bitsandbytes optimizer: %s', optim_name)
		except Exception:
			logger.warning('bitsandbytes not available; falling back to default optimizer')

	if args.early_stopping_patience and args.early_stopping_patience > 0:
		training_kwargs['load_best_model_at_end'] = True
		training_kwargs['metric_for_best_model'] = ('eval_exact_match_pct' if args.early_stopping_metric == 'exact_match' else 'eval_loss')
		training_kwargs['greater_is_better'] = (True if args.early_stopping_metric == 'exact_match' else False)
	else:
		training_kwargs['load_best_model_at_end'] = False

	# Allow overriding max_steps for smoke tests; if provided set num_train_epochs large enough
	if args.max_steps is not None and args.max_steps > 0:
		training_kwargs['max_steps'] = args.max_steps
		# Avoid saving too frequently in a tiny run; disable epoch-based saves if steps override is used
		training_kwargs['save_strategy'] = 'no'
		logger.info('Smoke test mode: overriding max_steps=%d', args.max_steps)

	training_args = TrainingArguments(**training_kwargs)

	# If TPU requested, set attribute (HF Trainer will route to XLA devices if torch_xla installed)
	if args.tpu_num_cores:
		setattr(training_args, 'tpu_num_cores', args.tpu_num_cores)

	gen_kwargs = {
		'max_new_tokens': 64,
		'do_sample': False,
		'num_return_sequences': 1,
		'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
		'eos_token_id': tokenizer.eos_token_id,
	}
	# NOTE: Generation kwargs tuned for deterministic exact-match; adjust for diversity in functional pass@k.

	callbacks = []
	if args.early_stopping_patience and args.early_stopping_patience > 0:
		callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

	# Add prediction logger callback to print and log example predictions at evaluation
	if args.use_wandb and not args.disable_prediction_logging:
		callbacks.append(PredictionLogger(tokenizer=tokenizer, raw_datasets=ds, gen_kwargs=gen_kwargs, sample_n=5, use_wandb=True))

	# Lightweight periodic train sample printer
	if args.print_train_example_every and args.print_train_example_every > 0:
		class TrainSamplePrinterCallback(TrainerCallback):
			def __init__(self, raw_datasets, tokenizer, gen_kwargs, every):
				self.raw_datasets = raw_datasets
				self.tokenizer = tokenizer
				self.gen_kwargs = gen_kwargs
				self.every = every
			def on_step_end(self, args, state, control, **kwargs):
				if state.global_step is None or state.global_step == 0:
					return
				if state.global_step % self.every != 0:
					return
				train_ds = self.raw_datasets.get('train')
				if not train_ds or len(train_ds) == 0:
					return
				# Random example
				import random
				ex = dict(train_ds[random.randrange(len(train_ds))])
				prompt = ex.get('prompt')
				target = ex.get('completion','').strip()
				if not prompt:
					return
				model = kwargs.get('model')
				if model is None:
					return
				device = next(model.parameters()).device
				inputs = self.tokenizer(prompt, return_tensors='pt')
				inputs = {k: v.to(device) for k,v in inputs.items()}
				with torch.no_grad():
					out_ids = model.generate(**inputs, **self.gen_kwargs)
				if isinstance(out_ids, torch.Tensor):
					seq = out_ids[0].cpu().tolist()
				else:
					seq = out_ids[0]
				full_text = self.tokenizer.decode(seq, skip_special_tokens=True)
				pred = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
				# Truncate for readability
				def _trunc(s, n=180):
					return (s[:n] + '…') if len(s) > n else s
				print(f"\n[TrainSample step={state.global_step}]\nPrompt(fragment): {_trunc(prompt)}\nTarget(fragment): {_trunc(target)}\nPred(fragment): {_trunc(pred)}\n---")
		callbacks.append(TrainSamplePrinterCallback(raw_datasets=ds, tokenizer=tokenizer, gen_kwargs=gen_kwargs, every=args.print_train_example_every))

	# Functional evaluator callback: runs generated code in a subprocess sandbox and logs pass@k
	class FunctionalEvaluatorCallback(TrainerCallback):
		# ==== SECTION: FUNCTIONAL EVALUATOR =========================================
		# Purpose: Evaluate *semantic* correctness by executing generated code.
		# Mechanics: For a sample of validation examples we:
		#   - Recreate prompt
		#   - Generate candidate solution (greedy by default mid-training)
		#   - Exec candidate in isolated process (timeout) & capture produced grid
		#   - Compare to target test_output -> functional success
		# Security: Minimal. Not safe for arbitrary untrusted code. Use a hardened sandbox in production.
		def __init__(self, tokenizer, raw_datasets, gen_kwargs=None, sample_n=50, num_return_sequences=1, exec_timeout=2.0, use_wandb=False):
			self.tokenizer = tokenizer
			self.raw_datasets = raw_datasets or {}
			self.gen_kwargs = gen_kwargs or {}
			self.sample_n = sample_n
			self.num_return_sequences = num_return_sequences
			self.exec_timeout = exec_timeout
			self.use_wandb = use_wandb

			# worker for subprocess execution
			def _worker(code, inp, q):
				try:
					ns = {'__builtins__': __builtins__}
					# Import generators if available into namespace
					try:
						import generators
						for name in dir(generators):
							if name.startswith('_'):
								continue
							try:
								ns[name] = getattr(generators, name)
							except Exception:
								continue
					except Exception:
						pass
					ns['test_input'] = inp
					exec(code, ns)
					out = ns.get('output_grid', ns.get('grid', None))
					q.put(('OK', out))
				except Exception:
					import traceback as tb
					q.put(('ERR', tb.format_exc()))

			self._worker = _worker

		def run_solution_subprocess(self, code_str, test_input, timeout=2.0):
			# Spawns a short-lived process; canonical pattern to avoid hangs / infinite loops.
			q = multiprocessing.Queue()
			p = multiprocessing.Process(target=self._worker, args=(code_str, test_input, q))
			p.start()
			try:
				res_type, res = q.get(timeout=timeout)
				p.join(timeout=0.1)
				if res_type == 'OK':
					return res, None
				else:
					return None, res
			except queue.Empty:
				try:
					p.terminate()
				except Exception:
					pass
				return None, f"Timeout after {timeout}s"

		def on_evaluate(self, args, state, control, **kwargs):
			# Called after each evaluation; we piggyback on that cadence for functional diagnostics.
			trainer = kwargs.get('trainer')
			model = kwargs.get('model') or (trainer.model if trainer else None)
			if model is None:
				return

			ds = self.raw_datasets.get('validation') or self.raw_datasets.get('test') or self.raw_datasets.get('train')
			if ds is None or len(ds) == 0:
				return

			import random
			indices = random.sample(range(len(ds)), min(self.sample_n, len(ds)))

			functional_success = 0
			passk_success = 0
			total = 0

			rows = []
			device = next(model.parameters()).device

			for i in indices:
				ex = dict(ds[i])
				formatted = None
				try:
					# Use same formatting
					from src.dsl.load_dataset import format_example_for_training
					formatted = format_example_for_training(ex)
				except Exception:
					continue

				prompt = formatted['prompt']
				candidates = []
				# Generate candidates (we'll use model.generate directly)
				inputs = self.tokenizer(prompt, return_tensors='pt')
				inputs = {k: v.to(device) for k, v in inputs.items()}
				with torch.no_grad():
					outs = model.generate(**inputs, **self.gen_kwargs, num_return_sequences=self.num_return_sequences)
				# outs may be tensor with multiple sequences
				for out in outs:
					out = out.cpu().tolist()
					input_len = inputs['input_ids'].shape[1]
					gen_ids = out[input_len:]
					text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
					candidates.append(text)

				any_functional = False
				for cand in candidates:
					cand_lines = [line for line in cand.splitlines() if line.strip()]
					full_solution = '\n'.join(cand_lines)
					computed_output, exec_error = self.run_solution_subprocess(full_solution, copy.deepcopy(ex.get('test_input')), timeout=self.exec_timeout)
					functional = False
					if exec_error is None:
						functional = (computed_output == ex.get('test_output'))
					rows.append({'prompt': prompt, 'target': ex.get('solution'), 'prediction': cand, 'functional': functional})
					if functional:
						any_functional = True

				if any_functional:
					functional_success += 1
					passk_success += 1
				total += 1

			# Compute rates
			functional_pct = (functional_success / total) if total else 0.0
			passk_pct = (passk_success / total) if total else 0.0

			metrics = {
				'functional_exact_pct': functional_pct * 100.0,
				f'pass_at_{self.num_return_sequences}': passk_pct * 100.0,
			}

			# Log to Trainer if available
			if trainer is not None:
				try:
					trainer.log(metrics)
				except Exception:
					pass

			# Log to W&B table if requested
			if self.use_wandb:
				try:
					import wandb
					table = wandb.Table(columns=["prompt", "target", "prediction", "functional"])
					for r in rows:
						table.add_data(r['prompt'], json.dumps(r['target']), r['prediction'], r['functional'])
					wandb.log({"functional_predictions": table}, step=getattr(state, 'global_step', None))
				except Exception:
					pass

	# Register functional evaluator
	if not args.disable_functional_eval:
		callbacks.append(FunctionalEvaluatorCallback(tokenizer=tokenizer, raw_datasets=ds, gen_kwargs=gen_kwargs, sample_n=args.functional_eval_sample_n, num_return_sequences=1, exec_timeout=2.0, use_wandb=args.use_wandb))

	trainer = ExactMatchTrainer(
		# ==== SECTION: TRAINER INSTANTIATION ========================================
		# Provide raw datasets & tokenizer separately for custom generation metrics; keep HF datasets tokenized.
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=collator,
		tokenizer_for_eval=tokenizer,
		raw_datasets=ds,
		gen_kwargs=gen_kwargs,
		use_wandb=args.use_wandb,
		train_eval_samples=(None if args.train_eval_samples <= 0 else args.train_eval_samples),
		callbacks=callbacks if callbacks else None,
	)

	logger.info('Starting training')
	trainer.train()

	logger.info('Saving model to %s', args.output_dir)
	trainer.save_model(args.output_dir)

	# Optional final test evaluation (skip when --skip_final_test)
	if not args.skip_final_test and 'test' in tokenized:
		# Final exact-match & loss on holdout; distinct path from mid-train eval to avoid confusion with trainer state.
		try:
			logger.info('Evaluating on test split (final)')
			test_dataset = tokenized['test']
			raw_correct, raw_total, raw_pct = trainer.compute_exact_match('test')
			# Run standard evaluate() to get loss (will return eval_loss key)
			test_metrics = trainer.evaluate(eval_dataset=test_dataset)
			if 'eval_loss' in test_metrics:
				test_metrics['test_loss'] = test_metrics.pop('eval_loss')
			test_metrics['test_exact_match_pct'] = raw_pct * 100.0
			logger.info('Test exact-match: %d/%d = %.4f', raw_correct, raw_total, raw_pct)
			# Log to W&B if enabled
			if args.use_wandb:
				try:
					import wandb as _wandb
					_wandb.log({k: v for k, v in test_metrics.items() if k.startswith('test_')})
				except Exception:
					pass
		except Exception as e:
			logger.warning('Test evaluation failed: %s', e)

	# Final functional execution evaluation (pass@k) if requested
	if args.final_functional_test and 'test' in ds:
		# Final pass@k evaluation optionally with sampling for diversity when k>1.
		logger.info('Starting final functional execution evaluation on test split (pass@%d)', args.final_functional_pass_k)
		import random, time, multiprocessing, queue as _queue, copy as _copy
		model = trainer.model
		model.eval()
		device = next(model.parameters()).device
		orig_use_cache = None
		if args.eval_temp_enable_cache and hasattr(model, 'config'):
			try:
				orig_use_cache = getattr(model.config, 'use_cache', None)
				model.config.use_cache = True
				logger.info('Temporarily enabled use_cache for evaluation generation')
			except Exception:
				pass

		gen_kwargs_final = dict(gen_kwargs)
		gen_kwargs_final['num_return_sequences'] = max(1, args.final_functional_pass_k)
		# Enable sampling if k>1 for diverse candidates
		if args.final_functional_pass_k > 1:
			gen_kwargs_final.setdefault('do_sample', True)
			gen_kwargs_final.setdefault('temperature', 0.8)
			gen_kwargs_final.setdefault('top_p', 0.95)

		def _exec_worker(code, inp, q):
			try:
				ns = {'__builtins__': __builtins__}
				try:
					import generators
					for name in dir(generators):
						if name.startswith('_'): continue
						try:
							ns[name] = getattr(generators, name)
						except Exception:
							continue
				except Exception:
					pass
				ns['test_input'] = inp
				exec(code, ns)
				out = ns.get('output_grid', ns.get('grid', None))
				q.put(('OK', out))
			except Exception:
				import traceback as _tb
				q.put(('ERR', _tb.format_exc()))

		def run_exec(code_str, test_input, timeout):
			q = multiprocessing.Queue()
			p = multiprocessing.Process(target=_exec_worker, args=(code_str, test_input, q))
			p.start()
			try:
				res_type, res = q.get(timeout=timeout)
				p.join(timeout=0.05)
				return res_type, res
			except _queue.Empty:
				try: p.terminate()
				except Exception: pass
				return 'TIMEOUT', None

		test_raw = ds['test']
		indices = list(range(len(test_raw)))
		if args.final_functional_sample_n is not None and args.final_functional_sample_n < len(indices):
			indices = random.sample(indices, args.final_functional_sample_n)

		functional_pass = 0
		passk_pass = 0
		processed = 0
		start_time = time.time()

		for idx in indices:
			ex = dict(test_raw[idx])
			prompt = ex.get('prompt')
			if not prompt:
				continue
			inputs = tokenizer(prompt, return_tensors='pt')
			inputs = {k: v.to(device) for k, v in inputs.items()}
			with torch.no_grad():
				outs = model.generate(**inputs, **gen_kwargs_final)
			if isinstance(outs, torch.Tensor):
				outs_list = outs
			else:
				outs_list = outs
			input_len = inputs['input_ids'].shape[1]
			any_func = False
			for seq in outs_list:
				seq_ids = seq[input_len:].cpu().tolist()
				cand = tokenizer.decode(seq_ids, skip_special_tokens=True).strip()
				cand_lines = [l for l in cand.splitlines() if l.strip()]
				code_str = '\n'.join(cand_lines)
				res_type, res = run_exec(code_str, _copy.deepcopy(ex.get('test_input')), timeout=args.final_functional_timeout)
				if res_type == 'OK' and res == ex.get('test_output'):
					any_func = True
			if any_func:
				functional_pass += 1
				passk_pass += 1
			processed += 1
			if processed % 20 == 0:
				elapsed = time.time() - start_time
				logger.info('Final functional eval progress %d/%d (%.1fs elapsed)', processed, len(indices), elapsed)

		functional_pct = (functional_pass/processed) if processed else 0.0
		passk_pct = (passk_pass/processed) if processed else 0.0
		logger.info('Final functional test results: functional_exact=%.2f%% pass@%d=%.2f%% on %d examples', functional_pct*100.0, args.final_functional_pass_k, passk_pct*100.0, processed)
		if args.use_wandb:
			try:
				import wandb as _wandb
				_wandb.log({'final_functional_exact_pct': functional_pct*100.0, f'final_pass_at_{args.final_functional_pass_k}': passk_pct*100.0, 'final_functional_examples': processed})
			except Exception:
				pass
		if orig_use_cache is not None:
			try:
				model.config.use_cache = orig_use_cache
			except Exception:
				pass

	if args.push_to_hub:
		logger.info('Pushing model to the Hub')
		trainer.push_to_hub()


if __name__ == '__main__':
	main()

