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
	prompt = example['prompt']
	completion = example['completion']

	prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
	completion_ids = tokenizer.encode(completion, add_special_tokens=False)

	# Ensure there's room for completion; truncate prompt from the left if needed
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
	parser.add_argument('--model_name_or_path', type=str, default='gpt2')
	parser.add_argument('--dataset_id', type=str, default='middles/dsl-arc-dataset')
	parser.add_argument('--output_dir', type=str, default='./dsl-finetuned')
	parser.add_argument('--per_device_train_batch_size', type=int, default=4)
	parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
	parser.add_argument('--num_train_epochs', type=int, default=1)
	parser.add_argument('--max_steps', type=int, default=None, help='Override number of training steps (used for smoke tests).')
	parser.add_argument('--learning_rate', type=float, default=5e-5)
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
	args = parser.parse_args()

	logger.info('Loading dataset: %s', args.dataset_id)
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
	ds = ds.map(lambda ex: format_example_for_training_local(ex), batched=False)

	# Load tokenizer and model
	logger.info('Loading tokenizer and model: %s', args.model_name_or_path)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	# Ensure pad token exists
	if tokenizer.pad_token is None:
		tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token or tokenizer.sep_token or '<|pad|>'})

	model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
	model.resize_token_embeddings(len(tokenizer))

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
		try:
			from peft import get_peft_model, LoraConfig, TaskType
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
				task_type=TaskType.CAUSAL_LM,
			)
			model = get_peft_model(model, lora_config)
			logger.info('Enabled LoRA with r=%d alpha=%d on modules=%s', args.peft_r, args.peft_alpha, target_modules)
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

	training_kwargs = dict(
		output_dir=args.output_dir,
		overwrite_output_dir=True,
		num_train_epochs=args.num_train_epochs,
		per_device_train_batch_size=args.per_device_train_batch_size,
		per_device_eval_batch_size=args.per_device_eval_batch_size,
		eval_strategy='epoch' if eval_dataset is not None else 'no',
		save_strategy='epoch',
		learning_rate=args.learning_rate,
		weight_decay=0.01,
		logging_steps=50,
		report_to=report_to,
		push_to_hub=args.push_to_hub,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
	)

	if args.fp16:
		training_kwargs['fp16'] = True
	if args.bf16:
		training_kwargs['bf16'] = True
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

	callbacks = []
	if args.early_stopping_patience and args.early_stopping_patience > 0:
		callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

	# Add prediction logger callback to print and log example predictions at evaluation
	if args.use_wandb:
		callbacks.append(PredictionLogger(tokenizer=tokenizer, raw_datasets=ds, gen_kwargs=gen_kwargs, sample_n=5, use_wandb=True))

	# Functional evaluator callback: runs generated code in a subprocess sandbox and logs pass@k
	class FunctionalEvaluatorCallback(TrainerCallback):
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
	callbacks.append(FunctionalEvaluatorCallback(tokenizer=tokenizer, raw_datasets=ds, gen_kwargs=gen_kwargs, sample_n=50, num_return_sequences=1, exec_timeout=2.0, use_wandb=args.use_wandb))

	trainer = ExactMatchTrainer(
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

	# Optional test evaluation (was not automatic). We run this AFTER saving so it doesn't affect early stopping.
	if 'test' in tokenized:
		try:
			logger.info('Evaluating on test split')
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

	if args.push_to_hub:
		logger.info('Pushing model to the Hub')
		trainer.push_to_hub()


if __name__ == '__main__':
	main()

