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
from transformers import EarlyStoppingCallback
import math
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_example_for_training_local(example):
	# Inline formatting mirroring load_dataset.format_example_for_training
	import json
	prompt = f"""Task: {example['task_type']}

Training Example 1:
Input: {json.dumps(example['train_input1'])}
Output: {json.dumps(example['train_output1'])}

Training Example 2:
Input: {json.dumps(example['train_input2'])}
Output: {json.dumps(example['train_output2'])}

Test Input: {json.dumps(example['test_input'])}
Test Output: """

	# Use the first solution line as the expected completion
	target = example['solution'][0] if isinstance(example.get('solution'), (list, tuple)) else example.get('solution', '')
	return {'prompt': prompt, 'completion': target}


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
	def __init__(self, *args, tokenizer_for_eval=None, raw_datasets=None, gen_kwargs=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokenizer_for_eval = tokenizer_for_eval
		self.raw_datasets = raw_datasets or {}
		self.gen_kwargs = gen_kwargs or {}

	def evaluate(self, eval_dataset=None, **kwargs):
		metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)

		# Compute exact-match on validation (if available)
		val_ds = self.raw_datasets.get('validation')
		if val_ds is None:
			return metrics

		model = self.model
		model.eval()
		device = next(model.parameters()).device

		correct = 0
		total = 0

		gen_kwargs = dict(self.gen_kwargs)

		for ex in val_ds:
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
				# list/tuple
				seq = out_ids[0]

			gen_text = self.tokenizer_for_eval.decode(seq, skip_special_tokens=True).strip()

			# Try to remove the prompt prefix from the generated text to get the completion
			comp = gen_text[len(prompt):].strip() if gen_text.startswith(prompt) else gen_text

			if comp == target:
				correct += 1
			total += 1

		exact_pct = (correct / total) if total else 0.0
		metrics['eval_exact_match_pct'] = exact_pct * 100.0
		print(f"Validation exact-match: {correct}/{total} = {exact_pct:.4f}")

		return metrics


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name_or_path', type=str, default='gpt2')
	parser.add_argument('--dataset_id', type=str, default='middles/dsl-arc-dataset')
	parser.add_argument('--output_dir', type=str, default='./dsl-finetuned')
	parser.add_argument('--per_device_train_batch_size', type=int, default=4)
	parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
	parser.add_argument('--num_train_epochs', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=5e-5)
	parser.add_argument('--max_length', type=int, default=1024)
	parser.add_argument('--push_to_hub', action='store_true')
	parser.add_argument('--early_stopping_patience', type=int, default=0, help='Number of evaluation calls with no improvement to wait before stopping. 0 disables early stopping')
	parser.add_argument('--early_stopping_metric', type=str, default='exact_match', choices=['exact_match', 'loss'], help='Metric to monitor for early stopping')
	args = parser.parse_args()

	logger.info('Loading dataset from hub: %s', args.dataset_id)
	ds = load_dataset(args.dataset_id)

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

	# Tokenize and build labels
	logger.info('Tokenizing and building labels (this may take a while)')
	def _tok(ex):
		return tokenize_and_build_labels(ex, tokenizer, max_length=args.max_length)

	tokenized = {}
	for split in ds.keys():
		tokenized_split = ds[split].map(lambda ex: _tok(ex), remove_columns=ds[split].column_names)
		tokenized[split] = tokenized_split

	# Convert dataset dict to DatasetDict-like object for Trainer input
	train_dataset = tokenized.get('train')
	eval_dataset = tokenized.get('validation')

	# Collator
	collator = DataCollatorForCausalLMWithPadding(tokenizer)

	# Training arguments
	training_args = TrainingArguments(
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
	push_to_hub=args.push_to_hub,
	load_best_model_at_end=(args.early_stopping_patience > 0),
	metric_for_best_model=('eval_exact_match_pct' if args.early_stopping_metric == 'exact_match' else 'eval_loss'),
	greater_is_better=(True if args.early_stopping_metric == 'exact_match' else False),
	)

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

	trainer = ExactMatchTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=collator,
		tokenizer_for_eval=tokenizer,
		raw_datasets=ds,
		gen_kwargs=gen_kwargs,
		callbacks=callbacks if callbacks else None,
	)

	logger.info('Starting training')
	trainer.train()

	logger.info('Saving model to %s', args.output_dir)
	trainer.save_model(args.output_dir)

	if args.push_to_hub:
		logger.info('Pushing model to the Hub')
		trainer.push_to_hub()


if __name__ == '__main__':
	main()
