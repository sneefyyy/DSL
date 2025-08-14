#!/usr/bin/env python3
"""
Evaluate a fine-tuned model on the DSL dataset using functional execution via interpreter.py.

The script supports local datasets (load_from_disk) or Hub datasets (load_dataset).
It generates candidate completions for each prompt, executes each candidate using
`interpreter.ExampleInterpreter.test_example()` and reports exact-match and
functional accuracy (pass@1 and pass@k).

Example:
  python evaluate_model.py --model_path /tmp/dsl-finetuned-smoke --dataset_path dsl_dataset --split test --num_samples 20 --num_return_sequences 3

"""
import argparse
import json
import os
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import traceback
import copy
import multiprocessing
import queue
import time
from src.dsl.interpreter import ExampleInterpreter
from src.dsl.load_dataset import format_example_for_training


def _solution_worker(code, inp, q):
	try:
		ns = {'__builtins__': __builtins__}
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


def generate_candidates(model, tokenizer, prompt, device, num_return_sequences=1, max_new_tokens=128, temperature=0.0, top_p=0.95, num_beams=1):
	# Tokenize prompt and generate only new tokens
	inputs = tokenizer(prompt, return_tensors='pt')
	input_ids = inputs.input_ids.to(device)
	attention_mask = inputs.attention_mask.to(device)

	gen_kwargs = {
		'max_new_tokens': max_new_tokens,
		'do_sample': False,
		'temperature': temperature,
		'top_p': top_p,
		'num_return_sequences': num_return_sequences,
		'num_beams': num_beams,
		'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
		'eos_token_id': tokenizer.eos_token_id,
	}

	# If requesting multiple sequences, fall back to sampling for diversity unless num_beams>1
	if num_return_sequences > 1 and num_beams <= 1:
		gen_kwargs['do_sample'] = True

	with torch.no_grad():
		outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

	candidates = []
	for out in outputs:
		# out is full sequence; extract generated portion after input_ids
		out = out.cpu().tolist()
		input_len = input_ids.shape[1]
		gen_ids = out[input_len:]
		text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
		candidates.append(text)

	return candidates


def evaluate(model_path, dataset_path=None, dataset_id=None, split='test', num_samples=100, num_return_sequences=1, max_new_tokens=128, temperature=0.0, top_p=0.95, num_beams=1, device=None, exec_timeout=2.0):
	device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	print("Loading model and tokenizer from:", model_path)
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

	# Load dataset
	if dataset_path and os.path.isdir(dataset_path):
		print(f"Loading dataset from disk: {dataset_path}")
		ds = load_from_disk(dataset_path)
	elif dataset_id:
		print(f"Loading dataset from hub: {dataset_id}, split={split}")
		ds = load_dataset(dataset_id)
	else:
		raise ValueError('Either dataset_path (local) or dataset_id (hub) must be provided')

	if split not in ds:
		raise ValueError(f"Split '{split}' not found in dataset. Available: {list(ds.keys())}")

	total = min(num_samples, len(ds[split]))
	print(f"Evaluating {total} examples from split '{split}'")

	interpreter = ExampleInterpreter()

	def run_solution_subprocess(code_str, test_input, timeout=2):
		"""
		Execute code_str in a subprocess with a fresh namespace and return (computed_output, exec_error).
		This avoids hanging the main process if generated code blocks or enters long loops.
		"""
		q = multiprocessing.Queue()
		p = multiprocessing.Process(target=_solution_worker, args=(code_str, test_input, q))
		p.start()
		try:
			res_type, res = q.get(timeout=timeout)
			p.join(timeout=0.1)
			if res_type == 'OK':
				return res, None
			else:
				return None, res
		except queue.Empty:
			# Timeout
			try:
				p.terminate()
			except Exception:
				pass
			return None, f"Timeout after {timeout}s"

	results = []
	exact_matches = 0
	functional_success = 0
	pass_at_k_success = 0

	for i in range(total):
		example = dict(ds[split][i])
		formatted = format_example_for_training(example)
		prompt = formatted['prompt']
		ground_truth = example.get('solution', [])
		# ground_truth may be list of lines; create ground_truth_str
		if isinstance(ground_truth, (list, tuple)):
			ground_truth_str = '\n'.join(ground_truth).strip()
		else:
			ground_truth_str = str(ground_truth).strip()

		candidates = generate_candidates(model, tokenizer, prompt, device, num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, num_beams=num_beams)

		any_functional = False
		any_exact = False
		candidate_results = []

		# Prepare namespace once per example for more consistent error capture
		base_namespace = interpreter.create_execution_namespace()

		for cand_idx, cand in enumerate(candidates):
			# Prepare a variant of example with this generated solution
			cand_lines = [line for line in cand.splitlines() if line.strip()]
			gen_example = dict(example)
			gen_example['solution'] = cand_lines

			# Execute solution using a fresh namespace built from interpreter to capture exceptions
			ns = interpreter.create_execution_namespace()
			ns['test_input'] = copy.deepcopy(example['test_input'])

			full_solution = '\n'.join(cand_lines)
			functional = False
			computed_output = None
			exec_error = None

			# Execute generated solution in a subprocess with timeout to avoid hangs
			computed_output, exec_error = run_solution_subprocess(full_solution, copy.deepcopy(example['test_input']), timeout=exec_timeout)
			if exec_error is None:
				functional = (computed_output == example.get('test_output'))

			exact = (cand.strip() == ground_truth_str)

			candidate_results.append({
				'candidate': cand,
				'exact_match': exact,
				'functional': functional,
				'computed_output': computed_output,
				'exec_error': exec_error,
			})

			if exact:
				any_exact = True
			if functional:
				any_functional = True

		if any_exact:
			exact_matches += 1
		# any_functional represents whether any of the returned candidates
		# succeeded functionally for this example. For num_return_sequences>1
		# this is the empirical pass@k (k = num_return_sequences).
		if any_functional:
			functional_success += 1
			pass_at_k_success += 1

		results.append({
			'index': i,
			'prompt_snippet': prompt[:500],
			'test_input': example.get('test_input'),
			'ground_truth': ground_truth_str,
			'candidates': candidate_results,
			'any_exact': any_exact,
			'any_functional': any_functional,
			'num_return_sequences': num_return_sequences,
		})

		# Print detailed per-example info when failures occur to help debugging
		if any_functional is False or any_exact is False:
			print(f"\nExample {i+1}/{total} — any_exact={any_exact} any_functional={any_functional}")
			print("Prompt snippet:\n", prompt[:500])
			# Print the full test input (grid) to avoid truncation when debugging
			try:
				print("Test Input (full):")
				print(json.dumps(example.get('test_input'), indent=2))
			except Exception:
				# Fallback to plain print if json serialization fails
				print("Test Input (full):", example.get('test_input'))
			for ci, cr in enumerate(candidate_results):
				print(f"\nCandidate {ci+1}:")
				print(cr['candidate'])
				print("  exact_match:", cr['exact_match'])
				print("  functional:", cr['functional'])
				if cr['computed_output'] is not None:
					print("  computed_output:", cr['computed_output'])
				if cr['exec_error']:
					print("  exec_error:\n", cr['exec_error'])

		# Print progress
		if (i + 1) % 10 == 0 or (i + 1) == total:
			print(f"Evaluated {i+1}/{total} — exact={exact_matches}/{i+1}, functional={functional_success}/{i+1}")

	print('\nFinal results:')
	print(f"Exact-match (1-best): {exact_matches}/{total} = {exact_matches/total:.4f}")
	print(f"Functional success (1-best): {functional_success}/{total} = {functional_success/total:.4f}")
	# Report empirical pass@k when multiple candidates were generated
	if num_return_sequences > 1:
		print(f"Pass@{num_return_sequences} (functional): {pass_at_k_success}/{total} = {pass_at_k_success/total:.4f}")

	# Save detailed results
	out_file = 'evaluation_results.jsonl'
	with open(out_file, 'w') as f:
		for r in results:
			f.write(json.dumps(r) + '\n')

	print(f"Detailed results saved to: {out_file}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, required=True)
	parser.add_argument('--dataset_path', type=str, default='dsl_dataset')
	parser.add_argument('--dataset_id', type=str, default=None)
	parser.add_argument('--split', type=str, default='test')
	parser.add_argument('--num_samples', type=int, default=20)
	parser.add_argument('--num_return_sequences', type=int, default=1)
	parser.add_argument('--max_new_tokens', type=int, default=128)
	parser.add_argument('--temperature', type=float, default=0.0)
	parser.add_argument('--top_p', type=float, default=0.95)
	parser.add_argument('--num_beams', type=int, default=1)
	parser.add_argument('--exec_timeout', type=float, default=2.0, help='Timeout in seconds for executing a single candidate')
	args = parser.parse_args()

	evaluate(args.model_path, dataset_path=args.dataset_path, dataset_id=args.dataset_id, split=args.split, num_samples=args.num_samples, num_return_sequences=args.num_return_sequences, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams, exec_timeout=args.exec_timeout)

