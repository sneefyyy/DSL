#!/usr/bin/env python3
"""Functional execution evaluation on the test split.

Usage:
  python scripts/functional_test_eval.py \
    --model_path ./dsl-finetuned \
    --dataset_id middles/dsl-arc-dataset \
    --pass_k 1 \
    --max_new_tokens 64

Computes pass@k style functional accuracy: a test example counts as correct if ANY of the k generated code candidates, when executed, produces the expected test_output.

Outputs a summary line and (optionally) per-example JSONL when --jsonl_out is provided.
"""
import argparse
import json
import os
import multiprocessing
import queue as _queue
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def format_example(example):
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
    return prompt


def make_exec_worker():
    def _worker(code, inp, q):
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
        except Exception as e:
            import traceback as tb
            q.put(('ERR', tb.format_exc()))
    return _worker


def run_exec(code_str, test_input, timeout, worker):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(code_str, test_input, q))
    p.start()
    try:
        res_type, res = q.get(timeout=timeout)
        p.join(timeout=0.05)
        return res_type, res
    except _queue.Empty:
        try:
            p.terminate()
        except Exception:
            pass
        return 'TIMEOUT', None


def evaluate(args):
    if os.path.isdir(args.dataset_id):
        from datasets import load_from_disk
        ds = load_from_disk(args.dataset_id)
    else:
        ds = load_dataset(args.dataset_id)
    if 'test' not in ds:
        raise ValueError('Dataset has no test split')
    test_ds = ds['test']
    if args.sample_n and args.sample_n < len(test_ds):
        test_ds = test_ds.select(range(args.sample_n))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token or tokenizer.sep_token or '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    # Re-enable cache for speed
    if hasattr(model, 'config'):
        try:
            model.config.use_cache = True
        except Exception:
            pass

    worker = make_exec_worker()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.pass_k > 1,
        num_return_sequences=args.pass_k,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    correct = 0
    total = 0
    jsonl_handle = open(args.jsonl_out, 'w') if args.jsonl_out else None
    start = time.time()

    for idx in range(len(test_ds)):
        ex = test_ds[idx]
        prompt = format_example(ex)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outs = model.generate(**inputs, **gen_kwargs)
        if isinstance(outs, torch.Tensor):
            outs_list = outs
        else:
            outs_list = outs
        input_len = inputs['input_ids'].shape[1]
        any_func = False
        cand_texts = []
        for seq in outs_list:
            seq_ids = seq[input_len:].cpu().tolist()
            cand = tokenizer.decode(seq_ids, skip_special_tokens=True).strip()
            cand_texts.append(cand)
            cand_lines = [l for l in cand.splitlines() if l.strip()]
            code_str = '\n'.join(cand_lines)
            r_type, r_val = run_exec(code_str, ex.get('test_input'), args.timeout, worker)
            if r_type == 'OK' and r_val == ex.get('test_output'):
                any_func = True
        total += 1
        if any_func:
            correct += 1
        if jsonl_handle:
            jsonl_handle.write(json.dumps({
                'index': idx,
                'prompt': prompt,
                'candidates': cand_texts,
                'functional': any_func
            }) + '\n')
        if total % args.log_every == 0:
            elapsed = time.time() - start
            print(f"Progress {total}/{len(test_ds)} functional={correct}/{total} ({correct/total*100:.2f}%) elapsed={elapsed:.1f}s")

    if jsonl_handle:
        jsonl_handle.close()

    pct = (correct / total) if total else 0.0
    print(f"FINAL functional pass@{args.pass_k}: {correct}/{total} = {pct*100:.2f}%")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', type=str, required=True, help='Fine-tuned model directory or Hub id')
    ap.add_argument('--dataset_id', type=str, default='middles/dsl-arc-dataset')
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--pass_k', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--timeout', type=float, default=2.0, help='Per candidate execution timeout (s)')
    ap.add_argument('--sample_n', type=int, default=None, help='If set, only evaluate first N test examples')
    ap.add_argument('--jsonl_out', type=str, default=None, help='Optional path to write per-example JSONL results')
    ap.add_argument('--log_every', type=int, default=25)
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
