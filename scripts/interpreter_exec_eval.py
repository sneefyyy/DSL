#!/usr/bin/env python3
"""
Evaluate a (base + LoRA adapter) causal LM on the DSL ARC test split by:
  1. Building the same style prompt used during fine-tuning (chat markers <|user|>/<|assistant|>).\n  2. Generating a completion.\n  3. Extracting the first line containing `output_grid` between <|assistant|> and <|end|>.\n  4. Executing that line in a restricted namespace with generator classes available.\n  5. Comparing produced `output_grid` to the example's test_output.\n
Usage example:
  python scripts/interpreter_exec_eval.py \
    --base-repo microsoft/Phi-3-mini-4k-instruct \
    --adapter-repo middles/phi3-mini-arc-with-eval-1 \
    --limit 200 \
    --max-new-tokens 96 \
    --output-jsonl phi3_adapter_eval.jsonl

Creates two files:
  - <output-jsonl>: per-example records
  - <output-jsonl>.summary.json: aggregate metrics
"""
import os
import re
import json
import time
import argparse
import traceback
import inspect
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# ---------------------------------------------------------------------
# Prompt construction (mirrors training format, but we build raw prompt)
# ---------------------------------------------------------------------
def build_prompt(ex: Dict[str, Any]) -> str:
    return (
        "<|user|>\n"
        "Write Python code that transforms test_input into the expected test output. "
        "Only output executable Python lines setting output_grid.\n"
        "Training Example 1:\n"
        f"Input: {json.dumps(ex['train_input1'])}\n"
        f"Output: {json.dumps(ex['train_output1'])}\n\n"
        "Training Example 2:\n"
        f"Input: {json.dumps(ex['train_input2'])}\n"
        f"Output: {json.dumps(ex['train_output2'])}\n\n"
        f"Test Input: {json.dumps(ex['test_input'])}\n"
        "Test Output: <|end|>\n"
        "<|assistant|>"
    )

# ---------------------------------------------------------------------
# Model loading (base + LoRA adapter)
# ---------------------------------------------------------------------
def load_model_and_tokenizer(base_repo: str, adapter_repo: Optional[str], device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_repo, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(base_repo, device_map="auto", trust_remote_code=True)
    if adapter_repo:
        try:
            _ = PeftConfig.from_pretrained(adapter_repo)
            model = PeftModel.from_pretrained(model, adapter_repo)
            print(f"Loaded adapter from {adapter_repo}")
        except Exception as e:
            print(f"Adapter load failed ({e}); continuing with base model only.")
    model.eval()
    return model, tokenizer

# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------
def generate_code(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return text

# ---------------------------------------------------------------------
# Extract code after <|assistant|> up to <|end|>
# ---------------------------------------------------------------------
ASSISTANT_BLOCK_RE = re.compile(r"<\|assistant\|>(.*?)(?:<\|end\|>|$)", re.DOTALL)

def extract_output_grid_line(full_text: str) -> str:
    match = ASSISTANT_BLOCK_RE.findall(full_text)
    if not match:
        return ""
    assistant_segment = match[-1]
    for line in assistant_segment.strip().splitlines():
        if "output_grid" in line:
            return line.strip()
    return assistant_segment.strip()

# ---------------------------------------------------------------------
# Safe execution sandbox
# ---------------------------------------------------------------------
SAFE_BUILTINS = {
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "enumerate": enumerate,
    "print": print,
}

def build_execution_namespace() -> Dict[str, Any]:
    namespace: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    try:
        import generators  # noqa: F401
        for name in dir(generators):
            obj = getattr(generators, name)
            if inspect.isclass(obj) and name.endswith("Generator"):
                namespace[name] = obj
    except Exception:
        pass
    return namespace

def execute_snippet(snippet: str, test_input) -> Dict[str, Any]:
    ns = build_execution_namespace()
    ns["test_input"] = test_input
    result = {"ok": False, "error": None, "output_grid": None}
    try:
        exec(snippet, ns, ns)  # noqa: S102
        result["output_grid"] = ns.get("output_grid")
        result["ok"] = result["output_grid"] is not None
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result

# ---------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------
def evaluate(
    base_repo: str,
    adapter_repo: Optional[str],
    limit: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output_jsonl: str,
):
    model, tokenizer = load_model_and_tokenizer(base_repo, adapter_repo, device="cuda")
    print("Loading test dataset...")
    ds = load_dataset("middles/dsl-arc-dataset-v0.0.1", split="test")

    total = 0
    exact = 0
    errors = 0
    start_time = time.time()

    with open(output_jsonl, "w") as f:
        for ex in ds:
            if limit and total >= limit:
                break
            total += 1
            prompt = build_prompt(ex)
            gen_full = generate_code(model, tokenizer, prompt, max_new_tokens, temperature, top_p)
            code_line = extract_output_grid_line(gen_full)
            exec_res = execute_snippet(code_line, ex["test_input"])
            predicted = exec_res.get("output_grid")
            expected = ex.get("test_output")
            is_exact = predicted == expected
            if exec_res["error"]:
                errors += 1
            if is_exact:
                exact += 1

            record = {
                "id": ex.get("id", total),
                "prompt": prompt,
                "generated_raw": gen_full,
                "extracted_code": code_line,
                "exec_ok": exec_res["ok"],
                "exec_error": exec_res["error"],
                "predicted_output": predicted,
                "expected_output": expected,
                "exact_match": is_exact,
            }
            f.write(json.dumps(record) + "\n")

            if total % 25 == 0:
                print(f"[{total}] exact={exact} errors={errors} acc={exact/total:.3f}")

    duration = time.time() - start_time
    summary = {
        "total": total,
        "exact": exact,
        "errors": errors,
        "accuracy": exact / total if total else 0.0,
        "runtime_sec": duration,
        "samples_per_sec": total / duration if duration > 0 else 0.0,
    }
    print("Summary:", json.dumps(summary, indent=2))
    with open(output_jsonl + ".summary.json", "w") as sf:
        json.dump(summary, sf, indent=2)
    return summary

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-repo", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--adapter-repo", default="middles/phi3-mini-arc-with-eval-1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--output-jsonl", default="model_eval_results.jsonl")
    return ap.parse_args()

def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        evaluate(
            base_repo=args.base_repo,
            adapter_repo=args.adapter_repo,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            output_jsonl=args.output_jsonl,
        )
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
