import json
import os
from datasets import Dataset, DatasetDict
import random

def load_json_files(json_dir):
    """Load all JSON files from the training directory.

    This function also tags each example with its task_type (from filename)
    and returns a list of examples.
    """
    all_examples = []

    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Add task type based on filename
                task_type = filename.replace('.json', '')
                for example in data:
                    example['task_type'] = task_type
                    all_examples.append(example)

    return all_examples

def create_hf_dataset(json_dir, train_frac=0.90, val_frac=0.05, seed=42):
    """Create a Hugging Face DatasetDict with train/validation/test splits.

    Splits are created as: train_frac, val_frac, and the remainder as test.
    The function removes exact-duplicate examples (by JSON serialization)
    before splitting to ensure no duplicate items across splits.
    """

    # Load all examples
    all_examples = load_json_files(json_dir)

    # Deduplicate examples by their JSON representation to avoid exact duplicates
    seen = set()
    unique_examples = []
    for ex in all_examples:
        # Use a stable serialization to detect duplicates
        try:
            key = json.dumps(ex, sort_keys=True)
        except TypeError:
            # Fallback: use str() if some objects aren't serializable
            key = str(ex)
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)

    # Shuffle for random split
    random.seed(seed)
    random.shuffle(unique_examples)

    n = len(unique_examples)
    if n == 0:
        # Return empty DatasetDict
        return DatasetDict({
            'train': Dataset.from_list([]),
            'validation': Dataset.from_list([]),
            'test': Dataset.from_list([]),
        })

    # Compute counts. Ensure train+val+test == n by letting test take the remainder.
    train_count = int(n * train_frac)
    val_count = int(n * val_frac)
    # Ensure at least one example per split when possible
    if n >= 3:
        if train_count == 0:
            train_count = max(1, n - 2)
        if val_count == 0:
            val_count = 1

    test_count = n - train_count - val_count
    if test_count < 0:
        # In rare rounding cases, fix by shifting to train
        test_count = 0
        train_count = n - val_count

    train_examples = unique_examples[:train_count]
    val_examples = unique_examples[train_count:train_count + val_count]
    test_examples = unique_examples[train_count + val_count:]

    # Convert to HF Dataset format
    """
    Compatibility wrapper: run src.dsl.create_hf_dataset as a module.
    """

    if __name__ == "__main__":
        import runpy
        runpy.run_module("src.dsl.create_hf_dataset", run_name="__main__")
        'validation': val_dataset,
