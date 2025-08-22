# Copied from repo root load_dataset.py
from datasets import load_from_disk, load_dataset
import json

def format_example_for_training(example):
    """Format a single example for model training (no task_type, full solution)."""
    
    # Create the input prompt
    prompt = (
        "Training Example 1:\n"
        f"Input: {json.dumps(example['train_input1'])}\n"
        f"Output: {json.dumps(example['train_output1'])}\n\n"
        "Training Example 2:\n"
        f"Input: {json.dumps(example['train_input2'])}\n"
        f"Output: {json.dumps(example['train_output2'])}\n\n"
        f"Test Input: {json.dumps(example['test_input'])}\n"
        "Test Output: "
    )

    # Get the solution
    sol = example.get('solution', '')
    if isinstance(sol, (list, tuple)):
        completion = "\n".join(str(line) for line in sol)
    else:
        completion = str(sol)
    
    return {
        'prompt': prompt,
        'completion': completion,
        'full_text': prompt + completion
    }

def prepare_dataset_for_training(dataset_path):
    """Load and prepare the dataset for training."""
    
    # Load from disk
    dataset = load_from_disk(dataset_path)
    
    # Apply formatting
    formatted_dataset = dataset.map(format_example_for_training)
    
    return formatted_dataset

# Example usage
if __name__ == "__main__":
    dataset = prepare_dataset_for_training("dsl_dataset/")
    
    # Show an example
    print("Example formatted for training:")
    print(dataset['train'][0]['prompt'][:500] + "...")
    print("\nExpected completion:")
    print(dataset['train'][0]['completion'])
