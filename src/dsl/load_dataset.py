# Copied from repo root load_dataset.py
from datasets import load_from_disk, load_dataset
import json

def format_example_for_training(example):
    """Format a single example for model training."""
    
    # Create the input prompt
    prompt = f"""Task: {example['task_type']}

Training Example 1:
Input: {json.dumps(example['train_input1'])}
Output: {json.dumps(example['train_output1'])}

Training Example 2:
Input: {json.dumps(example['train_input2'])}
Output: {json.dumps(example['train_output2'])}

Test Input: {json.dumps(example['test_input'])}
Test Output: """

    # The target/label is the solution
    target = example['solution'][0]  # Get the first solution line
    
    return {
        'prompt': prompt,
        'completion': target,
        'full_text': prompt + target
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
