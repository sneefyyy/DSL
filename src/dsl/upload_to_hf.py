from huggingface_hub import login
from datasets import load_from_disk

# Login to Hugging Face (will prompt for a token if needed)
login()

# Load the versioned dataset we created to avoid overwriting previous exports
dataset = load_from_disk("dsl_dataset_v0.0.1/")

# Push to hub using a versioned dataset name so we don't overwrite the main one
dataset.push_to_hub(
	"middles/dsl-arc-dataset-v0.0.1",
	private=True,
)

# Create a dataset card
dataset_card = """
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
- RepeatPattern
- CountAndTransform
"""

# Save dataset card
with open("README.md", "w") as f:
	f.write(dataset_card)

