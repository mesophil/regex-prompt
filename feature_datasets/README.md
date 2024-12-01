# Feature Activation Dataset Generator

This script generates a dataset of feature activations from a language model using a Sparse Autoencoder (SAE). The output is stored in JSONL format, with each line containing the text, its tokenization, and the corresponding feature activations.

## Requirements

```
transformer-lens
sae-lens
torch
transformers
datasets
```

## Usage

```python
from create_feature_dataset import create_feature_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Initialize model and SAE
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device=device,
)

# Create dataset
create_feature_dataset(
    dataset_path="NeelNanda/pile-10k",  # HuggingFace dataset path
    model=model,
    sae=sae,
    output_path="feature_activations.jsonl",
    max_examples=1000,  # Optional: limit number of examples
    activation_threshold=0.1  # Optional: minimum activation value
)
```

## Output Format

The script generates a JSONL file where each line is a JSON object with the following structure:

```javascript
{
    // The original text of the document
    "text": "This is an example document",
    
    // The token IDs from the tokenizer
    "tokens": [1234, 345, 456, ...],
    
    // Character-level offsets for each token in the text
    "offsets": [0, 5, 8, ...],
    
    // List of active features at each token position
    // Each inner list contains the feature indices that were active
    "active_features": [
        [1, 10001, 88822],  // Features active at first token
        [12323, 11324],     // Features active at second token
        ...
    ],
    
    // List of activation values corresponding to active_features
    // Each inner list contains the activation values for the active features
    "activations": [
        [0.10, 0.2, 10.0],  // Values for first token's features
        [0.23, 0.01],       // Values for second token's features
        ...
    ]
}
```

### Notes on the Format
- The length of `tokens` and `offsets` will be equal to the sequence length
- For each position i, `active_features[i]` and `activations[i]` will have the same length
- Only features with absolute activation values above `activation_threshold` are included
- Token offsets indicate the starting character position of each token in the original text

## Parameters

- `dataset_path`: HuggingFace dataset path to process
- `model`: Initialized HookedTransformer model
- `sae`: Initialized Sparse Autoencoder
- `output_path`: Where to save the JSONL file
- `max_examples`: Maximum number of examples to process (optional)
- `activation_threshold`: Minimum absolute value for an activation to be considered active (optional)
- `batch_size`: Number of examples to process at once (currently set to 1)

## Example Usage with Custom Parameters

```python
create_feature_dataset(
    dataset_path="NeelNanda/pile-10k",
    model=model,
    sae=sae,
    output_path="feature_activations.jsonl",
    max_examples=100,  # Process only 100 examples
    activation_threshold=0.5  # Higher threshold for stronger activations
)
```

## Reading the Dataset

You can read the dataset using standard JSON libraries:

```python
import json

# Read entire dataset into memory
with open("feature_activations.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

# Or process one example at a time
with open("feature_activations.jsonl", "r") as f:
    for line in f:
        example = json.loads(line)
        # Process example...
```