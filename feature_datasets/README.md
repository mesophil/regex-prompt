# Feature Activation Dataset Generator

This script generates a dataset of feature activations from a language model using a Sparse Autoencoder (SAE). The output is stored in JSONL format, with each line containing the text, its tokenization, and the corresponding feature activations.

## Usage

```bash
python -m feature_datasets.generate_dataset
```

```bash
python -m feature_datasets.piviot_dataset feature_activations.jsonl features.jsonl --activation_threshold 10
```
