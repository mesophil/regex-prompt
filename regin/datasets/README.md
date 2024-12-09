# Feature Activation Dataset Generator

This script generates a dataset of feature activations from a language model using a Sparse Autoencoder (SAE). The output is stored in JSONL format, with each line containing the text, its tokenization, and the corresponding feature activations.

## Usage

```bash
python -m regin.datasets.generate
```

```bash
python -m regin.datasets.pivot data/gpt-2-small/train.jsonl data/gpt-2-small/examples/train.jsonl --act_thresh 10
```
