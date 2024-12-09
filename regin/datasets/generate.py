from argparse import Namespace
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from transformers import AutoTokenizer

from regin.datatypes import save_jsonl

from ..datatypes import FeatureString


def get_tokens_and_offsets(text: str, tokenizer: AutoTokenizer) \
    -> tuple[List[int], List[int]]:
    """Get tokens and their character-level offsets for text."""
    encoding = tokenizer(text, return_offsets_mapping=True)
    return encoding['input_ids'], [offset[0] for offset in encoding['offset_mapping']]


def process_sae_activations(sae_output: torch.Tensor, threshold: float = 0.0) \
    -> tuple[List[List[int]], List[List[float]]]:
    """Process SAE output tensor into lists of active features and their activations.

    Args:
        sae_output: Tensor of shape (batch_size, seq_len, num_features)
        threshold: Minimum absolute value for an activation to be considered active

    Returns:
        Tuple of (active_features, activations) where each is a list of lists
    """
    # Convert to numpy for easier processing
    activations = sae_output.cpu().numpy()
    batch_size, seq_len, num_features = activations.shape
    
    active_features_list = []
    activation_values_list = []
    
    # Process each position in the sequence
    for pos in range(seq_len):
        # Get the activations for this position
        pos_activations = activations[0, pos]  # Using [0] since batch size is 1
        
        # Find active features (above threshold)
        active_mask = np.abs(pos_activations) > threshold
        active_indices = np.where(active_mask)[0]
        active_values = pos_activations[active_mask]
        
        # Convert to Python lists
        active_features_list.append(active_indices.tolist())
        activation_values_list.append(active_values.tolist())

    return active_features_list, activation_values_list


def create_feature_dataset(
    dataset_path: str,
    model: HookedTransformer,
    sae: SAE,
    max_examples: int = None,
    activation_threshold: float = 0.0,
    split: str = "train",
) -> None:
    """Create a JSONL dataset of feature activations.
    
    Args:
        dataset_path: HuggingFace dataset path
        model: TransformerLens model
        sae: Sparse autoencoder
        output_path: Path to save the JSONL file
        max_examples: Maximum number of examples to process
        activation_threshold: Minimum absolute value for an activation to be considered active
        split: Dataset split to load
    """
    # Load dataset
    dataset = load_dataset(
        path=dataset_path,
        split=split,
        streaming=False,
    )
    
    # Tokenize dataset
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=model.tokenizer,
        streaming=True,
        max_length=sae.cfg.context_size,
        add_bos_token=sae.cfg.prepend_bos,
    )

    device = next(model.parameters()).device

    processed_examples = 0

    pbar = tqdm(total=max_examples, desc="Processing examples")

    dataset = []
    for example in token_dataset:
        if max_examples and processed_examples >= max_examples:
            break
            
        # Get tokens and text
        raw_tokens = example["tokens"]
        
        raw_text = model.tokenizer.decode(raw_tokens, skip_special_tokens=False)
        tokens_list, offsets = get_tokens_and_offsets(raw_text, model.tokenizer)
        tokens = torch.tensor(tokens_list).to(device).unsqueeze(0)

        # Get SAE activations
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            sae_out = sae.encode(cache[sae.cfg.hook_name])
        
        # Process activations
        active_features, activations = process_sae_activations(
            sae_out,
            threshold=activation_threshold
        )

        # Create output dictionary
        feature_string = FeatureString(
            text=raw_text,
            tokens=tokens_list,
            offsets=offsets,
            active_features=active_features,
            activations=activations
        )

        # Write to JSONL
        dataset.append(feature_string)
        processed_examples += 1
        pbar.update(1)
    pbar.close()
    return dataset


def main(args: Namespace):
   # Initialize model and SAE as in your original script
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(args.model, device=device) # default "gpt2-small"

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=args.sae, # default "gpt2-small-res-jb"
        sae_id=args.sae_id, # default "blocks.8.hook_resid_pre",
        device=device,
    )

    # Create the dataset
    size = args.train_size + args.val_size + args.eval_size
    dataset = create_feature_dataset(
        model=model,
        sae=sae,
        dataset_path=args.dataset_path,# default"NeelNanda/pile-10k",
        max_examples=size,
        activation_threshold=args.act_thresh # default 0.0
    )
    assert len(dataset) == size

    # Partition into train, validation, and evaluation sets
    train_dataset = dataset[:args.train_size]
    val_dataset = dataset[args.train_size:args.train_size + args.val_size]
    eval_dataset = dataset[args.train_size + args.val_size:]

    # Save the datasets
    args.save_path.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.save_path / "train.jsonl", train_dataset)
    save_jsonl(args.save_path / "val.jsonl", val_dataset)
    save_jsonl(args.save_path/ "eval.jsonl", eval_dataset)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create feature dataset')

    parser.add_argument("--dataset_path", type=str, default="NeelNanda/pile-10k", help='HuggingFace dataset path')
    parser.add_argument("--model", type=str, default="gpt2-small", help='Model name or path')
    parser.add_argument("--sae", type=str, default="gpt2-small-res-jb", help='SAE release')
    parser.add_argument("--sae_id", type=str, default="blocks.8.hook_resid_pre", help='SAE ID')
    parser.add_argument("--save_path", type=Path, default=Path("data", "gpt-2-small"), help='Path to save the dataset')
    parser.add_argument('--train_size', type=int, default=1000, help='Size of training set')
    parser.add_argument('--val_size', type=int, default=1000, help='Size of validation set')
    parser.add_argument('--eval_size', type=int, default=8000, help='Size of evaluation set')
    parser.add_argument('--act_thresh', type=float, default=0.0, help='Activation threshold')

    args = parser.parse_args()

    main(args)
