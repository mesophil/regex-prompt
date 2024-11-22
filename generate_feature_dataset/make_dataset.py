from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
import torch
from transformers import AutoTokenizer
import json
from typing import List
import numpy as np


def get_token_offsets(text: str, tokenizer: AutoTokenizer) -> List[int]:
    """
    Get character-level offsets for each token in the text.
    """
    encoding = tokenizer(text, return_offsets_mapping=True)
    return [offset[0] for offset in encoding['offset_mapping']]


def process_sae_activations(sae_output: torch.Tensor, threshold: float = 0.0) -> tuple[List[List[int]], List[List[float]]]:
    """
    Process SAE output tensor into lists of active features and their activations.
    
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
    output_path: str,
    max_examples: int = None,
    activation_threshold: float = 0.0
) -> None:
    """
    Create a JSONL dataset of feature activations.
    
    Args:
        dataset_path: HuggingFace dataset path
        model: TransformerLens model
        sae: Sparse autoencoder
        output_path: Path to save the JSONL file
        max_examples: Maximum number of examples to process
        activation_threshold: Minimum absolute value for an activation to be considered active
    """
    # Load dataset
    dataset = load_dataset(
        path=dataset_path,
        split="train",
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
    
    with open(output_path, 'w') as f:
        processed_examples = 0
        
        for example in token_dataset:
            if max_examples and processed_examples >= max_examples:
                break
                
            # Get tokens and text
            tokens = example["tokens"].to(device).unsqueeze(0)
            text = model.tokenizer.decode(tokens[0])
            
            # Get token offsets
            offsets = get_token_offsets(text, model.tokenizer)
            
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
            output_dict = {
                "text": text,
                "tokens": tokens[0].cpu().tolist(),
                "offsets": offsets,
                "active_features": active_features,
                "activations": activations
            }
            
            # Write to JSONL
            f.write(json.dumps(output_dict) + '\n')
            processed_examples += 1
            
            if processed_examples % 100 == 0:
                print(f"Processed {processed_examples} examples")


if __name__ == "__main__":
    # Initialize model and SAE as in your original script
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device,
    )

    # Create the dataset
    create_feature_dataset(
        dataset_path="NeelNanda/pile-10k",
        model=model,
        sae=sae,
        output_path="feature_activations.jsonl",
        max_examples=1000,  # Adjust as needed
        activation_threshold=0.1  # Adjust threshold as needed
    )