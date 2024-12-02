import json
import re

def process_activating_examples_single_feature(data) -> str:
    """
    Extract the most important token (final token) from each activating example and create a regex pattern.

    Args:
        data (Dict): Input data with activating examples.

    Returns:
        str: A regex pattern to match the selected tokens.
    """
    activating_examples = data.get("activating_examples", [])
    
    final_tokens = [example["tokens"][-1] for example in activating_examples if example["tokens"]]
    
    if not final_tokens:
        raise ValueError("No valid tokens found in activating examples.")

    token_patterns = [re.escape(str(token)) for token in final_tokens]
    regex_pattern = f"({'|'.join(token_patterns)})"

    print(f"Final Tokens: {final_tokens}")
    print(f"Selection Regex: {regex_pattern}")
    
    return regex_pattern