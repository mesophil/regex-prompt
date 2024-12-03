"""This script takes tests a regex against a feature dataset.

It takes in a path to feature dataset a feature index and a regex as command line arguments.

See the generate_feature_dataset/README.md for more information on the feature dataset.
The script returns the precision and recall of the regex against the feature dataset.
"""

import json
import re
import argparse
from typing import List, Dict

from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load examples from the JSONL dataset."""
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def evaluate_regex(examples: List[Dict], feature_index: int, regex_pattern: str, activation_threshold: float = 0.0) -> tuple[float, float]:
    """
    Evaluate regex pattern against the dataset.
    
    Returns:
        tuple[float, float]: (precision, recall)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_regex_matches = 0
    total_feature_matches = 0

    pattern = re.compile(regex_pattern)
    
    for example in tqdm(examples):
        text = example['text']
        
        # Get positions where regex matches
        regex_matches = set()
        for match in pattern.finditer(text):
            end = match.end()
            token_pos = None
            for i in range(len(example['offsets'])):
                if end <= example['offsets'][i]:
                    token_pos = i - 1
                    break
            regex_matches.add(token_pos)

        # Get positions where feature is active
        feature_matches = set()
        for pos, (features, activations) in enumerate(zip(example['active_features'], example['activations'])):
            if feature_index in features:
                index = features.index(feature_index)
                if activations[index] > activation_threshold:
                    feature_matches.add(pos)
        
        # Calculate metrics
        total_regex_matches += len(regex_matches)
        total_feature_matches += len(feature_matches)
        true_positives += len(regex_matches & feature_matches)
        false_positives += len(regex_matches - feature_matches)
        false_negatives += len(feature_matches - regex_matches)


    logger.info(f"Total regex matches: {total_regex_matches}")
    logger.info(f"Total feature matches: {total_feature_matches}")

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

def main():
    parser = argparse.ArgumentParser(description='Evaluate regex against feature dataset')
    parser.add_argument('dataset_path', help='Path to feature dataset JSONL file')
    parser.add_argument('feature_index', type=int, help='Feature index to evaluate')
    parser.add_argument('regex', help='Regular expression pattern')
    parser.add_argument('--activation_threshold', type=float, default=0.0, help='Activation threshold')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    examples = load_dataset(args.dataset_path)
    
    # Evaluate regex
    precision, recall = evaluate_regex(examples, args.feature_index, args.regex, activation_threshold=args.activation_threshold)
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

if __name__ == "__main__":
    main()