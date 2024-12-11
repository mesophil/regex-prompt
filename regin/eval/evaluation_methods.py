"""This script takes tests a regex against a feature dataset.

It takes in a path to feature dataset a feature index and a regex as command line arguments.

See the generate_feature_dataset/README.md for more information on the feature dataset.
The script returns the precision and recall of the regex against the feature dataset.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List

from tqdm import tqdm


logger = logging.getLogger(__name__)



def evaluate_token_level(
        examples: List[Dict],
        feature_index: int,
        evaluator: Matcher,
        activation_threshold: float = 0.0
    ) -> tuple[float, float]:
    """Evaluate regex pattern against the dataset at a token level.
    
    Returns:
        tuple[float, float]: (precision, recall)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_regex_matches = 0
    total_feature_matches = 0


    for example in tqdm(examples):
        text = example['text']
        
        # Get positions where regex matches
        regex_matches = evaluator.get_matching_tokens(text, example['offsets'])

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

def evaluate_sequence_level(
    examples: List[Dict],
    feature_index: int,
    evaluator: Matcher,
    activation_threshold: float = 0.0
):
    """Evaluate regex pattern against the dataset at a sequence level.
    
    Returns:
        tuple[float, float]: (precision, recall)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_regex_matches = 0
    total_feature_matches = 0


    for example in tqdm(examples):
        text = example['text']
        
        # Check if regex matches anywhere in the string
        regex_matches = evaluator.has_any_matching_tokens(text)

        # Get positions where feature is active
        feature_matches = any(
            activations[features.index(feature_index)] > activation_threshold
            for features, activations in zip(example['active_features'], example['activations'])
            if feature_index in features
        )
        
        # Calculate metrics
        if regex_matches:
            total_regex_matches += 1
        if feature_matches:
            total_feature_matches += feature_matches
        if regex_matches and feature_matches:
            true_positives += 1
        if regex_matches and not feature_matches:
            false_positives += 1
        if not regex_matches and feature_matches:
            false_negatives += 1

    logger.info(f"Total regex matches: {total_regex_matches}")
    logger.info(f"Total feature matches: {total_feature_matches}")

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall