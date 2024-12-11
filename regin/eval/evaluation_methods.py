"""This script takes tests a regex against a feature dataset.

It takes in a path to feature dataset a feature index and a regex as command line arguments.

See the generate_feature_dataset/README.md for more information on the feature dataset.
The script returns the precision and recall of the regex against the feature dataset.
"""

import logging
from typing import List

from tqdm import tqdm

from regin.datatypes import Examples, FeatureString
from regin.eval.matchers import Matcher

logger = logging.getLogger(__name__)

def calculate_metrics(true_positives: int, false_positives: int, false_negatives: int) -> tuple[float, float]:
    """Calculate precision and recall from true positives, false positives and false negatives."""
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall


def get_token_level_feature_matches(feature_string: FeatureString, feature_index: int, activation_threshold: float) -> set[int]:
    """Get positions where feature is active at a token level."""
    feature_matches = set()
    for pos, (features, activations) in enumerate(zip(feature_string.active_features, feature_string.activations)):
        if feature_index in features:
            index = features.index(feature_index)
            if activations[index] > activation_threshold:
                feature_matches.add(pos)
    return feature_matches

def evaluate_token_level(
        dataset: List[FeatureString],
        feature_index: int,
        matcher: Matcher,
        activation_threshold: float = 0.0
    ) -> tuple[float, float]:
    """Evaluate regex pattern against the dataset at a token level.
    
    Args:
        dataset (List[FeatureString]): List of examples
        feature_index (int): Index of the feature
        matcher (Matcher): Matcher object
        activation_threshold (float, optional): Activation threshold. Defaults to 0.0.

    Returns:
        tuple[float, float]: (precision, recall)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_regex_matches = 0
    total_feature_matches = 0


    for example in tqdm(dataset):
        # Get positions where regex matches
        regex_matches = matcher.get_matching_tokens(
            example.text,
            example.offsets
        )

        # Get positions where feature is active
        feature_matches = get_token_level_feature_matches(
            example,
            feature_index,
            activation_threshold
        )

        # Calculate metrics
        total_regex_matches += len(regex_matches)
        total_feature_matches += len(feature_matches)
        true_positives += len(regex_matches & feature_matches)
        false_positives += len(regex_matches - feature_matches)
        false_negatives += len(feature_matches - regex_matches)

    logger.info(f"Total regex matches: {total_regex_matches}")
    logger.info(f"Total feature matches: {total_feature_matches}")

    return calculate_metrics(true_positives, false_positives, false_negatives)

def evaluate_sequence_level(
    dataset: List[FeatureString],
    feature_index: int,
    matcher: Matcher,
    activation_threshold: float = 0.0
):
    """Evaluate regex pattern against the dataset at a sequence level.
    
    Args:
        dataset (List[FeatureString]): List of examples
        feature_index (int): Index of the feature
        matcher (Matcher): Matcher object
        activation_threshold (float, optional): Activation threshold. Defaults to 0.0.
    
    Returns:
        tuple[float, float]: (precision, recall)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_regex_matches = 0
    total_feature_matches = 0


    for example in tqdm(dataset):
        # Check if regex matches anywhere in the string
        regex_matches = matcher.has_any_matching_tokens(example.text)

        # Get positions where feature is active
        feature_matches = bool(
            get_token_level_feature_matches(
                example,
                feature_index,
                activation_threshold
            )
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

    return calculate_metrics(true_positives, false_positives, false_negatives)


def get_mistakes_token_level(
        dataset: List[FeatureString],
        feature_index: int,
        matcher: Matcher,
        prefix_length: int = 20,
        activation_threshold: float = 0.0
    ) -> tuple[Examples, Examples]:
    """Get examples where the regex and feature don't match at a token level.
    
    Args:
        dataset (List[FeatureString]): List of examples
        feature_index (int): Index of the feature
        matcher (Matcher): Matcher object
        prefix_length (int, optional): Prefix length. Defaults to 20.
        activation_threshold (float, optional): Activation threshold. Defaults to 0.0.
    
    Returns:
        tuple[Examples, Examples]: (false_positives, false_negatives)
    """
    false_positive = Examples(
        feature_index=feature_index,
        activating_examples=[]
    )
    false_negatives = Examples(
        feature_index=feature_index,
        activating_examples=[]
    )

    for example in tqdm(dataset):
        # Get positions where regex matches
        regex_match_inds = matcher.get_matching_tokens(
            example.text,
            example.offsets
        )

        # Get positions where feature is active
        feature_matches_inds = get_token_level_feature_matches(
            example,
            feature_index,
            activation_threshold
        )

        false_positive_inds = regex_match_inds - feature_matches_inds
        false_negative_inds = feature_matches_inds - regex_match_inds
        
        for index in false_positive_inds:
            false_positive.activating_examples.append(
                example.slice(max(index - prefix_length, 0), index + 1)
            )
        
        for index in false_negative_inds:
            false_negatives.activating_examples.append(
                example.slice(max(index - prefix_length, 0), index + 1)
            )

    return false_positive, false_negatives
