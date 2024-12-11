import argparse
import logging

from regin.datatypes import FeatureString, load_jsonl
from regin.eval.evaluation_methods import evaluate_sequence_level, evaluate_token_level
from regin.eval.matchers import RegexMatcher


def main():
    logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    examples = load_jsonl(args.dataset_path, FeatureString)

    matcher = RegexMatcher(args.regex)

    # Evaluate regex
    if args.sequence_level:
        print("Evaluating at sequence level")
        precision, recall = evaluate_sequence_level(examples, args.feature_index, matcher, activation_threshold=args.activation_threshold)
    else:
        print("Evaluating at token level")
        precision, recall = evaluate_token_level(examples, args.feature_index, matcher, activation_threshold=args.activation_threshold)

    f1 = (2*precision*recall)/(precision + recall) if precision > 0 and recall > 0 else 0
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate regex against feature dataset')
    parser.add_argument('dataset_path', help='Path to feature dataset JSONL file')
    parser.add_argument('feature_index', type=int, help='Feature index to evaluate')
    parser.add_argument('regex', help='Regular expression pattern')
    parser.add_argument('--sequence_level', action='store_true', help='Evaluate at sequence level')
    parser.add_argument('--activation_threshold', type=float, default=0.0, help='Activation threshold')

    args = parser.parse_args()
    main()
