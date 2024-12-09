import argparse
import logging

from .evaluate_regex import evaluate_regex, load_dataset


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

    f1 = (2*precision*recall)/(precision + recall) if precision > 0 and recall > 0 else 0
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

if __name__ == "__main__":
    main()