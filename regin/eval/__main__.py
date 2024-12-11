import argparse
import logging
import json
from pathlib import Path

from regin.datatypes import FeatureString, load_jsonl
from regin.eval.evaluation_methods import evaluate_sequence_level, evaluate_token_level, get_mistakes_token_level
from regin.eval.matchers import RegexMatcher


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    dataset = load_jsonl(args.dataset_path, FeatureString)

    matcher = RegexMatcher(args.regex)

    # Evaluate regex
    if args.sequence_level:
        print(f"Evaluating at sequence level on {len(dataset)} examples")
        precision, recall = evaluate_sequence_level(
            dataset, 
            args.feature_index,
            matcher,
            activation_threshold=args.activation_threshold
        )
    else:
        print(f"Evaluating at token level on {len(dataset)} examples")
        precision, recall = evaluate_token_level(
            dataset,
            args.feature_index,
            matcher,
            activation_threshold=args.activation_threshold
        )


    if args.save_mistakes:
        if args.sequence_level:
            raise NotImplementedError("Saving mistakes not implemented for sequence level evaluation")
        else:
            false_positives, false_negatives = get_mistakes_token_level(
                dataset,
                args.feature_index,
                matcher,
                activation_threshold=args.activation_threshold,
                prefix_length=args.prefix_length,
            )


    f1 = (2*precision*recall)/(precision + recall) if precision > 0 and recall > 0 else 0

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        # Save feature_index, regex, precision, recall, f1 as json
        save_path = args.save_dir / "evaluation.csv"
        with open(save_path, 'w') as f:
            json.dump({
                "feature_index": args.feature_index,
                "dataset_path": args.dataset_path,
                "regex": args.regex,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }, f, indent=4)
        print(f"Saved evaluation results to {save_path}")
        # if present save mistakes
        if args.save_mistakes:
            false_negatives_path = args.save_dir / "false_negatives.jsonl"
            false_positives_path = args.save_dir / "false_positives.jsonl"
            with open(false_negatives_path, 'w') as f:
                f.write(false_negatives.model_dump_json())
            
            with open(false_positives_path, 'w') as f:
                f.write(false_positives.model_dump_json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate regex against feature dataset')
    parser.add_argument('dataset_path', help='Path to feature dataset JSONL file')
    parser.add_argument('feature_index', type=int, help='Feature index to evaluate')
    parser.add_argument('regex', help='Regular expression pattern')
    parser.add_argument('--sequence_level', action='store_true', help='Evaluate at sequence level')
    parser.add_argument('--activation_threshold', type=float, default=0.0, help='Activation threshold')
    parser.add_argument('--save_mistakes', action='store_true', help='Save mistakes')
    parser.add_argument('--save_dir', type=Path, help='Path to directory to save results', default=None)
    parser.add_argument('--prefix_length', type=int, default=20, help='Number of tokens to include in prefix for mistakes')

    args = parser.parse_args()
    main(args)
