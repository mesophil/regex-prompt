"""Convert feature dataset into examples of neuron activations."""
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from ..datatypes import Examples, FeatureString, TokenizedString, save_jsonl


def read_feature_strings(file_path: Path) -> List[FeatureString]:
    """Read in a JSONL file of feature strings."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return [FeatureString.model_validate_json(line) for line in lines]


def pivot(dataset: List[FeatureString], prefix_length: int, activation_threshold: float = 0.0) -> list[Examples]:
    """Pivot the dataset to be indexed by feature."""
    feature_examples: Dict[int, Examples] = dict()

    for row in tqdm(dataset):
        tokenized_string = TokenizedString(
            text=row.text,
            tokens=row.tokens,
            offsets=row.offsets
        )
        for pos, (features, activations) in enumerate(zip(row.active_features, row.activations)):
            if pos < prefix_length:
                continue

            for feature, activation in zip(features, activations):
                if activation < activation_threshold:
                    continue

                if feature not in feature_examples:
                    feature_examples[feature] = Examples(
                        feature_index=feature,
                        activating_examples=[]
                    )

                prefix = tokenized_string.slice(max(0, pos - prefix_length), pos + 1)

                feature_examples[feature].activating_examples.append(
                    prefix
                )

    return list(feature_examples.values())


def main(args):
    dataset = read_feature_strings(Path(args.feature_dataset))
    pivoted = pivot(dataset, args.prefix_length, activation_threshold=args.act_thresh)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(Path(args.save_path), pivoted)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pivot feature dataset')
    parser.add_argument('feature_dataset', help='Path to feature dataset JSONL file', type=Path)
    parser.add_argument('save_path', help='Path to output JSONL file', type=Path)
    parser.add_argument('--prefix_length', type=int, default=20, help='Number of tokens to include in prefix')
    parser.add_argument('--act_thresh', type=float, default=0.0, help='Threshold for activation')

    args = parser.parse_args()
    main(args)
