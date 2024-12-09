from pathlib import Path
from typing import List, Type, TypeVar

from pydantic import BaseModel, NonNegativeInt


class TokenizedString(BaseModel):
    text: str
    tokens: List[NonNegativeInt]
    offsets: List[NonNegativeInt]


    def model_post_init(self, __context) -> None:
        assert len(self.tokens) == len(self.offsets), f"{len(self.tokens)} != {len(self.offsets)}"

    def slice(self, start: int, end: int) -> 'TokenizedString':
        """Slice the tokenized string by token index."""
        # check index bounds
        assert len(self.tokens) == len(self.offsets), f"{len(self.tokens)} != {len(self.offsets)}"
        if start < 0 or end > len(self.tokens) or start > end:
            raise ValueError(
                f"Invalid slice indices: {start}, {end}"
                f"for tokens of length {len(self.tokens)}"
            )

        if end == len(self.tokens):
            sliced_text = self.text[self.offsets[start]:]
        else:
            sliced_text = self.text[self.offsets[start]:self.offsets[end]]

        sliced_tokens = self.tokens[start:end]
        sliced_offsets = self.offsets[start:end]
        sliced_offsets = [offset - sliced_offsets[0] for offset in sliced_offsets]
        return TokenizedString(
            text=sliced_text,
            tokens=sliced_tokens,
            offsets=sliced_offsets
        )

    def get_str_tokens(self) -> List[str]:
        tokens = []
        for i in range(len(self.offsets)):
            start = self.offsets[i]
            if i + 1 < len(self.offsets):
                stop = self.offsets[i + 1]
            else:
                stop = len(self.text)
            tokens.append(self.text[start:stop])
        return tokens


class Examples(BaseModel):
    feature_index: NonNegativeInt
    """The index of the feature"""
    activating_examples: List[TokenizedString]
    """A string that causes the feature to activate on the final token."""

class FeatureString(TokenizedString):
    active_features: List[List[NonNegativeInt]]
    activations: List[List[float]]

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        assert len(self.active_features) == len(self.activations)
        assert len(self.active_features) == len(self.tokens)

T = TypeVar('T', bound=BaseModel)

def save_jsonl(file_path: Path, items: List[T]) -> None:
    """Save a list of items to a JSONL file."""
    with open(file_path, 'w') as f:
        for item in items:
            f.write(item.model_dump_json() + '\n')

def load_jsonl(file_path: Path, model: Type[T]) -> List[T]:
    """Load a list of items from a JSONL file."""
    items = []
    with open(file_path, 'r') as f:
        for line in f:
            items.append(model.model_validate_json(line))
    return items
