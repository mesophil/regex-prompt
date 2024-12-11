


class Matcher(ABC):
    """Evaluator class for evaluating regex patterns against a dataset."""
    @abstractmethod
    def get_matching_tokens(self, text: str, offsets: list[int]) -> set[int]:
        ...

    @abstractmethod
    def has_any_matching_tokens(self, text: str) -> bool:
        ...

class RegexMatcher(Matcher):
    """Evaluator class for evaluating regex patterns against a dataset."""
    def __init__(self, regex_pattern: str):
        self.pattern = re.compile(regex_pattern)

    def get_matching_tokens(self, text: str, offsets: list[int]) -> set[int]:
        """Get positions where regex matches."""
        matches = set()
        for match in self.pattern.finditer(text):
            end = match.end()
            token_pos = None
            for i in range(len(offsets)):
                if end <= offsets[i]:
                    token_pos = i - 1
                    break
            matches.add(token_pos)
        return matches

    def has_any_matching_tokens(self, text: str) -> bool:
        """Check if regex matches anywhere in the string."""
        return bool(self.pattern.search(text))


