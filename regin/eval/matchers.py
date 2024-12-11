import re
from abc import ABC, abstractmethod
from regex_methods.llms_exp_llms.inference import test_description as llm_test_description

class Matcher(ABC):
    """Evaluator class for evaluating regex patterns against a dataset."""
    @abstractmethod
    def get_matching_tokens(self, text: str, offsets: list[int]) -> set[int]:
        ...

    @abstractmethod
    def has_any_matching_tokens(self, text: str) -> bool:
        ...

class LLMDescriptionMatcher(Matcher):
    def __init__(self, description: str, system_prompt_path: str='regex_methods/llms_exp_llms/eval_prompt.txt', model: str='llama-3.3-70b-specdec'):
        self.description = description
        self.system_prompt_path = system_prompt_path
        self.model = model 
    def has_any_matching_tokens(self, text: str) -> bool:
        decision = llm_test_description(self.description, text)
        return decision
    def get_matching_tokens(self, text, offsets):
        raise NotImplementedError("LLMs explaining LLM's does not filter on specific tokens")
        
        
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


