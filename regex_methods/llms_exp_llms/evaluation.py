from regin.eval.matchers import LLMDescriptionMatcher
from regin.eval.evaluation_methods import evaluate_sequence_level, calculate_metrics
from regin.datatypes import FeatureString, load_jsonl, Examples
from regex_methods.llms_exp_llms.inference import make_prompt_from_tokens, get_description_from_prompt
import json 
import logging
from typing import Optional

logging.basicConfig(
    filename='my.log',
    level=logging.DEBUG,
    format='%(asctime)s-%(levelname)s-%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_feature_examples(dataset_path: str, feature_index: int):
    
    examples = load_jsonl(dataset_path, Examples)
    feature = None
    for ex in examples:
        if ex.feature_index == feature_index:
            feature = ex.feature
            break 
        
    return feature

def get_activating_tokens_from_example(example: Examples):
    
    positive_examples = []
    for activating_example in example.activating_examples:
        last_token_of_example = activating_example.get_str_tokens()[-1]
        positive_examples.append(last_token_of_example)
    return positive_examples

def generate_description_for_feature(positive_example: Examples, negative_example: Optional[Examples] = None, *args, **kwargs):
    
    positive_tokens = get_activating_tokens_from_example(positive_example)
    if negative_example is not None:
        negative_tokens = get_activating_tokens_from_example(negative_example)
    else:
        negative_tokens = []

    prompt = make_prompt_from_tokens(negative_tokens, positive_tokens)
    
    description = get_description_from_prompt(prompt, *args, **kwargs)
    
    return description