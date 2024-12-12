from regin.datatypes import load_jsonl, Examples, FeatureString
from regex_methods.regex_pipe import run_pipe_preloaded
from regin.eval.evaluation_methods import evaluate_sequence_level, evaluate_token_level, get_mistakes_token_level
from regin.eval.matchers import RegexMatcher
import time
import pandas as pd
import numpy as np
import logging

from collections import defaultdict, deque

from typing import List

train_path = "data/gpt-2-small/examples/train.jsonl"
eval_path = "data/gpt-2-small/examples/eval.jsonl"
val_path = "data/gpt-2-small/examples/val.jsonl"
eval_unpivoted_path = "data/gpt-2-small/eval.jsonl"
val_unpivoted_path = "data/gpt-2-small/val.jsonl"

logging.basicConfig(
        filename='my.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def run_proj():
    logging.info("-----------BEGIN----------")
    train_set = load_jsonl(train_path, Examples)
    eval_set_unpivoted = load_jsonl(eval_unpivoted_path, FeatureString)
    val_set_unpivoted = load_jsonl(val_unpivoted_path, FeatureString)

    # hyperparameter
    activation_threshold = 10
    prefix_length = 20

    regexes = {} # (feature_index, regex)
    f1_scores = []
    np.random.shuffle(train_set)

    false_pos = defaultdict(deque)

    iterations = 5
    for _ in range(iterations):
        for ex in train_set[:10]:
            feature_index = ex.feature_index
            regex = run_pipe_preloaded(data=ex.activating_examples, 
                                       negative_examples=false_pos[feature_index] if feature_index in false_pos else [])
            
            regexes[feature_index] = regex
            
            matcher = RegexMatcher(regex)
            seq_precision, seq_recall = evaluate_sequence_level(dataset=val_set_unpivoted, 
                                                                feature_index=feature_index, 
                                                                matcher=matcher, 
                                                                activation_threshold=activation_threshold)
            
            f1 = (2*seq_precision*seq_recall)/(seq_precision + seq_recall) if seq_precision > 0 and seq_recall > 0 else 0
            f1_scores.append((feature_index, seq_precision, seq_recall, f1, regex))

            with open("partial_output.txt", "a") as file:\
                file.write(f"Feature {feature_index}: Prec {seq_precision:.3f}, Rec {seq_recall:.3f}, F1 {f1:.3f}. Regex: {regex}\n")
            
            false_pos_samp, false_neg_samp = get_mistakes_token_level(dataset=val_set_unpivoted, 
                                                            feature_index=feature_index, 
                                                            matcher=matcher, 
                                                            prefix_length=prefix_length, 
                                                            activation_threshold=activation_threshold)
            
            logging.info(f"False positives: {false_pos_samp}")

            if false_pos_samp:
                for ex in false_pos_samp.activating_examples:
                    false_pos[feature_index].append(ex.text)
                    if len(false_pos[feature_index]) > 20:
                        false_pos[feature_index].popleft()
    
    df = pd.DataFrame(f1_scores, columns=["Feature Index", "Precision", "Recall", "F1 Score", "Regex"])

    df.to_csv('f1_scores.csv', index=False)

    logging.info("-----------END----------")

if __name__ == "__main__":
    run_proj()