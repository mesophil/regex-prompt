from regin.datatypes import load_jsonl, FeatureString
from regex_methods.regex_pipe import run_pipe_preloaded
from regin.eval.evaluation_methods import evaluate_sequence_level, evaluate_token_level
from regin.eval.matchers import RegexMatcher

import pandas as pd

train_path = "/data/gpt-2-small/examples/train.jsonl"
eval_path = "/data/gpt-2-small/examples/eval.jsonl"
val_path = "/data/gpt-2-small/examples/val.jsonl"
eval_unpivoted_path = "/data/gpt-2-small/eval.jsonl"


def run_proj():
    train_set = load_jsonl(train_path, FeatureString)
    eval_set_unpivoted = load_jsonl(eval_set_unpivoted, FeatureString)

    # hyperparameter
    activation_threshold = 10

    regexes = {} # (feature_index, regex)

    for ex in train_set:
        feature_index = ex['feature_index']
        regex = run_pipe_preloaded(ex)
        regexes.append[feature_index] = regex

    f1_scores = []
    for ind in regexes:
        matcher = RegexMatcher(regexes[ind])
        seq_precision, seq_recall = evaluate_sequence_level(eval_set_unpivoted, ind, matcher, activation_threshold)
        f1 = (2*seq_precision*seq_recall)/(seq_precision + seq_recall) if seq_precision > 0 and seq_recall > 0 else 0
        f1_scores.append((ind, f1))

    
    df = pd.DataFrame(f1_scores, columns=["Feature Index", "F1 Score"])

    df.to_csv('f1_scores.csv', index=False)

if __name__ == "__main__":
    run_proj()