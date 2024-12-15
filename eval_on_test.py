from regin.datatypes import load_jsonl, Examples, FeatureString
from regex_methods.regex_pipe import run_pipe_multi
from regin.eval.evaluation_methods import evaluate_sequence_level, evaluate_token_level, get_mistakes_token_level
from regin.eval.matchers import RegexMatcher
import time
import pandas as pd
import numpy as np
import logging
import re
import csv

from config import activation_threshold

from regex_methods.simple_or import simple_or_regex

from collections import defaultdict, deque

from typing import List

eval_unpivoted_path = "data/gpt-2-small/eval_5000.jsonl"

logging.basicConfig(
        filename='my.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

np.random.seed(123)

def test(FILENAME):
    logging.info("-----------BEGIN TEST----------")
    eval_set_unpivoted = load_jsonl(eval_unpivoted_path, FeatureString)

    with open(FILENAME, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    results = []
    regexes = {}

    for row in data:
        feature_index = int(row['Feature Index'])
        f1_score = float(row['F1 Score'])
        regex = row['Regex']
        if feature_index not in regexes or f1_score > regexes[feature_index]['f1_score']:
            regexes[feature_index] = {'regex': regex, 'f1_score': f1_score}

    for feature_index in regexes:
        regex = regexes[feature_index]['regex']
        matcher = RegexMatcher(regex)
        seq_precision, seq_recall = evaluate_sequence_level(dataset=eval_set_unpivoted, 
                                                            feature_index=feature_index, 
                                                            matcher=matcher, 
                                                            activation_threshold=activation_threshold)
                
        f1 = (2*seq_precision*seq_recall)/(seq_precision + seq_recall) if seq_precision > 0 and seq_recall > 0 else 0
        results.append([feature_index, f1])

    logging.info("-----------END TEST----------")

    df = pd.DataFrame(results, columns=["Feature Index", "F1 Score"])
    df.to_csv('results/regin-5-neg_test_f1_scores.csv', index=False)

if __name__ == "__main__":
    FILENAME = 'results/regin-5-neg_f1_scores.csv'
    test(FILENAME=FILENAME)