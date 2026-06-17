"""Answer normalization and evaluation metrics (QA-generic)."""

import re
import string
from collections import Counter
from typing import Tuple


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    np_ = normalize_answer(prediction)
    ng_ = normalize_answer(ground_truth)
    ZERO = (0.0, 0.0, 0.0)
    if np_ in ['yes', 'no', 'noanswer'] and np_ != ng_:
        return ZERO
    if ng_ in ['yes', 'no', 'noanswer'] and np_ != ng_:
        return ZERO
    pt = np_.split()
    gt = ng_.split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO
    precision = 1.0 * num_same / len(pt)
    recall = 1.0 * num_same / len(gt)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def em(prediction: str, ground_truth: str) -> int:
    np_ = normalize_answer(prediction)
    ng_ = normalize_answer(ground_truth)
    if ng_ in ['yes', 'no'] and ng_ in np_[:3]:
        return 1
    return 1 if ng_ == np_ else 0
