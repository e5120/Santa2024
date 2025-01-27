import random
from abc import ABCMeta, abstractmethod

import torch
import numpy as np


STOP_WORDS = [
    'we', 'that', 'as', 'it', 'with',
    'of', 'in', 'is', 'not', 'you',
    'from', 'and','to', 'the',
]


def get_probability(index, tokens, stopwords=STOP_WORDS, min_value=0.0):
    if isinstance(index, str):
        token = index
    else:
        token = tokens[index]
    if token in stopwords:
        p = np.array([min_value] * len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in stopwords:
                if tokens[i] == token:
                    p[i] = 0
                else:
                    p[i] = 1
        p /= p.sum()
    else:
        char = token[0]
        p = [min_value] * len(tokens)
        for i in range(len(tokens)):
            if token == tokens[i]:
                p[i] = 0
            elif char == tokens[i][0] and tokens[i] not in stopwords:
                p[i] = 1
        p = np.array(p, dtype=float)
        p_sum = p.sum()
        if p_sum > 0:
            p = p / p_sum
        else:
            p[:] = 1 / len(tokens)
    assert np.abs(1-np.sum(p)) < 1e-8 and len(p) == len(tokens), f"{p_sum}, {token}"
    return p


class Operator(metaclass=ABCMeta):
    def __call__(self, tokens):
        if isinstance(tokens, (np.ndarray, torch.Tensor)):
            tokens = tokens.tolist()
        return self.apply(tokens)

    @abstractmethod
    def apply(self, tokens):
        raise NotImplementedError


class OnePointShuffle(Operator):
    def apply(self, tokens):
        i = random.choice(np.arange(1, len(tokens)-1))
        tokens = tokens[i:] + tokens[:i]
        return tokens


class PairPointShuffle(Operator):
    def __init__(self, fix_ids=[], window_size=None, min_p=None):
        self.fix_ids = fix_ids
        self.window_size = window_size
        self.min_p = min_p
        if window_size is not None:
            assert min_p is None or min_p > 0

    def apply(self, tokens):
        remain_ids = list(filter(lambda x: x not in self.fix_ids, range(len(tokens))))
        i = np.random.choice(remain_ids)
        if self.min_p is None:
            p = None
        else:
            tmp_p = get_probability(i, tokens, min_value=self.min_p)
            p = np.array([tmp_p[index] for index in remain_ids])
            p /= p.sum()
        j = np.random.choice(remain_ids, p=p)
        while self.window_size and np.abs(i - j) > self.window_size:
            j = np.random.choice(remain_ids, p=p)
        tokens[i], tokens[j] = tokens[j], tokens[i]
        return tokens


class TokensShuffle(Operator):
    def __init__(self, min_tokens=1, max_tokens=3, fix_ids=[], min_p=None):
        assert min_tokens <= max_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.fix_ids = fix_ids
        self.min_p = min_p

    def apply(self, tokens):
        assert len(tokens) > self.max_tokens
        tokens = tokens.copy()
        remain_ids = list(filter(lambda x: x not in self.fix_ids, range(len(tokens))))
        while True:
            i = np.random.choice(remain_ids)
            if self.min_p is None:
                p = None
            else:
                tmp_p = get_probability(i, tokens, min_value=self.min_p)
                p = np.array([tmp_p[index] for index in remain_ids])
                p /= p.sum()
            j = np.random.choice(remain_ids, p=p)
            i_end = i + random.randint(self.min_tokens, self.max_tokens+1)
            j_end = j + random.randint(self.min_tokens, self.max_tokens+1)
            if i_end <= j:
                break
        tokens = tokens[:i] + tokens[j: j_end] + tokens[i_end: j] + tokens[i: i_end] + tokens[j_end:]
        fix_tokens = [tokens[i] for i in self.fix_ids]
        for fix_id, fix_token in zip(self.fix_ids, fix_tokens, strict=True):
            tokens.remove(fix_token)
            tokens = tokens[:fix_id] + [fix_token] + tokens[fix_id:]
        return tokens


class TokensReverse(Operator):
    def __init__(self, min_tokens=3, max_tokens=3, fix_ids=[]):
        assert min_tokens <= max_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.fix_ids = fix_ids

    def apply(self, tokens):
        assert len(tokens) > self.max_tokens
        is_valid = False
        while not is_valid:
            i = random.choice(range(len(tokens)-1))
            j = random.choice(range(self.min_tokens, self.max_tokens+1))
            k = min(i+j, len(tokens))
            is_valid = True
            for fix_id in self.fix_ids:
                if i <= fix_id < k:
                    is_valid = False
                    break
        tokens[i:k] = tokens[i:k][::-1]
        return tokens


class TokensInsert(Operator):
    def __init__(self, min_tokens=2, max_tokens=2, fix_ids=[], window_size=None, min_p=None):
        assert min_tokens <= max_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.fix_ids = fix_ids
        self.window_size = window_size
        self.min_p = min_p
        if window_size is not None:
            assert min_p is None or min_p > 0

    def apply(self, tokens):
        assert len(tokens) > self.max_tokens
        tokens = tokens.copy()
        remain_ids = list(filter(lambda x: x not in self.fix_ids, range(len(tokens))))
        i = np.random.choice(remain_ids)
        j = random.choice(range(self.min_tokens, self.max_tokens+1))
        k = min(i+j, len(tokens))
        sub_tokens = tokens[i:k]
        main_tokens = tokens[:i] + tokens[k:]
        if self.window_size is None:
            candidate_ids = range(len(main_tokens))
        else:
            mid_id = (i + k) // 2
            candidate_ids = range(max(0, mid_id-self.window_size), min(len(main_tokens), mid_id+self.window_size))
        if self.min_p is None:
            p = None
        else:
            tmp_p = get_probability(sub_tokens[0], main_tokens, min_value=self.min_p)
            p = np.array([tmp_p[index] for index in candidate_ids])
            p /= p.sum()
        l = np.random.choice(candidate_ids, p=p)
        tokens = main_tokens[:l] + sub_tokens + main_tokens[l:]
        fix_tokens = [tokens[i] for i in self.fix_ids]
        for fix_id, fix_token in zip(self.fix_ids, fix_tokens, strict=True):
            tokens.remove(fix_token)
            tokens = tokens[:fix_id] + [fix_token] + tokens[fix_id:]
        return tokens


class TokensRandomShuffle(Operator):
    def __init__(self, num_shuffles=10, fix_ids=[], window_size=None, min_p=None):
        self.num_shuffles = num_shuffles
        self.pps = PairPointShuffle(
            fix_ids=fix_ids,
            window_size=window_size,
            min_p=min_p,
        )

    def apply(self, tokens):
        num_shuffles = np.random.randint(2, self.num_shuffles+1)
        for _ in range(num_shuffles):
            tokens = self.pps(tokens)
        return tokens


class SkipInsert(Operator):
    def __init__(self, min_tokens=2, max_tokens=3, skip_size=2, start_id=0, fix_ids=[]):
        assert min_tokens <= max_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.skip_size = skip_size
        self.start_id = start_id

    def apply(self, tokens):
        assert len(tokens) > self.max_tokens
        tokens = tokens.copy()
        i = np.random.choice(range(self.start_id, len(tokens)))
        j = random.choice(range(self.min_tokens, self.max_tokens+1))
        k = min(i+j*self.skip_size, len(tokens)-1)
        sub_ids = range(i, k, self.skip_size)
        main_ids = list(filter(lambda x: x not in sub_ids, range(len(tokens))))
        sub_tokens = [tokens[l] for l in sub_ids]
        main_tokens = [tokens[l] for l in main_ids]
        candidate_ids = range(len(main_tokens))
        l = np.random.choice(candidate_ids, p=None)
        tokens = main_tokens[:l] + sub_tokens + main_tokens[l:]
        return tokens
