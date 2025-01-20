import random
from abc import ABCMeta, abstractmethod

import torch
import numpy as np


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
        i = random.choice(np.arange(1, len(tokens)))
        tokens = tokens[i:] + tokens[:i]
        return tokens


class PairPointShuffle(Operator):
    def __init__(self, fix_ids=[]):
        self.fix_ids = fix_ids

    def apply(self, tokens):
        remain_ids = list(filter(lambda x: x not in self.fix_ids, range(len(tokens))))
        i, j = random.sample(remain_ids, k=2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
        return tokens


class TokensShuffle(Operator):
    def __init__(self, min_tokens=1, max_tokens=3, fix_ids=[]):
        assert min_tokens <= max_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.fix_ids = fix_ids

    def apply(self, tokens):
        assert len(tokens) > self.max_tokens
        tokens = tokens.copy()
        fix_tokens = [tokens[i] for i in self.fix_ids]
        while True:
            i, j = sorted(random.sample(range(len(tokens)), k=2))
            i_end = i + random.randint(self.min_tokens, self.max_tokens+1)
            j_end = j + random.randint(self.min_tokens, self.max_tokens+1)
            if i_end <= j:
                break
        tokens = tokens[:i] + tokens[j: j_end] + tokens[i_end: j] + tokens[i: i_end] + tokens[j_end:]
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
        tokens[i:k] = tokens[i:k][::-1]
        return tokens


class TokensInsert(Operator):
    def __init__(self, min_tokens=2, max_tokens=2, fix_ids=[]):
        assert min_tokens <= max_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.fix_ids = fix_ids

    def apply(self, tokens):
        assert len(tokens) > self.max_tokens
        tokens = tokens.copy()
        fix_tokens = [tokens[i] for i in self.fix_ids]
        i = random.choice(range(len(tokens)-1))
        j = random.choice(range(self.min_tokens, self.max_tokens+1))
        k = min(i+j, len(tokens))
        sub_tokens = tokens[i:k]
        main_tokens = tokens[:i] + tokens[k:]
        l = random.choice(range(len(main_tokens)))
        tokens = main_tokens[:l] + sub_tokens + main_tokens[l:]
        for fix_id, fix_token in zip(self.fix_ids, fix_tokens, strict=True):
            tokens.remove(fix_token)
            tokens = tokens[:fix_id] + [fix_token] + tokens[fix_id:]
        return tokens


class TokensRandomShuffle(Operator):
    def __init__(self, num_shuffles=10, fix_ids=[]):
        self.num_shuffles = num_shuffles
        self.pps = PairPointShuffle(fix_ids=fix_ids)

    def apply(self, tokens):
        num_shuffles = np.random.randint(2, self.num_shuffles+1)
        for _ in range(num_shuffles):
            tokens = self.pps(tokens)
        return tokens
