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
    def apply(self, tokens):
        i, j = random.sample(range(len(tokens)), k=2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
        return tokens


class TokensShuffle(Operator):
    def __init__(self, min_tokens=1, max_tokens=3):
        assert min_tokens < max_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def apply(self, tokens):
        assert len(tokens) > self.max_tokens
        while True:
            i, j = random.sample(range(len(tokens)), k=2)
            if i > j:
                i, j = j, i
            i_end = i + random.randint(self.min_tokens, self.max_tokens+1)
            j_end = j + random.randint(self.min_tokens, self.max_tokens+1)
            if i_end <= j:
                break
        tokens = tokens[:i] + tokens[j: j_end] + tokens[i_end: j] + tokens[i: i_end] + tokens[j_end:]
        return tokens


class TokensReverse(Operator):
    def __init__(self, min_tokens=2, max_tokens=3):
        assert min_tokens < max_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def apply(self, tokens):
        assert len(tokens) > self.max_tokens
        i = random.choice(range(len(tokens)-1))
        j = random.choice(range(self.min_tokens, self.max_tokens+1))
        k = min(i+j, len(tokens))
        tokens[i:k] = tokens[i:k][::-1]
        return tokens


class TokensInsert(Operator):
    def __init__(self, min_tokens=2, max_tokens=3):
        assert min_tokens < max_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def apply(self, tokens):
        assert len(tokens) > self.max_tokens
        i = random.choice(range(len(tokens)))
        j = random.choice(range(self.min_tokens, self.max_tokens+1))
        k = min(i+j, len(tokens))
        sub_tokens = tokens[i:k]
        main_tokens = tokens[:i] + tokens[k:]
        l = random.choice(range(len(main_tokens)))
        tokens = main_tokens[:l] + sub_tokens + main_tokens[l:]
        return tokens
