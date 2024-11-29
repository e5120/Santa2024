import random
from abc import ABCMeta, abstractmethod

import numpy as np


class Operator(metaclass=ABCMeta):
    def __call__(self, tokens):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return self.apply(tokens)

    @abstractmethod
    def apply(self, tokens):
        raise NotImplementedError


class PairPointShuffle(Operator):
    def apply(self, tokens):
        i, j = random.sample(range(len(tokens)), k=2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
        return tokens


class OnePointShuffle(Operator):
    def apply(self, tokens):
        i = random.choice(np.arange(1, len(tokens)))
        tokens = tokens[i:] + tokens[:i]
        return tokens


class TokensShuffle(Operator):
    def __init__(self, max_tokens=3):
        self.max_tokens = max_tokens

    def apply(self, tokens):
        assert len(tokens) > self.max_tokens
        while True:
            i, j = random.sample(range(len(tokens)), k=2)
            if i > j:
                i, j = j, i
            i_end = i + random.randint(1, self.max_tokens)
            j_end = j + random.randint(1, self.max_tokens)
            if i_end <= j:
                break
        tokens = tokens[:i] + tokens[j: j_end] + tokens[i_end: j] + tokens[i: i_end] + tokens[j_end:]
        return tokens
