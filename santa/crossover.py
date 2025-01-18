import random
from abc import ABCMeta, abstractmethod

import torch
import numpy as np


class BaseCrossover(metaclass=ABCMeta):
    def __call__(self, p1, p2):
        if isinstance(p1, (np.ndarray, torch.Tensor)):
            p1 = p1.tolist()
        if isinstance(p2, (np.ndarray, torch.Tensor)):
            p2 = p2.tolist()
        return self.apply(p1, p2)

    @abstractmethod
    def apply(self, p1, p2):
        raise NotImplementedError


class OrderCrossover(BaseCrossover):
    def apply(self, p1, p2):
        n = len(p1)
        start, end = sorted(random.sample(range(n), 2))
        sub_tokens = p1[start: end+1]
        c1 = self.ox_mapping(p1, p2, start, end, n)
        c2 = self.ox_mapping(p2, p1, start, end, n)
        return [c1, c2]

    def ox_mapping(self, p1, p2, start, end, n):
        sub_tokens = p1[start: end+1]
        p2 = p2.copy()
        for token in sub_tokens:
            p2.remove(token)
        c = p2[:start] + sub_tokens + p2[start:]
        return c


class PartiallyMappedCrossover(BaseCrossover):
    def apply(self, p1, p2):
        x, y = np.random.choice(range(len(p1)), size=2, replace=False)
        x, y = (y, x) if x > y else (x, y)
        c1 = self.pmx_mapping(p1, p2, x, y)
        c2 = self.pmx_mapping(p2, p1, x, y)
        return [c1, c2]

    def pmx_mapping(self, c, p, x, y):
        # print(c)
        c = c.copy()
        p = p.copy()
        n = len(c)
        sub_order = p[x: y]
        c[x: y] = sub_order
        dup_index = []
        remain_order = p[:x] + p[y:]
        for i in range(x):
            if c[i] in sub_order:
                c[i] = -1
                dup_index.append(i)
            else:
                remain_order.remove(c[i])
        for i in range(y, n):
            if c[i] in sub_order:
                c[i] = -1
                dup_index.append(i)
            else:
                remain_order.remove(c[i])
        assert len(dup_index) == len(remain_order)
        for i, o in zip(dup_index, remain_order):
            c[i] = o
        return c
