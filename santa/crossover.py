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
        s, e = sorted(np.random.choice(range(n), size=2, replace=False))
        c1 = self.oc_mapping(p1, p2, s, e)
        c2 = self.oc_mapping(p2, p1, s, e)
        return [c1, c2]

    def oc_mapping(self, p1, p2, s, e):
        p1 = p1.copy()
        p2 = p2.copy()
        sub_tokens = p1[s: e]
        for token in sub_tokens:
            p2.remove(token)
        child = p2[:s] + sub_tokens + p2[s:]
        return child


class ImprOrderCrossover(BaseCrossover):
    def apply(self, p1, p2):
        n = len(p1)
        size = np.random.randint(1, n-1)
        swap_ids = sorted(np.random.choice(range(n), size=size, replace=False))
        c1 = self.ioc_mapping(p1, p2, swap_ids)
        c2 = self.ioc_mapping(p2, p1, swap_ids)
        return [c1, c2]

    def ioc_mapping(self, p1, p2, ids):
        n = len(p1)
        c = [-1] * n
        p1 = p1.copy()
        for i in ids:
            c[i] = p2[i]
            p1.remove(p2[i])
        remain_ids = list(filter(lambda x: x not in ids, range(n)))
        assert len(p1) == len(remain_ids)
        for token, i in zip(p1, remain_ids):
            c[i] = token
        return c


class PartiallyMappedCrossover(BaseCrossover):
    def apply(self, p1, p2):
        s, e = sorted(np.random.choice(range(len(p1)), size=2, replace=False))
        c1 = self.pmx_mapping(p1, p2, s, e)
        c2 = self.pmx_mapping(p2, p1, s, e)
        return [c1, c2]

    def pmx_mapping(self, c, p, s, e):
        c = c.copy()
        p = p.copy()
        n = len(c)
        sub_order = p[s: e]
        c[s: e] = sub_order
        dup_index = []
        remain_order = p[:s] + p[e:]
        for i in range(s):
            if c[i] in sub_order:
                dup_index.append(i)
            else:
                remain_order.remove(c[i])
        for i in range(e, n):
            if c[i] in sub_order:
                dup_index.append(i)
            else:
                remain_order.remove(c[i])
        assert len(dup_index) == len(remain_order)
        for i, o in zip(dup_index, remain_order):
            c[i] = o
        return c


class CyclicCrossover(BaseCrossover):
    def __init__(self, start_id=0):
        self.s = start_id

    def apply(self, p1, p2):
        s = np.random.randint(len(p1)) if self.s is None else self.s
        c1 = self.cx_mapping(p1, p2, s)
        c2 = self.cx_mapping(p2, p1, s)
        return [c1, c2]

    def cx_mapping(self, p1, p2, s):
        n = len(p1)
        c = [-1] * n
        remain_ids = list(range(n))
        p = p2.copy()
        while c[s] == -1:
            remain_ids.remove(s)
            p.remove(p1[s])
            c[s] = p1[s]
            s = p2[s]
        for i, token in zip(remain_ids, p):
            c[i] = token
        return c
