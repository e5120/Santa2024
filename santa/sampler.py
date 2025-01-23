import random


class UniformSampler(object):
    def __init__(self, ops):
        self.ops = ops

    def sample(self):
        return random.choice(self.ops)


class OrderSampler(object):
    def __init__(self, ops):
        self.ops = ops
        self.n_ops = len(ops)
        self.index = 0

    def sample(self):
        op = self.ops[self.index]
        self.index = (self.index + 1) % self.n_ops
        return op
