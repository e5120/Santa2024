import random


class UniformSampler(object):
    def __init__(self, ops):
        self.ops = ops

    def sample(self):
        return random.choice(self.ops)
