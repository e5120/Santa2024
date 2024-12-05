import itertools

import numpy as np
# import hydra
import lightning as L


def setup(cfg):
#     cfg.dir.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    L.seed_everything(cfg["seed"])


def sub_permutations(tokens, fixed_ids=[]):
    tokens = np.array(tokens)
    fixed_ids = np.array(sorted(fixed_ids))
    mutable_tokens = np.array([
        tokens[i] for i in range(len(tokens)) if i not in fixed_ids
    ])
    assert len(mutable_tokens) < 11
    perms = list(map(list, itertools.permutations(mutable_tokens)))
    for perm in perms:
        for i in fixed_ids:
            perm.insert(i, tokens[i])
    return perms
