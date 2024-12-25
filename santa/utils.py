import os
import datetime
import itertools
import math
import pickle
import tempfile
from pathlib import Path

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


def save_text(text, score, target_id, output_dir="./output"):
    f, i = math.modf(score)
    i = int(i)
    f = int(f * 100000)
    prefix_filename = f"id{target_id}_{i:0>4}.{f:0>5}"
    same_files = list(Path(output_dir).glob(f"{prefix_filename}*.txt"))
    has_same_text = False
    for filename in same_files:
        with open(filename) as f:
            tmp = f.read().strip()
            print(tmp)
            if text == tmp:
                has_same_text = True
                break
    if not has_same_text:
        with open(Path(output_dir, f"{prefix_filename}_{len(same_files)}.txt"), "w") as f:
            f.write(text)


def get_log_path(target_id, root_dir="./logs"):
    gpu_id = os.environ["CUDA_VISIBLE_DEVICES"].replace(",", "-") if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu"
    return Path(root_dir, f"id{target_id}_logs_{gpu_id}.pkl").resolve()


def load_logs(target_id, root_dir="./logs"):
    file_path = get_log_path(target_id, root_dir=root_dir)
    print(file_path)
    if file_path.is_file():
        return pickle.load(open(file_path, "rb"))
    else:
        return {}


def save_logs(logs, target_id, root_dir="./logs"):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=root_dir) as f:
        pickle.dump(logs, f)
        f.flush()
        os.fsync(f.fileno())
        tmp_file_path = f.name
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    file_path = get_log_path(target_id, root_dir=root_dir)
    os.replace(tmp_file_path, file_path)


def save_history(text_history, score_history, target_id, root_dir="./logs"):
    today = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(root_dir, f"id{target_id}_{today}.pkl")
    logs = {
        "text": text_history,
        "score": score_history,
    }
    pickle.dump(logs, open(file_path, "wb"))
