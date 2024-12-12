import math
from pathlib import Path

import hydra
import pandas as pd

import santa.operator
import santa.sampler
from santa.sa import simulated_annealing
from santa.metrics import PerplexityCalculator
from santa.utils import setup


TOKEN = "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"


@hydra.main(config_path="conf", config_name="optimize", version_base=None)
def main(cfg):
    # setup(cfg)
    df = pd.read_csv(Path(cfg.dir.data_dir, "sample_submission.csv"))
    ops = []
    for op in cfg.operators:
        ops.append(
            getattr(santa.operator, op.operator)(**op.operator_kwargs)
        )
    sampler = getattr(santa.sampler, cfg.sampler)(ops, **cfg.sampler_kwargs)
    best_scores = []
    if cfg.initial_solution is None:
        best_text = df.loc[cfg.target_id, "text"]
    else:
        cfg.target_id = int(Path(cfg.initial_solution).name[2])
        with open(cfg.initial_solution) as f:
            best_text = f.readline()
    precomputed = dict(cfg.precomputed)
    scorer = PerplexityCalculator(
        model_path=cfg.model_path,
        load_in_8bit=cfg.load_in_8bit,
        device_map=cfg.device_map,
    )
    for _ in range(cfg.num_cycles):
        best_text, best_score, precomputed = simulated_annealing(
            best_text, sampler, scorer,
            temp_start=cfg.temp_start,
            temp_end=cfg.temp_end,
            cooling_rate=cfg.cooling_rate,
            steps_per_temp=cfg.steps_per_temp,
            alpha=cfg.alpha,
            precomputed=precomputed,
            verbose=cfg.verbose,
            logging_step=cfg.logging_step,
        )
        print(f"\nbest score: {best_score:.5f}, # of search: {len(precomputed)}, best order: {best_text}")
        f, i = math.modf(best_score)
        with open(f"{cfg.dir.output_dir}/id{cfg.target_id}_{int(i):0>4}.{int(f*100000)}.txt", "w") as f:
            f.write(best_text)
        best_scores.append(best_score)
    print(best_scores)
    scorer.clear_gpu_memory()


if __name__=="__main__":
    main()
