from pathlib import Path

import hydra
import pandas as pd

import santa.operator
import santa.sampler
from santa.sa import simulated_annealing
from santa.metrics import PerplexityCalculator
from santa.utils import setup, save_text, load_logs, save_logs, save_history


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
    scorer = PerplexityCalculator(
        model_path=cfg.model_path,
        load_in_8bit=cfg.load_in_8bit,
        device_map=cfg.device_map,
    )
    precomputed = load_logs(cfg.target_id, root_dir=cfg.dir.log_dir)
    text_history, score_history = [], []
    for _ in range(cfg.num_cycles):
        best_text, best_score, precomputed, th, sh = simulated_annealing(
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
        text_history += th
        score_history += sh
        print(f"\nbest score: {best_score:.5f}, # of search: {len(precomputed)}, best order: {best_text}")
        save_text(best_text, best_score, cfg.target_id, output_dir=cfg.dir.output_dir)
        best_scores.append(best_score)
        save_logs(precomputed, cfg.target_id, root_dir=cfg.dir.log_dir)
        save_history(text_history, score_history, cfg.target_id, root_dir=cfg.dir.log_dir)
    print(best_scores)
    scorer.clear_gpu_memory()


if __name__=="__main__":
    main()
