from pathlib import Path

import hydra
import pandas as pd

import santa.operator
import santa.sampler
from santa.sa import simulated_annealing
from santa.metrics import PerplexityCalculator
from santa.utils import save_text, load_logs, save_logs


@hydra.main(config_path="conf", config_name="optimize", version_base=None)
def main(cfg):
    ops = [
        getattr(santa.operator, op.operator)(fix_ids=cfg.fix_ids, **op.operator_kwargs)
        for op in cfg.operators
    ]
    sampler = getattr(santa.sampler, cfg.sampler)(ops, **cfg.sampler_kwargs)
    if cfg.initial_solution is None:
        df = pd.read_csv(Path(cfg.dir.data_dir, "sample_submission.csv"))
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
    best_scores, current_scores = [], []
    for _ in range(cfg.num_cycles):
        best_text, best_score, current_text, current_score, precomputed = simulated_annealing(
            best_text, sampler, scorer, precomputed=precomputed, **cfg.sa_kwargs,
        )
        print(f"\nbest score: {best_score:.5f}\nbest text: {best_text}")
        save_text(best_text, best_score, cfg.target_id, output_dir=cfg.dir.output_dir)
        save_text(current_text, current_score, cfg.target_id, output_dir=cfg.dir.output_dir)
        best_scores.append(best_score)
        current_scores.append(current_score)
        precomputed.update(load_logs(cfg.target_id, root_dir=cfg.dir.log_dir))
        save_logs(precomputed, cfg.target_id, root_dir=cfg.dir.log_dir)
    print("current scores\n", current_scores)
    print("best scores\n", best_scores)
    scorer.clear_gpu_memory()


if __name__=="__main__":
    main()
