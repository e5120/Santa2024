from pathlib import Path

import hydra
import pandas as pd

import santa.operator
import santa.sampler
from santa.ga import genetic_algorithm
from santa.metrics import PerplexityCalculator
from santa.utils import setup, save_text, load_logs, save_logs


TOKEN = "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"


@hydra.main(config_path="conf", config_name="optimize_ga", version_base=None)
def main(cfg):
    # setup(cfg)
    df = pd.read_csv(Path(cfg.dir.data_dir, "sample_submission.csv"))
    crossover_ops = []
    for op in cfg.crossover_operators:
        crossover_ops.append(
            getattr(santa.operator, op.name)(**op.kwargs)
        )
    crossover_sampler = getattr(santa.sampler, cfg.crossover_sampler.name)(crossover_ops, **cfg.crossover_sampler.kwargs)
    mutate_ops = []
    for op in cfg.mutate_operators:
        mutate_ops.append(
            getattr(santa.operator, op.name)(**op.kwargs)
        )
    mutate_sampler = getattr(santa.sampler, cfg.mutate_sampler.name)(mutate_ops, **cfg.mutate_sampler.kwargs)
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
    for _ in range(cfg.num_cycles):
        # 初期集団の生成
        best_tokens = best_text.split(" ")
        init_population = [best_text]
        for _ in range(cfg.pop_size - 1):
            op = mutate_sampler.sample()
            init_population.append(" ".join(op(best_tokens)))
        # 最適化
        texts, scores, precomputed = genetic_algorithm(
            init_population, scorer, cfg.n_gens,
            crossover_sampler, mutate_sampler,
            cfg.mutate_rate, cfg.elite_rate,
            precomputed=precomputed,
            logging_step=cfg.logging_step,
        )
        # ロギング
        best_text = texts[0]
        best_score = scores[0]
        print(f"\nbest score: {best_score:.5f}, # of search: {len(precomputed)}, best order: {best_text}")
        save_text(best_text, best_score, cfg.target_id, output_dir=cfg.dir.output_dir)
        best_scores.append(best_score)
        save_logs(precomputed, cfg.target_id, root_dir=cfg.dir.log_dir)
    print(best_scores)
    scorer.clear_gpu_memory()


if __name__=="__main__":
    main()
