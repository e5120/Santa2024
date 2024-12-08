from pathlib import Path

import hydra
import torch
import numpy as np
import pandas as pd

from santa.cma_es import FastCMA
from santa.metrics import PerplexityCalculator
from santa.utils import setup


TOKEN = "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"


@hydra.main(config_path="conf", config_name="optimize_cma_es", version_base=None)
def main(cfg):
    # setup(cfg)
    df = pd.read_csv(Path(cfg.dir.data_dir, "sample_submission.csv"))
    tokens = np.array(df.loc[cfg.target_id, "text"].strip().split(" "))
    scorer = PerplexityCalculator(
        model_path=cfg.model_path,
        load_in_8bit=cfg.load_in_8bit,
        device_map=cfg.device_map,
    )
    with torch.no_grad():
        best_reward = None
        best_text = None
        cma_es = FastCMA(N = len(tokens), samples=cfg.pop_size)
        for epoch in range(cfg.epochs):
            try:
                res = cma_es.step(objective_f=scorer.get_perplexity, tokens=tokens, step_size=cfg.lr)
            except Exception as e:
                print(e)
                break
            if best_reward is None:
                best_reward = res[0]['fitness']
                best_text = res[0]['tokens']
            if res[0]['fitness'] < best_reward:
                # plot_sol(res[0]['parameters'])
                best_reward = res[0]['fitness']
                best_text = res[0]['tokens']
                # print("%i %f" % (epoch, best_reward))
            print("%i %f" % (epoch, res[0]["fitness"]))
    with open(f"{cfg.dir.output_dir}/id{cfg.target_id}_{best_reward:.5f}.txt", "w") as f:
        f.write(best_text)
    scorer.clear_gpu_memory()


if __name__=="__main__":
    main()
