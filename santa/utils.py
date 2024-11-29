import hydra
import lightning as L


def setup(cfg):
    cfg.dir.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    L.seed_everything(cfg["seed"])
