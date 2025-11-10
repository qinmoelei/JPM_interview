from __future__ import annotations
import argparse, os, numpy as np, pandas as pd
from typing import Dict
from src.utils.io import load_config, ensure_dir
from src.utils.logging import get_logger
from src.model.trainer import VPForecaster, training_step

logger = get_logger(__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Placeholder: In a real run, load processed tensors from datahandler steps.
    # Here we create tiny synthetic arrays to validate the training loop.
    B, T, state_dim, driver_dim = 4, 8, 10, 16
    np.random.seed(0)
    states = np.abs(np.random.randn(B, T, state_dim)).astype("float32")
    covs   = np.random.randn(B, T, driver_dim).astype("float32")
    targets= states.copy()  # dummy

    model = VPForecaster(driver_dim=driver_dim, state_dim=state_dim)
    optim = __import__("tensorflow").keras.optimizers.Adam(learning_rate=cfg.training["learning_rate"])

    for epoch in range(cfg.training["epochs"]):
        logs = training_step(model, {"states": states, "covs": covs, "target_states": targets}, optim, w_identity=cfg.training["weight_identity"])
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}: {logs}")

if __name__ == "__main__":
    main()
