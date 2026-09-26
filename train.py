from __future__ import annotations

import argparse
import json
from pathlib import Path

from mixed_weak_pcrb.config import load_config
from mixed_weak_pcrb.trainer import Trainer
from mixed_weak_pcrb.runtime import configure_device


def main():
    parser = argparse.ArgumentParser(description="Train the fixed-G=8 PCRB-Net model.")
    parser.add_argument("--config", required=True, help="Path to a resolved JSON config.")
    parser.add_argument("--epochs", type=int, help="Optional short-run override.")
    parser.add_argument("--output", help="Fresh output directory; existing runs are never overwritten.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    config = load_config(args.config)
    if args.output:
        config["output_dir"] = args.output
    environment = configure_device(args.device)
    environment["precision"] = config["dtype"]
    output = Trainer(config).train(args.epochs)
    (Path(output) / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
