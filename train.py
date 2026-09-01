from __future__ import annotations

import argparse

from mixed_weak_pcrs.config import load_config
from mixed_weak_pcrs.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train the fixed-G=8 PCRS-Net model.")
    parser.add_argument("--config", required=True, help="Path to a resolved JSON config.")
    parser.add_argument("--epochs", type=int, help="Optional short-run override.")
    args = parser.parse_args()
    Trainer(load_config(args.config)).train(args.epochs)


if __name__ == "__main__":
    main()
