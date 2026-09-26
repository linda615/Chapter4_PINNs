"""Train the local-load eight-field first-order strong-form control."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/local_strong.json")
    parser.add_argument("--output", type=Path, help="New or empty run directory; existing results are never overwritten.")
    parser.add_argument("--epochs", type=int, help="Override epochs; the published configuration uses 20000.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--no-xla", action="store_true", help="Disable XLA; the reported GPU run used XLA.")
    parser.add_argument("--cuda-data-dir", type=Path, help="Optional CUDA toolkit root containing bin/ and nvvm/libdevice/ for GPU XLA.")
    parser.add_argument("--intra-threads", type=int, default=2)
    parser.add_argument("--inter-threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.intra_threads, args.inter_threads) < 1:
        parser.error("Thread counts must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    os.environ.update({
        "TF_NUM_INTRAOP_THREADS": str(args.intra_threads),
        "TF_NUM_INTEROP_THREADS": str(args.inter_threads),
        "OMP_NUM_THREADS": str(args.intra_threads),
        "TF_ENABLE_ONEDNN_OPTS": "1",
    })
    if args.cuda_data_dir:
        if args.device != "gpu" or not args.cuda_data_dir.is_dir():
            parser.error("--cuda-data-dir requires --device gpu and an existing toolkit directory")
        cuda_dir = str(args.cuda_data_dir.resolve())
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f' --xla_gpu_cuda_data_dir="{cuda_dir}"'
        os.environ["PATH"] = str(args.cuda_data_dir / "bin") + os.pathsep + os.environ.get("PATH", "")
    from mixed_weak_pcrb.config import load_config
    from mixed_weak_pcrb.runtime import configure_device

    config = load_config(args.config)
    if config.get("dtype") != "float32":
        parser.error("The published strong control requires dtype=float32")
    if config.get("physics", {}).get("problem") != "local_gaussian":
        parser.error("This entry point implements the local Gaussian-load control")
    if args.output:
        config["output_dir"] = str(args.output)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    config["training"]["jit_compile"] = not args.no_xla
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    environment = configure_device(args.device)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.intra_threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.inter_threads)
    environment.update({
        "intra_op_threads": args.intra_threads, "inter_op_threads": args.inter_threads,
        "xla_training": not args.no_xla, "precision": "float32", "onednn": True,
        "seed": config.get("seed", 42), "tf32_enabled": False,
        "memory_growth": args.device == "gpu",
        "checkpoint_selection": "final epoch; no reference-based selection",
    })
    config["execution"] = environment
    config["loss_form"] = "first_order_eight_field_strong"
    from mixed_weak_pcrb.strong_form import StrongTrainer
    device_name = "/GPU:0" if args.device == "gpu" else "/CPU:0"
    with tf.device(device_name):
        trainer = StrongTrainer(config)
        if args.device == "gpu" and not all("GPU:0" in v.device for v in trainer.model.trainable_variables):
            raise RuntimeError("Requested GPU placement was not satisfied")
        result = trainer.train()
    print(f"Training artifacts written to: {result}")


if __name__ == "__main__":
    main()
