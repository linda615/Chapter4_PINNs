"""Run a fixed-quadrature ablation or a normalization-seed repeat."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mixed_weak_pcrb.config import load_config


def prepare_config(config):
    """Retain active settings while adapting archived fixed-rule configurations."""
    runtime = copy.deepcopy(config)
    if runtime.get("loss_form", "mixed_local_weak") != "mixed_local_weak":
        raise ValueError("This entry point trains the mixed local weak form only.")
    weak = runtime["weak_form"]
    if weak.get("quadrature_adapt", {}).get("enabled", False):
        raise ValueError("Ablations require fixed quadrature; adaptation is enabled.")
    if weak.get("h_adapt", {}).get("enabled", False):
        raise ValueError("Ablations require a fixed subdomain partition.")
    if int(weak["quadrature_order"]) != 8:
        raise ValueError("The archived ablations use fixed quadrature G=8.")

    removed = []
    groups = [("weak_form.weights", weak["weights"])]
    groups.extend(
        (f"weak_form.weight_schedule[{i}]", stage)
        for i, stage in enumerate(weak.get("weight_schedule", []))
    )
    for location, values in groups:
        if "quadrature" in values:
            if weak.get("quadrature_adapt", {}).get("enabled") is not False:
                raise ValueError(
                    "An archived quadrature penalty requires explicitly disabled "
                    "quadrature_adapt before it can be treated as inactive."
                )
            removed.append({"key": location + ".quadrature", "value": values.pop("quadrature")})
    conversion = {
        "removed_inactive_settings": removed,
        "reason": (
            "The source fixed-quadrature training path did not use the quadrature "
            "consistency penalty. Only its inactive weight keys are removed; "
            "the four physical losses, scales, schedules and network are unchanged."
        ),
    }
    return runtime, conversion


def check_output(path):
    output = Path(path).expanduser().resolve()
    archive = (Path(__file__).resolve().parent / "pretrained").resolve()
    if output == archive or archive in output.parents:
        raise ValueError("Training output must be outside the pretrained archive.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Training output is not an empty directory: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="An ablation or seed JSON configuration.")
    parser.add_argument("--output", help="Fresh output directory; defaults to config.output_dir.")
    parser.add_argument("--epochs", type=int, help="Optional positive short-run override.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show settings without training.")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")

    archived = load_config(args.config)
    if args.output:
        archived["output_dir"] = args.output
    config, conversion = prepare_config(archived)
    check_output(config["output_dir"])
    if args.dry_run:
        print(json.dumps({"device": args.device, "config": config, "conversion": conversion}, indent=2))
        return

    from mixed_weak_pcrb.runtime import configure_device

    environment = configure_device(args.device)
    from mixed_weak_pcrb.trainer import Trainer

    environment["precision"] = config["dtype"]
    output = Path(Trainer(config).train(args.epochs))
    for filename, content in (
        ("input_config.json", archived),
        ("configuration_conversion.json", conversion),
        ("environment.json", environment),
    ):
        (output / filename).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
