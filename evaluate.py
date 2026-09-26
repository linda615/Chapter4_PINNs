from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mixed_weak_pcrb.config import load_config
from mixed_weak_pcrb.validation import evaluate
from mixed_weak_pcrb.runtime import configure_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PCRB-Net checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a resolved JSON config.")
    parser.add_argument("--weights", help="Checkpoint path; defaults to pretrained/<case>.")
    parser.add_argument("--output", help="Validation output directory.")
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--strong-grid-size", type=int, default=81)
    parser.add_argument("--test-order", type=int, default=8)
    parser.add_argument("--quadrature-order", type=int, default=14)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="CPU reproduces the thesis evaluation convention.")
    args = parser.parse_args()

    config_path = Path(args.config)
    case_name = config_path.stem
    weights = Path(args.weights) if args.weights else Path("pretrained")/case_name/"model.weights.h5"
    output = Path(args.output) if args.output else Path("validation")/case_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output is not empty: {output}; choose --output")
    environment = configure_device(args.device)
    environment.update({"network_dtype": load_config(config_path)["dtype"], "field_error_accumulation_dtype": "float64", "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()})
    report, output = evaluate(
        load_config(config_path),
        args.grid_size,
        args.strong_grid_size,
        args.test_order,
        args.quadrature_order,
        weights_path=weights,
        validation_output=output,
    )
    (output / "evaluation_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    print(f"Validation results written to: {output}")
    print("Field relative L2 errors:")
    for row in report["field_errors"]:
        print(f"  {row['field']:>8s}: {row['relative_l2']:.6e}")
    print("Peak magnitude and location errors:")
    for row in report["peak_response"]:
        print(
            f"  {row['field']:>8s}: magnitude={row['relative_peak_error']:.6e}, "
            f"location={row['normalized_location_error']:.6e}"
        )
    print(
        "Center deflection relative error: "
        f"{report['center_deflection']['relative_error']:.6e}"
    )
    print(
        "Global balance relative error:   "
        f"{report['global_balance']['relative_error']:.6e}"
    )


if __name__ == "__main__":
    main()
