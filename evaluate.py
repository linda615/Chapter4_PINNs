from __future__ import annotations

import argparse
from pathlib import Path

from mixed_weak_pcrs.config import load_config
from mixed_weak_pcrs.validation import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PCRS-Net checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a resolved JSON config.")
    parser.add_argument("--weights", help="Checkpoint path; defaults to pretrained/<case>.")
    parser.add_argument("--output", help="Validation output directory.")
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--strong-grid-size", type=int, default=81)
    parser.add_argument("--test-order", type=int, default=8)
    parser.add_argument("--quadrature-order", type=int, default=14)
    args = parser.parse_args()

    config_path = Path(args.config)
    case_name = config_path.stem
    weights = Path(args.weights) if args.weights else Path("pretrained")/case_name/"model.weights.h5"
    output = Path(args.output) if args.output else Path("validation")/case_name
    report, output = evaluate(
        load_config(config_path),
        args.grid_size,
        args.strong_grid_size,
        args.test_order,
        args.quadrature_order,
        weights_path=weights,
        validation_output=output,
    )

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
