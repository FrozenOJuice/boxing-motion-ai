"""Local reviewed-footage binary detector commands. Never opens reserved recordings."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Fixed V4-only training-fit smoke test")
    smoke.add_argument("--config", default="configs/detector_smoke_v1.json")
    smoke.add_argument("--output", required=True, help="Fresh ignored reports/ directory")
    inference = commands.add_parser(
        "infer", help="Causal probabilities on registered reviewed intervals"
    )
    inference.add_argument("--run", required=True, help="Smoke output directory under reports/")
    inference.add_argument("--output", required=True)
    inference.add_argument("--recording", choices=["V4", "V6"], default="V4")
    inference.add_argument(
        "--decoder-config",
        default="configs/decoder_v1.json",
        help="Fixed, untuned timestamp-based event policy JSON",
    )
    args = parser.parse_args()
    from .smoke import predict, run

    try:
        result = (
            run(Path.cwd(), args.config, args.output)
            if args.command == "smoke"
            else predict(Path.cwd(), args.run, args.output, args.recording, args.decoder_config)
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Detector command failed: {exc}\n")
    if args.command == "smoke":
        print(
            "Training-only loss:",
            result["initial"]["weighted_loss"],
            "->",
            result["final"]["weighted_loss"],
        )
    else:
        print(result)


if __name__ == "__main__":
    main()
