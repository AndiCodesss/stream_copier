from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the local intent classifier from reviewed execution datasets.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["data/interpretation/full_ai_corpus_sonnet_1000"],
        help="Reviewed corpus directories or reviewed_examples.jsonl files.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/training_data.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/training_data_summary.json"),
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    command_prefix = [sys.executable, "-m"]

    dataset_command = [
        *command_prefix,
        "app.services.interpretation.build_reviewed_execution_dataset",
        *args.inputs,
        "--jsonl-out",
        str(args.dataset),
        "--summary-out",
        str(args.summary),
    ]
    train_command = [
        *command_prefix,
        "app.services.interpretation.train_local_classifier",
        "--dataset",
        str(args.dataset),
        "--artifact-dir",
        str(args.artifact_dir),
        "--device",
        args.device,
    ]

    subprocess.run(dataset_command, check=True)
    subprocess.run(train_command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
