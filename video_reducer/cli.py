from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from .core import QUALITY_PRESETS, ReducerConfig, reduce_videos

BACKEND_CHOICES = ["auto", "nvenc", "x264", "qsv", "amf"]

DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def _prompt_for_directory() -> Path:
    while True:
        raw = input("Enter the path to the video folder: ").strip().strip('"')
        if not raw:
            logging.warning("Empty input; please provide a folder path.")
            continue
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            return path
        logging.warning("%s is not a valid directory; try again.", path)


def _prompt_overwrite(default: bool = False) -> bool:
    raw = input("Overwrite existing files? [y/N]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def _prompt_quality(default: str = "auto") -> str:
    choices = "/".join(QUALITY_PRESETS)
    default_normalized = default.lower() if default else "auto"
    while True:
        raw = input(f"Select quality ({choices}) [{default_normalized}]: ").strip().lower()
        if not raw:
            return default_normalized
        if raw in QUALITY_PRESETS:
            return raw
        logging.warning("Invalid quality selection '%s'. Choose one of: %s.", raw, ", ".join(QUALITY_PRESETS))


def _prompt_backend(default: str = "auto") -> str:
    choices = "/".join(BACKEND_CHOICES)
    default_normalized = default.lower() if default else "auto"
    while True:
        raw = input(f"Select encoder backend ({choices}) [{default_normalized}]: ").strip().lower()
        if not raw:
            return default_normalized
        if raw in BACKEND_CHOICES:
            return raw
        logging.warning(
            "Invalid backend selection '%s'. Choose one of: %s.", raw, ", ".join(BACKEND_CHOICES)
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="video_reducer",
        description="Reduce video sizes using NVIDIA NVENC while keeping Jellyfin compatibility.",
    )
    parser.add_argument("input", nargs="?", help="Path to the input directory containing videos.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original files instead of writing to an output folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Custom output directory root when not overwriting (default: ./output/<input-name>).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent encodes (default: 1).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity (default: info).",
    )
    parser.add_argument(
        "--codec",
        default="auto",
        choices=["auto", "h264", "hevc"],
        help="Preferred codec for NVENC encoding (default: auto).",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=BACKEND_CHOICES,
        help="Video encoder backend (default: auto).",
    )
    parser.add_argument(
        "--quality",
        default=None,
        type=str.lower,
        choices=QUALITY_PRESETS,
        help="Target video quality preset (default: auto).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    log_level = getattr(logging, args.log_level.upper(), DEFAULT_LOG_LEVEL)
    logging.basicConfig(format=LOG_FORMAT, level=log_level)

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
    else:
        input_path = _prompt_for_directory()

    overwrite = args.overwrite if args.input else _prompt_overwrite(default=args.overwrite)
    quality = args.quality or "auto"
    if not args.input:
        quality = _prompt_quality(default=quality)
    backend = args.backend
    if not args.input:
        backend = _prompt_backend(default=backend)

    config = ReducerConfig(
        input_path=input_path,
        overwrite=overwrite,
        output_root=args.output,
        max_workers=args.max_workers,
        preferred_codec=None if args.codec == "auto" else args.codec,
        quality=quality,
        encoder_backend=backend,
    )

    reduce_videos(config)


if __name__ == "__main__":  # pragma: no cover
    main()
