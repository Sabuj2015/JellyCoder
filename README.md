# Video Encoder Utility

![PyPI - Version](https://img.shields.io/pypi/v/jellycoder)
[![Python package](https://github.com/0x78f1935/JellyCoder/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/0x78f1935/JellyCoder/actions/workflows/python-package.yml)

## Overview

`jelly_coder` scans a folder for supported videos (`.mkv`, `.mp4`, `.wmv`, `.mwv`) and re-encodes them with size-aware defaults. It keeps subtitles/metadata, flips containers when MP4 constraints are violated, and preserves a mirrored directory tree when you choose not to overwrite in place.

Key capabilities:

- Automatic encoder selection with hardware-first preference (NVENC → QSV → AMF → x264) and explicit backend override flags.
- Height presets (`auto`, `1080p`, `720p`, `480p`, `360p`) with bitrate scaling so smaller outputs also shrink file size.
- Smart audio handling: copies compatible streams, forces stereo when sources are mono or pseudo-mono (e.g., WMV files with only one active channel), and warns about downmixing.
- Hardware fallbacks: if a chosen hardware encoder fails to create a session, the tool transparently repeats the job with `libx264` so the batch continues.

## Prerequisites

- Windows PowerShell 5.1 (default shell for the repo scripts).
- Python 3.11+ with virtual environment support.
- ffmpeg and ffprobe in `PATH`.
  - Hardware backends require vendor drivers and an ffmpeg build with NVENC/QSV/AMF enabled.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

## Usage

```powershell
# Show CLI help
python -m jelly_coder --help

# Reduce a folder, mirror outputs under ./output/<dir>, auto-select backend
python -m jelly_coder --input D:\media --quality 720p

# Force Intel QSV, enable overwrite, run at debug verbosity
python -m jelly_coder --input D:\media --encoder-backend qsv --overwrite --log-level debug

# Legacy wrapper remains available
python encode_videos.py --input D:\media
```

### Key Flags

- `--input PATH`: Directory scanned recursively for supported video extensions.
- `--encoder-backend BACKEND`: `auto` (default), `nvenc`, `x264`, `qsv`, or `amf`.
- `--preferred-codec CODEC`: Hint `h264` or `hevc`; respected when the backend supports it.
- `--quality PRESET`: Downscale preset (`auto`, `1080p`, `720p`, `480p`, `360p`).
- `--workers N`: Concurrent encodes (default 1; hardware encoders generally behave best at 1).
- `--overwrite`: Replace sources in place. When omitted, outputs land in `./output/<input-folder>`.
- `--log-level LEVEL`: `info` (default), `debug`, `warning`, etc.

### Output Behavior

- MP4 targets convert SubRip/ASS to `mov_text` automatically; incompatible streams trigger a fallback to MKV.
- Attached artwork, metadata, and subtitle tracks are propagated.
- When the output is larger than the input, a warning is emitted so you can review or delete the file.
- Audio streams are copied when safe; otherwise they are re-encoded to AAC 192k stereo, duplicating the dominant channel if the source is effectively mono.

## Encoder Backends

- **Auto**: Queries ffmpeg encoders and picks the best available hardware backend (NVENC → QSV → AMF → x264).
- **NVENC/QSV/AMF**: Uses vendor hardware. Hardware decode fallbacks are attempted before giving up.
- **x264**: Software-only; always available.
- If a hardware backend exhausts its attempts (e.g., `Error creating a MFX session: -9` on QSV), the utility logs a warning and re-runs the encode with `libx264` automatically.

## Audio Handling

- ffprobe insights drive copy-or-transcode decisions.
- WMV inputs and other stereo tracks with severe channel imbalance trigger a sampling step; if only one channel has meaningful signal, the tool duplicates it so the result plays on both speakers.
- Mono and surround sources are converted to stereo with informative log messages about the change.

## Development

- Run `python -m pytest` and `flake8` before committing; coverage is enforced at 100%.
- `python -m jelly_coder --help` verifies CLI wiring after refactors.
- VS Code launchers in `.vscode/launch.json` provide ready-to-run debug sessions.

## Troubleshooting

- Validate ffmpeg exposes the expected encoders: `ffmpeg -hide_banner -encoders | Select-String nvenc`.
- Hardware backends may need up-to-date GPU/Media drivers; keep Windows and vendor packages current.
- Delete partial outputs in `output/` if you want to re-run without `--overwrite`.
