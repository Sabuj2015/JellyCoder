# Video Encoder Utility

## Overview

- `--codec CODEC`: Choose `h264` (default, widest compatibility) or `hevc` for higher compression.
- `--quality` (`auto`/`1080p`/`720p`/`480p`/`360p`): Downscale video to the selected height while maintaining aspect ratio (default `auto`).

## Prerequisites

## Installation

.\.venv\Scripts\Activate.ps1

# Install project in editable mode (no external deps today)

pip install -e .

````

## Usage

```powershell
# Show CLI help

# Legacy wrapper remains available and forwards to the same CLI
python encode_videos.py
````

### Key Flags

- `--input PATH`: Directory that will be scanned for `.mkv`/`.mp4` files.
- `--overwrite`: Replace source files in place. Omit to mirror output under `./output/<input-name>`.
- `--workers N`: Number of concurrent encodes (default 1; NVENC prefers single jobs).
- `--log-level LEVEL`: Adjust logging verbosity (`info`, `debug`, etc.).
- `--codec CODEC`: Choose `h264` (default, widest compatibility) or `hevc` for higher compression.
- `--quality PRESET`: Automatically downscale video height to `1080p`, `720p`, `480p`, `360p`, or leave unchanged with `auto` (default).

### Output Behavior

- When overwrite is disabled, the command writes to `output/<input-folder>` relative to the process working directory.
- Subtitles and metadata are copied forward automatically.
- If the encoded file is larger than the source, a warning is logged so you can review it later.
- Files targeting MP4 automatically convert SubRip/ASS subtitles to `mov_text`; if subtitles, extra video streams, or attachments make MP4 unsuitable, the encoder falls back to MKV for that title.

## Development

- Run `python -m video_reducer --help` after edits to ensure argument parsing still works.
- Use `python -m video_reducer --input <path> --overwrite --codec h264` for smoke tests against short clips.
- The VS Code launch configurations in `.vscode/launch.json` provide ready-to-run debug sessions.

## Troubleshooting

- If ffmpeg cannot find NVENC, verify the build exposes `h264_nvenc`/`hevc_nvenc` via `ffmpeg -encoders | Select-String nvenc`.
- Windows PowerShell may buffer progress output; use the integrated terminal for the cleanest experience.
- Delete partial outputs in `output/` if you need to re-run without the `--overwrite` flag.
