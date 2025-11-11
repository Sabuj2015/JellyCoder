from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional

from array import array

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".wmv", ".mwv"}
NVENC_PRESET = "p6"
NVENC_RC_MODE = "vbr_hq"
NVENC_CQ = 24
NVENC_LOOKAHEAD = "20"
NVENC_SPATIAL_AQ = "1"
NVENC_TEMPORAL_AQ = "1"
NVENC_TUNE = "hq"
AUDIO_BITRATE = "192k"
DEFAULT_MAX_WORKERS = 1
PROGRESS_BAR_WIDTH = 40
PROGRESS_UPDATE_INTERVAL = 0.3
PREFERRED_NVENC_ENCODERS = ["h264_nvenc", "hevc_nvenc"]
NVENC_PROFILE_MAP = {"hevc_nvenc": "main", "h264_nvenc": "high"}
HEVC_BITRATE_RATIO = 0.6
H264_BITRATE_RATIO = 0.75
MIN_TARGET_BITRATE_KBPS = 350
MP4_ALLOWED_AUDIO_CODECS = {"aac", "ac3", "eac3", "mp3", "alac"}
MP4_ALLOWED_SUBTITLE_CODECS = {"mov_text"}
MP4_CONVERTIBLE_SUBTITLE_CODECS = {"subrip", "srt", "ass", "ssa"}
MP4_ALLOWED_ATTACHED_PIC_CODECS = {"mjpeg", "png"}
QUALITY_PRESETS: tuple[str, ...] = ("auto", "1080p", "720p", "480p", "360p")
QUALITY_HEIGHT_MAP = {
    "1080p": 1080,
    "720p": 720,
    "480p": 480,
    "360p": 360,
}
SUPPORTED_ENCODER_BACKENDS = ("auto", "nvenc", "x264", "qsv", "amf")
X264_ENCODER_NAME = "libx264"
X264_DEFAULT_PRESET = "medium"
QSV_ENCODERS = {"h264": "h264_qsv", "hevc": "hevc_qsv"}
QSV_DEFAULT_PRESET = "medium"
AMF_ENCODERS = {"h264": "h264_amf", "hevc": "hevc_amf"}
AMF_DEFAULT_QUALITY = "quality"
DEFAULT_ENCODER_PRIORITY = ("nvenc", "qsv", "amf", "x264")
EFFECTIVE_MONO_SAMPLE_SECONDS = 90
EFFECTIVE_MONO_IMBALANCE_THRESHOLD_DB = 18.0


def _safe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _safe_fraction(value: object) -> Optional[float]:
    if not isinstance(value, str) or "/" not in value:
        return None
    num_str, den_str = value.split("/", 1)
    num = _safe_float(num_str)
    den = _safe_float(den_str)
    if num is None or not den:
        return None
    fps = num / den
    return fps if fps else None


def _detect_pseudo_mono_channel(
    path: Path,
    sample_seconds: int = EFFECTIVE_MONO_SAMPLE_SECONDS,
    imbalance_threshold_db: float = EFFECTIVE_MONO_IMBALANCE_THRESHOLD_DB,
) -> Optional[int]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "2",
        "-f",
        "s16le",
        "-t",
        str(sample_seconds),
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    except FileNotFoundError:
        logging.debug("ffmpeg not found while sampling audio for %s.", path)
        return None
    except subprocess.CalledProcessError as exc:
        logging.debug("ffmpeg failed while sampling audio for %s: %s", path, exc)
        return None

    data = result.stdout

    usable = len(data) - (len(data) % 4)
    if usable < 4:
        return None

    samples = array("h")
    samples.frombytes(data[:usable])

    left_sq = 0
    right_sq = 0
    limit = len(samples) - (len(samples) % 2)
    for idx in range(0, limit, 2):
        left = samples[idx]
        right = samples[idx + 1]
        left_sq += left * left
        right_sq += right * right

    if left_sq == 0 and right_sq == 0:
        return None

    dominant = 0 if left_sq >= right_sq else 1
    quiet = right_sq if dominant == 0 else left_sq
    loud = left_sq if dominant == 0 else right_sq

    if quiet == 0:
        return dominant

    ratio = loud / quiet
    if ratio <= 1.0:
        return None

    imbalance_db = 10.0 * math.log10(ratio)
    if imbalance_db >= imbalance_threshold_db:
        return dominant
    return None


@dataclass(slots=True)
class MediaInfo:
    frames: Optional[int]
    duration: Optional[float]
    bitrate_kbps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    audio_codec: Optional[str]
    audio_bitrate_kbps: Optional[float]
    audio_channels: Optional[int]
    subtitle_codecs: List[str]
    video_codecs: List[str]
    attached_pic_codecs: List[str]
    data_stream_codecs: List[str]


@dataclass(slots=True)
class EncoderSelection:
    encoder: str
    output_extension: str
    backend: str
    codec_family: str


@dataclass(slots=True)
class ReducerConfig:
    input_path: Path
    overwrite: bool = False
    output_root: Optional[Path] = None
    max_workers: int = DEFAULT_MAX_WORKERS
    preferred_codec: Optional[str] = None
    quality: str = "auto"
    encoder_backend: str = "auto"


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install it before running this command.")


def _normalize_codec(preferred: Optional[str]) -> Optional[str]:
    if preferred is None:
        return None
    mapping = {
        "h264": "h264",
        "avc": "h264",
        "x264": "h264",
        "hevc": "hevc",
        "h265": "hevc",
    }
    normalized = preferred.lower()
    if normalized not in mapping:
        raise ValueError(f"Unsupported codec preference: {preferred}")
    return mapping[normalized]


def _query_ffmpeg_video_encoders() -> set[str]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Unable to query ffmpeg encoders. Check your ffmpeg installation.") from exc

    available_encoders: set[str] = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("------"):
            continue
        if not stripped.startswith("V"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            available_encoders.add(parts[1])
    return available_encoders


def _make_selection(encoder: str, backend: str) -> EncoderSelection:
    codec_family = "hevc" if "hevc" in encoder else "h264"
    extension = ".mkv" if codec_family == "hevc" else ".mp4"
    return EncoderSelection(encoder=encoder, output_extension=extension, backend=backend, codec_family=codec_family)


def select_encoder(backend: str = "auto", preferred: Optional[str] = None) -> EncoderSelection:
    backend_normalized = (backend or "auto").lower()
    if backend_normalized not in SUPPORTED_ENCODER_BACKENDS:
        raise ValueError(f"Unsupported encoder backend: {backend}")

    preferred_family = _normalize_codec(preferred)
    available_encoders = _query_ffmpeg_video_encoders()

    def select_nvenc() -> Optional[EncoderSelection]:
        candidate_order: List[str]
        if preferred_family == "hevc":
            candidate_order = ["hevc_nvenc", "h264_nvenc"]
        elif preferred_family == "h264":
            candidate_order = ["h264_nvenc", "hevc_nvenc"]
        else:
            candidate_order = list(PREFERRED_NVENC_ENCODERS)
        for candidate in PREFERRED_NVENC_ENCODERS:
            if candidate not in candidate_order:
                candidate_order.append(candidate)

        for candidate in candidate_order:
            if candidate in available_encoders:
                if preferred_family and not candidate.startswith(preferred_family):
                    logging.warning(
                        "Requested NVENC codec %s is not available. Falling back to %s.",
                        preferred_family,
                        candidate,
                    )
                return _make_selection(candidate, "nvenc")

        if preferred_family:
            logging.warning(
                "Requested NVENC codec %s is not available.",
                preferred_family,
            )
        return None

    def select_qsv() -> Optional[EncoderSelection]:
        if preferred_family == "hevc":
            order = [QSV_ENCODERS["hevc"], QSV_ENCODERS["h264"]]
        else:
            order = [QSV_ENCODERS["h264"], QSV_ENCODERS["hevc"]]
        for candidate in order:
            if candidate in available_encoders:
                if preferred_family and preferred_family not in candidate:
                    logging.warning(
                        "Requested QSV codec %s is not available. Falling back to %s.",
                        preferred_family,
                        candidate,
                    )
                return _make_selection(candidate, "qsv")
        if preferred_family:
            logging.warning("Requested QSV codec %s is not available.", preferred_family)
        return None

    def select_amf() -> Optional[EncoderSelection]:
        if preferred_family == "hevc":
            order = [AMF_ENCODERS["hevc"], AMF_ENCODERS["h264"]]
        else:
            order = [AMF_ENCODERS["h264"], AMF_ENCODERS["hevc"]]
        for candidate in order:
            if candidate in available_encoders:
                if preferred_family and preferred_family not in candidate:
                    logging.warning(
                        "Requested AMF codec %s is not available. Falling back to %s.",
                        preferred_family,
                        candidate,
                    )
                return _make_selection(candidate, "amf")
        if preferred_family:
            logging.warning("Requested AMF codec %s is not available.", preferred_family)
        return None

    def select_x264() -> Optional[EncoderSelection]:
        if preferred_family == "hevc":
            logging.warning("libx264 does not support HEVC; encoding will use H.264 instead.")
        if X264_ENCODER_NAME in available_encoders:
            return _make_selection(X264_ENCODER_NAME, "x264")
        return None

    backend_selectors = {
        "nvenc": select_nvenc,
        "x264": select_x264,
        "qsv": select_qsv,
        "amf": select_amf,
    }

    if backend_normalized == "auto":
        for candidate_backend in DEFAULT_ENCODER_PRIORITY:
            selector = backend_selectors[candidate_backend]
            selection = selector()
            if selection:
                return selection
        raise RuntimeError("ffmpeg does not expose a supported hardware encoder; libx264 is also unavailable.")

    selection = backend_selectors[backend_normalized]()
    if selection:
        return selection

    if backend_normalized == "nvenc":
        raise RuntimeError(
            "ffmpeg does not expose an NVENC encoder (h264_nvenc or hevc_nvenc). "
            "Install an NVENC-enabled build and ensure NVIDIA drivers are up to date."
        )

    raise RuntimeError(f"ffmpeg does not expose the requested encoder backend: {backend_normalized}")


def discover_videos(base_dir: Path, ignore_dir: Optional[Path]) -> List[Path]:
    videos: List[Path] = []
    ignore_dir_resolved = ignore_dir.resolve() if ignore_dir else None
    for root, dirs, files in os.walk(base_dir):
        current = Path(root).resolve()
        if ignore_dir_resolved:
            dirs[:] = [d for d in dirs if (current / d).resolve(strict=False) != ignore_dir_resolved]
            if current == ignore_dir_resolved or ignore_dir_resolved in current.parents:
                continue
        for name in files:
            path = current / name
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
    return sorted(videos)


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units[:-1]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} {units[-1]}"


def build_output_path(
    src: Path,
    base_input: Path,
    overwrite: bool,
    output_root: Optional[Path],
    extension: str,
) -> Path:
    relative = src.relative_to(base_input)
    target_name = relative.with_suffix(extension)
    if overwrite:
        return src.with_suffix(extension)
    assert output_root is not None
    return output_root / target_name


def probe_media_info(path: Path) -> MediaInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=bit_rate,duration:stream=index,codec_type,codec_name,bit_rate,nb_frames,avg_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        logging.debug("ffprobe not found; media metrics unavailable.")
        return MediaInfo(
            frames=None,
            duration=None,
            bitrate_kbps=None,
            width=None,
            height=None,
            audio_codec=None,
            audio_bitrate_kbps=None,
            audio_channels=None,
            subtitle_codecs=[],
            video_codecs=[],
            attached_pic_codecs=[],
            data_stream_codecs=[],
        )
    except subprocess.CalledProcessError as exc:
        logging.debug("ffprobe failed for %s: %s", path, exc.stderr.strip())
        return MediaInfo(
            frames=None,
            duration=None,
            bitrate_kbps=None,
            width=None,
            height=None,
            audio_codec=None,
            audio_bitrate_kbps=None,
            audio_channels=None,
            subtitle_codecs=[],
            video_codecs=[],
            attached_pic_codecs=[],
            data_stream_codecs=[],
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logging.debug("Unable to parse ffprobe output for %s: %s", path, exc)
        return MediaInfo(
            frames=None,
            duration=None,
            bitrate_kbps=None,
            width=None,
            height=None,
            audio_codec=None,
            audio_bitrate_kbps=None,
            audio_channels=None,
            subtitle_codecs=[],
            video_codecs=[],
            attached_pic_codecs=[],
            data_stream_codecs=[],
        )

    streams = data.get("streams", [])
    total_frames: Optional[int] = None
    duration_sec: Optional[float] = None
    audio_codec: Optional[str] = None
    audio_bitrate_kbps: Optional[float] = None
    audio_channels: Optional[int] = None
    primary_width: Optional[int] = None
    primary_height: Optional[int] = None
    subtitle_codecs: List[str] = []
    video_codecs: List[str] = []
    attached_pic_codecs: List[str] = []
    data_stream_codecs: List[str] = []

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            if isinstance(stream.get("codec_name"), str):
                video_codecs.append(stream["codec_name"].lower())
            disposition = stream.get("disposition") or {}
            if disposition.get("attached_pic") == 1:
                codec_name = stream.get("codec_name")
                if isinstance(codec_name, str):
                    attached_pic_codecs.append(codec_name.lower())
        if codec_type == "video" and total_frames is None:
            nb_frames = stream.get("nb_frames")
            if isinstance(nb_frames, str) and nb_frames.isdigit():
                total_frames = int(nb_frames)

            duration_candidate = _safe_float(stream.get("duration"))
            if duration_candidate is not None:
                duration_sec = duration_candidate

            if total_frames is None:
                fps = _safe_fraction(stream.get("avg_frame_rate"))
                if fps and duration_sec is not None:
                    total_frames = int(duration_sec * fps)
        if codec_type == "video" and (primary_width is None or primary_height is None):
            width = stream.get("width")
            if isinstance(width, int) and primary_width is None:
                primary_width = width
            height = stream.get("height")
            if isinstance(height, int) and primary_height is None:
                primary_height = height
        elif codec_type == "audio" and audio_codec is None:
            codec_name = stream.get("codec_name")
            if isinstance(codec_name, str):
                audio_codec = codec_name.lower()
            stream_bitrate = _safe_float(stream.get("bit_rate"))
            if stream_bitrate is not None:
                audio_bitrate_kbps = stream_bitrate / 1000.0
            channels_value = stream.get("channels")
            if isinstance(channels_value, int):
                audio_channels = channels_value
        elif codec_type == "subtitle":
            codec_name = stream.get("codec_name")
            if isinstance(codec_name, str):
                subtitle_codecs.append(codec_name.lower())
        elif codec_type in {"data", "attachment"}:
            codec_name = stream.get("codec_name")
            if isinstance(codec_name, str):
                data_stream_codecs.append(codec_name.lower())

    bitrate_kbps: Optional[float] = None
    format_section = data.get("format", {})
    bit_rate_value = _safe_float(format_section.get("bit_rate"))
    if bit_rate_value is not None:
        bitrate_kbps = bit_rate_value / 1000.0

    if duration_sec is None:
        duration_candidate = _safe_float(format_section.get("duration"))
        if duration_candidate is not None:
            duration_sec = duration_candidate

    return MediaInfo(
        frames=total_frames,
        duration=duration_sec,
        bitrate_kbps=bitrate_kbps,
        width=primary_width,
        height=primary_height,
        audio_codec=audio_codec,
        audio_bitrate_kbps=audio_bitrate_kbps,
        audio_channels=audio_channels,
        subtitle_codecs=subtitle_codecs,
        video_codecs=video_codecs,
        attached_pic_codecs=attached_pic_codecs,
        data_stream_codecs=data_stream_codecs,
    )


def _format_progress_bar(ratio: Optional[float]) -> str:
    if ratio is None:
        ratio = 0.0
    ratio = max(0.0, min(1.0, ratio))
    filled = int(PROGRESS_BAR_WIDTH * ratio)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    percent = f"{ratio * 100:5.1f}%"
    return f"[{bar}] {percent}"


def _render_progress_line(
    current_frame: Optional[int],
    total_frames: Optional[int],
    out_time_sec: Optional[float],
    total_duration: Optional[float],
) -> tuple[str, Optional[float]]:
    ratio: Optional[float] = None
    label: str

    if total_frames and total_frames > 0 and current_frame is not None:
        ratio = current_frame / total_frames
        label = f"frame {current_frame}/{total_frames}"
    elif total_duration and total_duration > 0 and out_time_sec is not None:
        ratio = out_time_sec / total_duration
        label = f"time {out_time_sec:0.1f}s/{total_duration:0.1f}s"
    elif current_frame is not None:
        label = f"frame {current_frame}"
    elif out_time_sec is not None:
        label = f"time {out_time_sec:0.1f}s"
    else:
        label = "encoding"

    bar = _format_progress_bar(ratio)
    return f"{bar} {label}", ratio


def _display_progress(line: str) -> str:
    padded = line.ljust(PROGRESS_BAR_WIDTH + 30)
    sys.stdout.write("\r" + padded)
    sys.stdout.flush()
    return padded


def _clear_progress(line: str) -> None:
    if not line:
        return
    sys.stdout.write("\r" + " " * len(line) + "\r")
    sys.stdout.flush()


def run_ffmpeg_with_progress(cmd: List[str], total_frames: Optional[int], total_duration: Optional[float]) -> None:
    current_frame: Optional[int] = None
    out_time_sec: Optional[float] = None
    last_render_time = 0.0
    last_line = ""
    progress_lines: List[str] = []

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            progress_lines.append(line)
            if line.startswith("frame="):
                try:
                    current_frame = int(line.split("=", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("out_time_ms="):
                try:
                    out_time_sec = int(line.split("=", 1)[1]) / 1_000_000
                except ValueError:
                    pass
            elif line == "progress=end":
                current_frame = total_frames if total_frames else current_frame
                out_time_sec = total_duration if total_duration else out_time_sec

            now = time.time()
            if now - last_render_time >= PROGRESS_UPDATE_INTERVAL or line == "progress=end":
                display_line, _ = _render_progress_line(
                    current_frame,
                    total_frames,
                    out_time_sec,
                    total_duration,
                )
                last_line = _display_progress(display_line)
                last_render_time = now

        if last_line:
            display_line, _ = _render_progress_line(
                total_frames if total_frames else current_frame,
                total_frames,
                total_duration if total_duration else out_time_sec,
                total_duration,
            )
            last_line = _display_progress(display_line)
        sys.stdout.write("\n")
        sys.stdout.flush()
    finally:
        if process.stdout:
            process.stdout.close()

    stderr_output = ""
    if process.stderr:
        stderr_output = process.stderr.read()
        process.stderr.close()

    return_code = process.wait()
    if return_code != 0:
        _clear_progress(last_line)
        raise subprocess.CalledProcessError(return_code, cmd, output="\n".join(progress_lines), stderr=stderr_output)


def encode_video(
    src: Path,
    dst: Path,
    overwrite: bool,
    encoder: str,
    output_extension: str,
    quality: str,
    *,
    _allow_encoder_fallback: bool = True,
    _media_info_override: Optional[MediaInfo] = None,
) -> None:
    media_info = _media_info_override or probe_media_info(src)

    effective_extension = output_extension
    is_mp4_output = effective_extension.lower() == ".mp4"
    subtitle_args: List[str] = ["-c:s", "copy"]

    encoder_lower = encoder.lower()
    is_nvenc_encoder = encoder_lower.endswith("_nvenc")
    is_qsv_encoder = encoder_lower.endswith("_qsv")
    is_amf_encoder = encoder_lower.endswith("_amf")
    is_x264_encoder = encoder_lower == X264_ENCODER_NAME

    if is_mp4_output:
        subtitle_codecs = [code.lower() for code in media_info.subtitle_codecs]
        allowed_subtitle_codecs = MP4_ALLOWED_SUBTITLE_CODECS | MP4_CONVERTIBLE_SUBTITLE_CODECS
        unsupported_subs = [code for code in subtitle_codecs if code not in allowed_subtitle_codecs]
        if unsupported_subs:
            logging.warning(
                "Subtitle codec(s) %s are not compatible with MP4; switching to MKV container for this file.",
                ", ".join(sorted(set(unsupported_subs))),
            )
            effective_extension = ".mkv"
            is_mp4_output = False
            subtitle_args = ["-c:s", "copy"]
        elif subtitle_codecs and any(code in MP4_CONVERTIBLE_SUBTITLE_CODECS for code in subtitle_codecs):
            subtitle_args = ["-c:s", "mov_text"]
            logging.info("Subtitle streams converted to mov_text for MP4 compatibility.")
        else:
            subtitle_args = ["-c:s", "copy"]

        mp4_incompatible_reasons: List[str] = []
        if len(media_info.video_codecs) > 1:
            mp4_incompatible_reasons.append("multiple video streams present")
        unsupported_attached = [
            codec for codec in media_info.attached_pic_codecs if codec not in MP4_ALLOWED_ATTACHED_PIC_CODECS
        ]
        if unsupported_attached:
            mp4_incompatible_reasons.append(
                "attached pictures with codecs " + ", ".join(sorted(set(unsupported_attached)))
            )
        if media_info.data_stream_codecs:
            mp4_incompatible_reasons.append("data/attachment streams present")

        if mp4_incompatible_reasons:
            logging.warning(
                "MP4 container not suitable (%s); switching to MKV for this file.",
                "; ".join(mp4_incompatible_reasons),
            )
            effective_extension = ".mkv"
            is_mp4_output = False
            subtitle_args = ["-c:s", "copy"]

    if effective_extension != output_extension:
        dst = dst.with_suffix(effective_extension)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and dst.exists():
        logging.info("Skipping existing file: %s", dst)
        return

    original_size = src.stat().st_size
    if overwrite:
        temp_dst = dst.with_name(f"_{dst.name}")
    else:
        temp_dst = dst

    if temp_dst.exists():
        try:
            temp_dst.unlink()
        except OSError:
            pass

    total_frames = media_info.frames
    total_duration = media_info.duration
    logging.debug(
        "Media info for %s - frames: %s, duration: %s, bitrate: %s kbps",
        src,
        total_frames,
        total_duration,
        media_info.bitrate_kbps,
    )

    target_bitrate_kbps: Optional[int] = None
    if media_info.bitrate_kbps:
        ratio = HEVC_BITRATE_RATIO if "hevc" in encoder_lower else H264_BITRATE_RATIO
        proposed = int(media_info.bitrate_kbps * ratio)
        target_bitrate_kbps = max(proposed, MIN_TARGET_BITRATE_KBPS)
        ceiling = int(media_info.bitrate_kbps * 0.95)
        if ceiling > 0:
            target_bitrate_kbps = min(target_bitrate_kbps, ceiling)
        if target_bitrate_kbps <= 0:
            target_bitrate_kbps = None

    audio_args: List[str]
    audio_description_parts: List[str] = []
    if media_info.audio_codec:
        audio_description_parts.append(media_info.audio_codec)
    if media_info.audio_bitrate_kbps:
        audio_description_parts.append(f"~{media_info.audio_bitrate_kbps:.0f} kbps")
    audio_description = " ".join(audio_description_parts) if audio_description_parts else "unknown"

    audio_codec_lower = media_info.audio_codec.lower() if media_info.audio_codec else None
    audio_channels = media_info.audio_channels
    actual_mono_source = audio_channels is not None and audio_channels < 2

    pseudo_mono_channel: Optional[int] = None
    if (
        not actual_mono_source
        and audio_codec_lower
        and (audio_channels is None or audio_channels >= 2)
    ):
        pseudo_mono_channel = _detect_pseudo_mono_channel(src)

    force_stereo = actual_mono_source or pseudo_mono_channel is not None

    audio_filter_arg: Optional[str] = None
    if pseudo_mono_channel is not None:
        channel_token = f"c{pseudo_mono_channel}"
        audio_filter_arg = f"pan=stereo|c0={channel_token}|c1={channel_token}"

    audio_can_copy = audio_codec_lower is not None and not force_stereo
    if audio_can_copy and is_mp4_output:
        audio_can_copy = audio_codec_lower in MP4_ALLOWED_AUDIO_CODECS

    if audio_can_copy:
        audio_args = ["-c:a", "copy"]
        logging.info("Audio stream will be copied (%s).", audio_description)
    else:
        target_channels = 2
        audio_args = ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", str(target_channels)]
        log_message = "Audio stream will be transcoded to AAC %s (original %s)." % (AUDIO_BITRATE, audio_description)
        if pseudo_mono_channel is not None:
            channel_label = "left" if pseudo_mono_channel == 0 else "right"
            log_message += (
                f" Detected {channel_label} channel dominance; "
                f"duplicating {channel_label} channel to both outputs."
            )
        elif actual_mono_source:
            log_message += " Mono source detected; forcing stereo output."
        elif audio_channels and audio_channels > target_channels:
            log_message += f" Downmixing from {audio_channels} channels to stereo."
        logging.info(log_message)

    normalized_quality = quality.lower()
    scale_filter: Optional[str] = None
    if normalized_quality != "auto":
        target_height = QUALITY_HEIGHT_MAP.get(normalized_quality)
        if target_height is None:
            logging.warning("Unknown quality preset '%s'; defaulting to auto.", quality)
            normalized_quality = "auto"
        else:
            source_height = media_info.height
            if source_height is None or source_height > target_height:
                scale_filter = f"scale=-2:{target_height}"
                if source_height:
                    logging.info(
                        "Video will be scaled from %sp to %sp height (maintaining aspect ratio).",
                        source_height,
                        target_height,
                    )
                else:
                    logging.info(
                        "Video will be scaled to a maximum height of %sp (source height unknown).",
                        target_height,
                    )
                if target_bitrate_kbps and source_height:
                    scale_ratio = min(target_height / source_height, 1.0)
                    adjusted = max(int(target_bitrate_kbps * scale_ratio * scale_ratio), MIN_TARGET_BITRATE_KBPS)
                    if adjusted != target_bitrate_kbps:
                        logging.debug(
                            "Adjusting target bitrate from %s kbps to %s kbps based on scale ratio %.3f.",
                            target_bitrate_kbps,
                            adjusted,
                            scale_ratio,
                        )
                        target_bitrate_kbps = adjusted
            else:
                logging.info(
                    "Source height %sp is already <= target %sp; no scaling applied.",
                    source_height,
                    target_height,
                )

    def build_ffmpeg_cmd(use_hw_decode: bool, force_hw_format: bool) -> List[str]:
        cmd: List[str] = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
        ]

        if use_hw_decode:
            cmd.extend(["-hwaccel", "cuda"])
            if force_hw_format and scale_filter is None:
                cmd.extend(["-hwaccel_output_format", "cuda"])

        cmd.extend(["-progress", "pipe:1", "-nostats"])

        cmd.extend([
            "-i",
            str(src),
            "-map",
            "0",
            "-c:v",
            encoder,
        ])

        if is_nvenc_encoder:
            nvenc_profile = NVENC_PROFILE_MAP.get(encoder)
            if nvenc_profile:
                cmd.extend(["-profile:v", nvenc_profile])
            if NVENC_TUNE:
                cmd.extend(["-tune", NVENC_TUNE])
            cmd.extend([
                "-preset",
                NVENC_PRESET,
                "-rc",
                NVENC_RC_MODE,
                "-cq",
                str(NVENC_CQ),
            ])
        elif is_x264_encoder:
            cmd.extend(["-preset", X264_DEFAULT_PRESET])
        elif is_qsv_encoder:
            cmd.extend(["-preset", QSV_DEFAULT_PRESET])
        elif is_amf_encoder:
            cmd.extend(["-quality", AMF_DEFAULT_QUALITY])

        if target_bitrate_kbps:
            video_bitrate = f"{target_bitrate_kbps}k"
            maxrate = f"{int(target_bitrate_kbps * 1.15)}k"
            bufsize = f"{int(target_bitrate_kbps * 2)}k"
            cmd.extend([
                "-b:v",
                video_bitrate,
                "-maxrate",
                maxrate,
                "-bufsize",
                bufsize,
            ])
        else:
            cmd.extend(["-b:v", "0"])

        if is_nvenc_encoder:
            if NVENC_LOOKAHEAD:
                cmd.extend(["-rc-lookahead", NVENC_LOOKAHEAD])
            if NVENC_SPATIAL_AQ:
                cmd.extend(["-spatial_aq", NVENC_SPATIAL_AQ])
            if NVENC_TEMPORAL_AQ:
                cmd.extend(["-temporal_aq", NVENC_TEMPORAL_AQ])

        if scale_filter:
            cmd.extend(["-vf", scale_filter])

        cmd.extend([
            "-pix_fmt",
            "yuv420p",
        ])

        if audio_filter_arg:
            cmd.extend(["-af", audio_filter_arg])

        cmd.extend(audio_args)

        cmd.extend(subtitle_args or [])

        cmd.extend([
            "-map_metadata",
            "0",
        ])

        if is_mp4_output:
            cmd.extend(["-movflags", "+faststart"])

        cmd.append(str(temp_dst))

        return cmd

    logging.info("Encoding %s -> %s", src, dst)
    if total_frames:
        logging.info("Estimated frames to process: %s", total_frames)
    if not total_frames and total_duration:
        logging.info("Estimated duration to process: %.1f seconds", total_duration)
    if media_info.bitrate_kbps:
        logging.info("Source video bitrate ≈ %.0f kbps", media_info.bitrate_kbps)
    if target_bitrate_kbps:
        logging.info("Target video bitrate set to %s kbps", target_bitrate_kbps)

    if is_nvenc_encoder:
        attempts = (
            (True, True, "Hardware decode (GPU frames)"),
            (True, False, "Hardware decode (system frames)"),
            (False, False, "CPU decode"),
        )
    elif is_qsv_encoder:
        attempts = ((False, False, "Intel QSV"),)
    elif is_amf_encoder:
        attempts = ((False, False, "AMD AMF"),)
    else:
        attempts = ((False, False, "Software decode"),)
    success = False
    last_error: Optional[subprocess.CalledProcessError] = None
    for index, (use_hw_decode, force_hw_format, label) in enumerate(attempts):
        cmd = build_ffmpeg_cmd(use_hw_decode, force_hw_format)
        logging.debug("Attempting encode via %s", label)
        try:
            run_ffmpeg_with_progress(cmd, total_frames, total_duration)
            success = True
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if temp_dst.exists():
                try:
                    temp_dst.unlink()
                except OSError:
                    pass
            if index < len(attempts) - 1:
                logging.warning("%s path failed; attempting fallback.", label)

    fallback_encoder: Optional[str] = None
    fallback_label: Optional[str] = None
    if is_nvenc_encoder or is_qsv_encoder or is_amf_encoder:
        fallback_encoder = X264_ENCODER_NAME
        if is_nvenc_encoder:
            fallback_label = "NVENC"
        elif is_qsv_encoder:
            fallback_label = "QSV"
        else:
            fallback_label = "AMF"

    if not success and last_error is not None:
        stdout_output = getattr(last_error, "stdout", None)
        if stdout_output is None:
            stdout_output = getattr(last_error, "output", "")
        if (
            fallback_encoder
            and _allow_encoder_fallback
            and encoder_lower != fallback_encoder
        ):
            logging.warning(
                "%s encoding failed after all retries; falling back to %s.",
                fallback_label,
                fallback_encoder,
            )
            encode_video(
                src=src,
                dst=dst,
                overwrite=overwrite,
                encoder=fallback_encoder,
                output_extension=output_extension,
                quality=quality,
                _allow_encoder_fallback=False,
                _media_info_override=media_info,
            )
            return
        raise RuntimeError(
            f"ffmpeg failed for {src}:\nSTDOUT:\n{stdout_output}\nSTDERR:\n{last_error.stderr}"
        ) from last_error

    try:
        temp_dst.stat()
    except FileNotFoundError:
        raise RuntimeError(f"Expected output file missing for {src}") from None

    if overwrite:
        src.unlink()
        temp_dst.rename(dst)
        final_path = dst
    else:
        final_path = temp_dst

    final_size = final_path.stat().st_size

    if final_size >= original_size:
        logging.warning(
            "Output is larger than input (%s vs %s) for %s.",
            format_size(final_size),
            format_size(original_size),
            final_path,
        )
    else:
        savings = original_size - final_size
        logging.info(
            "Reduced %s by %s (from %s to %s).",
            final_path.name,
            format_size(savings),
            format_size(original_size),
            format_size(final_size),
        )


def process_videos(
    videos: Iterable[Path],
    base_input: Path,
    overwrite: bool,
    output_root: Optional[Path],
    encoder: str,
    output_extension: str,
    max_workers: int,
    quality: str,
) -> None:
    tasks = {}
    encode = partial(
        encode_video,
        overwrite=overwrite,
        encoder=encoder,
        output_extension=output_extension,
        quality=quality,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for src in videos:
            dst = build_output_path(src, base_input, overwrite, output_root, output_extension)
            future = pool.submit(encode, src=src, dst=dst)
            tasks[future] = src

        for future in as_completed(tasks):
            src = tasks[future]
            try:
                future.result()
                logging.info("Finished: %s", src)
            except Exception as exc:  # noqa: BLE001
                logging.error("Failed: %s\nReason: %s", src, exc)


def reduce_videos(config: ReducerConfig) -> None:
    base_input = config.input_path.expanduser().resolve()
    if not base_input.is_dir():
        raise ValueError(f"Input path is not a directory: {base_input}")

    ensure_ffmpeg_available()
    selection = select_encoder(config.encoder_backend, config.preferred_codec)
    quality_choice = (config.quality or "auto").lower()
    logging.info(
        "Using %s via %s backend (output extension %s).",
        selection.encoder,
        selection.backend.upper(),
        selection.output_extension,
    )
    logging.info("Quality preset set to '%s'.", quality_choice)

    if config.overwrite:
        output_root = None
        ignore_dir = None
        logging.info("Overwrite enabled; source files will be replaced in place.")
    else:
        root = config.output_root or (Path.cwd() / "output" / base_input.name)
        output_root = root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        ignore_dir = output_root
        logging.info("Overwrite disabled; writing converted files under %s.", output_root)

    logging.info("Beginning scan of %s.", base_input)
    videos = discover_videos(base_input, ignore_dir)
    if not videos:
        known_extensions = ", ".join(sorted(ext.upper().lstrip(".") for ext in VIDEO_EXTENSIONS))
        logging.info("No %s files found.", known_extensions)
        return

    logging.info("Found %d video(s) to process.", len(videos))
    process_videos(
        videos=videos,
        base_input=base_input,
        overwrite=config.overwrite,
        output_root=output_root,
        encoder=selection.encoder,
        output_extension=selection.output_extension,
        max_workers=config.max_workers,
        quality=quality_choice,
    )
