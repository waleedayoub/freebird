from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import requests

from freebird.config import MEDIA_DIR

logger = logging.getLogger(__name__)


def _event_dir(trace_id: str) -> Path:
    d = MEDIA_DIR / trace_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_image(url: str, trace_id: str) -> Path | None:
    if not url:
        return None
    dest = _event_dir(trace_id) / "keyshot.jpg"
    if dest.exists():
        return dest
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info("Downloaded keyshot for %s (%.1f KB)", trace_id, len(resp.content) / 1024)
        return dest
    except Exception:
        logger.exception("Failed to download image for %s", trace_id)
        return None


async def download_video(m3u8_url: str, trace_id: str) -> Path | None:
    if not m3u8_url:
        return None
    dest = _event_dir(trace_id) / "video.mp4"
    if dest.exists():
        return dest
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-i", m3u8_url,
            "-c", "copy",
            "-bsf:a", "aac_adtstoasc",
            str(dest),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("ffmpeg video download failed for %s: %s",
                         trace_id, stderr.decode()[-500:])
            dest.unlink(missing_ok=True)
            return None
        logger.info("Downloaded video for %s", trace_id)
        return dest
    except Exception:
        logger.exception("Failed to download video for %s", trace_id)
        dest.unlink(missing_ok=True)
        return None


async def extract_audio(video_path: Path, trace_id: str) -> Path | None:
    dest = _event_dir(trace_id) / "audio.wav"
    if dest.exists():
        return dest
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "1",
            str(dest),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("Audio extraction failed for %s: %s",
                         trace_id, stderr.decode()[-500:])
            dest.unlink(missing_ok=True)
            return None
        logger.info("Extracted audio for %s", trace_id)
        return dest
    except Exception:
        logger.exception("Failed to extract audio for %s", trace_id)
        dest.unlink(missing_ok=True)
        return None
