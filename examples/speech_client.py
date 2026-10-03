# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Breeze TTS Omni project
"""Request WAV or streaming PCM from an already-running speech server."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave


def synthesize(
    *, base_url: str, model: str, text: str, output: Path,
    stream: bool = False, timeout: float = 120.0, max_new_tokens: int = 256,
    api_key: str | None = None, force: bool = False,
) -> dict:
    """Save validated audio; leave existing files intact if the request fails."""
    if not text.strip() or not model.strip():
        raise ValueError("Text and model must be nonempty")
    if not math.isfinite(timeout) or timeout <= 0 or max_new_tokens <= 0:
        raise ValueError("Timeout and max_new_tokens must be positive")
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Base URL must be an HTTP(S) server URL without query or fragment")
    if parsed.username or parsed.password:
        raise ValueError("Use BREEZE_API_KEY instead of credentials in the URL")
    output = Path(output)
    extension = ".pcm" if stream else ".wav"
    if output.suffix.lower() != extension:
        raise ValueError(f"Output must use {extension} for this response format")
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}; use --force to replace it")
    payload = {
        "model": model, "input": text, "response_format": "pcm" if stream else "wav",
        "max_new_tokens": max_new_tokens, "stream": stream,
    }
    if stream:
        payload["stream_format"] = "audio"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/audio/speech",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    started = time.perf_counter()
    first_byte = None
    received = 0
    try:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get_content_type()
                allowed = {"audio/pcm", "audio/raw", "application/octet-stream"} if stream else {
                    "audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream",
                }
                if content_type not in allowed:
                    raise ValueError(f"Expected audio; server returned {content_type}")
                expected_length = response.headers.get("Content-Length")
                with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".part", delete=False) as handle:
                    temp_path = Path(handle.name)
                    while chunk := response.read1(65536):
                        if first_byte is None:
                            first_byte = time.perf_counter() - started
                        received += len(chunk)
                        handle.write(chunk)
                if expected_length is not None and received != int(expected_length):
                    raise ValueError("Truncated HTTP audio response")
        except urllib.error.HTTPError as exc:
            # Do not echo response bodies: they may contain private input or server paths.
            exc.close()
            raise ValueError(f"Speech endpoint returned HTTP {exc.code}; inspect server logs") from exc
        elapsed = time.perf_counter() - started
        if received == 0:
            raise ValueError("Server returned empty audio")
        report = {
            "output": str(output), "format": "pcm" if stream else "wav", "bytes": received,
            "elapsed_s": round(elapsed, 6), "first_body_byte_s": round(first_byte, 6),
        }
        if stream:
            if received % 2:
                raise ValueError("PCM response ends with an incomplete 16-bit sample")
        else:
            with wave.open(str(temp_path), "rb") as audio:
                frames = audio.getnframes()
                channels = audio.getnchannels()
                rate = audio.getframerate()
                width = audio.getsampwidth()
                if not frames or not rate or not channels or audio.getcomptype() != "NONE":
                    raise ValueError("WAV response contains no uncompressed audio frames")
                frame_bytes = 0
                while data := audio.readframes(65536):
                    frame_bytes += len(data)
                if frame_bytes != frames * channels * width:
                    raise ValueError("WAV data is shorter than its declared frame count")
                duration = frames / rate
                report.update(sample_rate=rate, channels=channels, audio_duration_s=round(duration, 6),
                              rtf=round(elapsed / duration, 6))
        if force:
            os.replace(temp_path, output)
            temp_path = None
        else:
            # Exclusive creation also protects files created while the request was in flight.
            with output.open("xb") as final:
                try:
                    with temp_path.open("rb") as source:
                        shutil.copyfileobj(source, final)
                except BaseException:
                    final.close()
                    output.unlink()
                    raise
        return report
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server origin (without /v1)")
    parser.add_argument("--model", required=True, help="Model name advertised by the server")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stream", action="store_true", help="Save raw PCM with stream_format=audio")
    parser.add_argument("--timeout", type=float, default=120.0, help="Socket timeout in seconds")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--force", action="store_true", help="Replace output only after validating new audio")
    args = parser.parse_args()
    try:
        report = synthesize(**vars(args), api_key=os.environ.get("BREEZE_API_KEY"))
    except (ValueError, OSError, EOFError, wave.Error, http.client.HTTPException, urllib.error.URLError) as exc:
        print(f"Speech request failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
