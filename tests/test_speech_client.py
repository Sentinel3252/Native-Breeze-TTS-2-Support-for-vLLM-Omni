# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Breeze TTS Omni project
"""Test transport and artifact handling with a local HTTP server, not a model."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import wave

from examples.speech_client import synthesize


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x01\x00" * 2400)
    return buffer.getvalue()


class SpeechClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                self.server.received = {
                    "path": self.path, "payload": json.loads(self.rfile.read(length)),
                    "authorization": self.headers.get("Authorization"),
                }
                self.send_response(self.server.status)
                self.send_header("Content-Type", self.server.content_type)
                self.send_header("Content-Length", str(len(self.server.body) + self.server.extra_length))
                self.end_headers()
                self.wfile.write(self.server.body)

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "speech.wav"
        self.server.status = 200
        self.server.body = wav_bytes()
        self.server.content_type = "audio/wav"
        self.server.extra_length = 0
        self.server.received = None

    def request(self, **kwargs):
        settings = {"base_url": self.base_url, "model": "checkpoint", "text": "你好，Breeze。",
                    "output": self.output, "timeout": 3}
        settings.update(kwargs)
        return synthesize(**settings)

    def assert_no_artifact(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob("*.part")), [])

    def test_wav_request_and_audio_metadata(self):
        report = self.request(api_key="test-token")
        self.assertEqual(self.output.read_bytes(), self.server.body)
        self.assertEqual(self.server.received["path"], "/v1/audio/speech")
        self.assertEqual(self.server.received["payload"]["input"], "你好，Breeze。")
        self.assertEqual(self.server.received["authorization"], "Bearer test-token")
        self.assertEqual(report["sample_rate"], 24000)
        self.assertAlmostEqual(report["audio_duration_s"], 0.1)
        self.assertNotIn("test-token", json.dumps(report))
        self.assertEqual(list(self.output.parent.glob("*.part")), [])

    def test_streaming_selects_raw_audio_and_preserves_pcm(self):
        self.output = self.output.with_suffix(".pcm")
        self.server.body = b"\x01\x00" * 128
        self.server.content_type = "audio/pcm"
        report = self.request(stream=True)
        self.assertEqual(self.output.read_bytes(), self.server.body)
        self.assertEqual(self.server.received["payload"]["stream_format"], "audio")
        self.assertEqual(report["format"], "pcm")
        self.assertNotIn("sample_rate", report)

    def test_existing_output_is_not_overwritten_or_requested(self):
        self.output.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            self.request()
        self.assertEqual(self.output.read_bytes(), b"original")
        self.assertIsNone(self.server.received)

    def test_http_error_preserves_existing_output_even_with_force(self):
        self.output.write_bytes(b"original")
        self.server.status = 400
        self.server.body = b"private diagnostic"
        with self.assertRaisesRegex(ValueError, "HTTP 400") as error:
            self.request(force=True)
        self.assertNotIn("private diagnostic", str(error.exception))
        self.assertEqual(self.output.read_bytes(), b"original")

    def test_force_replaces_only_valid_audio(self):
        self.output.write_bytes(b"original")
        self.request(force=True)
        self.assertEqual(self.output.read_bytes(), self.server.body)

    def test_non_audio_response_rejected(self):
        self.server.content_type = "application/json"
        self.server.body = b'{"error":"bad request"}'
        with self.assertRaisesRegex(ValueError, "Expected audio"):
            self.request()
        self.assert_no_artifact()

    def test_empty_audio_rejected(self):
        self.server.body = b""
        with self.assertRaisesRegex(ValueError, "empty audio"):
            self.request()
        self.assert_no_artifact()

    def test_truncated_http_response_rejected(self):
        self.server.extra_length = 20
        with self.assertRaisesRegex(ValueError, "Truncated HTTP"):
            self.request()
        self.assert_no_artifact()

    def test_truncated_wav_data_rejected(self):
        self.server.body = wav_bytes()[:-10]
        with self.assertRaisesRegex(ValueError, "declared frame count"):
            self.request()
        self.assert_no_artifact()

    def test_partial_pcm_sample_rejected(self):
        self.output = self.output.with_suffix(".pcm")
        self.server.body = b"\x01"
        self.server.content_type = "audio/pcm"
        with self.assertRaisesRegex(ValueError, "incomplete 16-bit"):
            self.request(stream=True)
        self.assert_no_artifact()

    def test_invalid_text_and_stream_extension_fail_before_request(self):
        with self.assertRaises(ValueError):
            self.request(text=" ")
        with self.assertRaises(ValueError):
            self.request(stream=True)
        self.assertIsNone(self.server.received)


if __name__ == "__main__":
    unittest.main()
