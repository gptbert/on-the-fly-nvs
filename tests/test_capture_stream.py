"""Capture lifecycle tests without a camera, OpenCV installation, or network."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "capture"
        self.cv2 = Mock()
        self.cap = self.cv2.VideoCapture.return_value
        self.cap.isOpened.return_value = True
        self.cap.read.return_value = (True, SimpleNamespace(shape=(48, 64, 3)))
        self.cv2.imwrite.return_value = True
        path = Path(__file__).resolve().parents[1] / "scripts" / "capture_stream.py"
        spec = importlib.util.spec_from_file_location("capture_under_test", path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"cv2": self.cv2}):
            spec.loader.exec_module(self.module)
        self.now = 0.0

    def clock(self):
        self.now += .03
        return self.now

    def run_capture(self, *extra, url="http://camera.invalid/video"):
        with patch("sys.argv", ["capture_stream", str(self.output), "--seconds", "2",
                                "--fps", "10", *extra]):
            with patch.dict(os.environ, {"STREAM_URL": url}):
                with patch.object(self.module.time, "monotonic", side_effect=self.clock):
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self.module.main()

    def report(self):
        return json.loads((self.output / "capture.json").read_text())

    def test_success_is_bounded_and_does_not_record_private_url(self):
        self.run_capture(url="http://user:secret@camera.invalid/video")
        report = self.report()
        self.assertTrue(report["complete"])
        self.assertGreaterEqual(len(report["frames"]), 8)
        self.assertLessEqual(len(report["frames"]), 21)
        self.assertNotIn("secret", json.dumps(report))
        self.assertEqual(report["frames"][0]["name"], "000000.jpg")
        self.assertEqual(report["clock"], "receiver_monotonic_not_camera_timestamp")
        self.cap.release.assert_called_once()

    def test_existing_capture_is_not_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_capture()
        self.cv2.VideoCapture.assert_not_called()

    def test_missing_url_fails_before_creating_files(self):
        with self.assertRaises(SystemExit):
            self.run_capture(url="")
        self.assertFalse(self.output.exists())

    def test_open_failure_is_marked_incomplete_and_released(self):
        self.cap.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Cannot open"):
            self.run_capture()
        self.assertFalse(self.report()["complete"])
        self.cap.release.assert_called_once()

    def test_disconnect_is_not_reported_as_completed_capture(self):
        self.cap.read.return_value = (False, None)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.run_capture()
        self.assertFalse(self.report()["complete"])
        self.cap.release.assert_called_once()

    def test_failed_image_write_is_not_added_to_manifest(self):
        self.cv2.imwrite.return_value = False
        with self.assertRaisesRegex(OSError, "save"):
            self.run_capture()
        self.assertEqual(self.report()["frames"], [])
        self.assertFalse(self.report()["complete"])
        self.cap.release.assert_called_once()

    def test_nonfinite_and_unbounded_recording_parameters_are_rejected(self):
        for flag, value in (("--seconds", "nan"), ("--seconds", "601"),
                            ("--fps", "inf"), ("--fps", "0")):
            with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit):
                self.run_capture(flag, value)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
