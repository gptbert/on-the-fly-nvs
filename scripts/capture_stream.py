"""Record a bounded reconstruction input from STREAM_URL without logging its URL."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--fps", type=float, default=2)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 600:
        parser.error("seconds must be between 1 and 600")
    if not math.isfinite(args.fps) or not .1 <= args.fps <= 10:
        parser.error("fps must be between 0.1 and 10")
    url = os.environ.get("STREAM_URL")
    if not url:
        parser.error("set STREAM_URL in the environment")
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output / "images"
    images.mkdir()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000,
    ])
    frames = []
    started = time.monotonic()
    next_sample = started
    error = None
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open the camera stream")
        shape = None
        print("Capture started; move the phone slowly around a static subject.", flush=True)
        while time.monotonic() - started < args.seconds:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera stream stopped before recording completed")
            now = time.monotonic()
            if now < next_sample:
                continue
            if shape is None:
                shape = frame.shape
            if frame.shape != shape:
                raise RuntimeError("Capture resolution changed during recording")
            name = f"{len(frames):06d}.jpg"
            if not cv2.imwrite(str(images / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError("Could not save a capture frame")
            frames.append({"name": name, "received_seconds": now - started,
                           "width": frame.shape[1], "height": frame.shape[0]})
            next_sample = now + 1 / args.fps
            if len(frames) % 10 == 0:
                print(f"Saved {len(frames)} frames ({now - started:.1f}s)", flush=True)
        if len(frames) < 8:
            raise RuntimeError("Too few frames for the default reconstruction bootstrap")
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        cap.release()
        manifest = {"requested_seconds": args.seconds, "requested_fps": args.fps,
                    "elapsed_seconds": time.monotonic() - started,
                    "clock": "receiver_monotonic_not_camera_timestamp",
                    "complete": error is None, "error": error, "frames": frames}
        (args.output / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"Capture complete: {len(frames)} frames in {args.output}", flush=True)


if __name__ == "__main__":
    main()
