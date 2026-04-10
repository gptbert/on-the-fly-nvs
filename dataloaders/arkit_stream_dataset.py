#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
ARKit Streaming Dataset
=======================
Receives synchronized (frame, pose) pairs from an iOS ARKit app via WebSocket.

Wire protocol (binary WebSocket message):
  [4 bytes] big-endian uint32  – byte length of the JSON header
  [N bytes] UTF-8 JSON header  – camera metadata
  [rest]    JPEG bytes         – the captured frame

JSON header fields:
  transform  float[16]  row-major 4×4 camera-to-world matrix (ARKit convention:
                         right-handed, Y-up, camera looks down –Z)
  fx         float      horizontal focal length in pixels (at native resolution)
  fy         float      vertical focal length in pixels (at native resolution)
  cx         float      principal-point X in pixels (at native resolution)
  cy         float      principal-point Y in pixels (at native resolution)

The server converts the ARKit C2W matrix to the OpenCV-convention W2C matrix
expected by the rest of the pipeline (flip Y & Z columns, then invert).
"""

import json
import queue
import struct
import threading

import cv2
import numpy as np
import torch
from websockets.sync.server import serve, ServerConnection


class ARKitStreamDataset:
    def __init__(self, host: str = "0.0.0.0", port: int = 9000, downsampling: float = 1.5):
        self.downsampling = downsampling
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._width: int | None = None
        self._height: int | None = None
        self._size_event = threading.Event()
        self.num_frames = 0

        t = threading.Thread(target=self._run_server, args=(host, port), daemon=True)
        t.start()
        print(f"ARKit WebSocket server listening on ws://{host}:{port}")
        print("Waiting for ARKit client to connect...")

    # ------------------------------------------------------------------
    # Internal server
    # ------------------------------------------------------------------

    def _run_server(self, host: str, port: int) -> None:
        with serve(self._handle_client, host, port, max_size=20 * 1024 * 1024) as srv:
            srv.serve_forever()

    def _handle_client(self, websocket: ServerConnection) -> None:
        print("ARKit client connected.")
        try:
            for raw in websocket:
                if not isinstance(raw, (bytes, bytearray)):
                    continue

                # Parse wire protocol
                if len(raw) < 4:
                    continue
                json_len = struct.unpack(">I", raw[:4])[0]
                if len(raw) < 4 + json_len:
                    continue

                meta = json.loads(raw[4 : 4 + json_len])
                jpeg_bytes = raw[4 + json_len :]

                # Decode JPEG
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                # Downsample
                ds = self.downsampling
                if ds > 0 and ds != 1.0:
                    frame_bgr = cv2.resize(
                        frame_bgr,
                        (0, 0),
                        fx=1.0 / ds,
                        fy=1.0 / ds,
                        interpolation=cv2.INTER_AREA,
                    )

                # Record image size on first frame
                if self._width is None:
                    h, w = frame_bgr.shape[:2]
                    self._height, self._width = h, w
                    self._size_event.set()

                # RGB tensor on CUDA
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = (
                    torch.from_numpy(frame_rgb)
                    .permute(2, 0, 1)
                    .cuda()
                    .float()
                    .div(255.0)
                )

                # ---- Pose conversion ----
                # ARKit delivers camera-to-world (C2W) in a right-handed,
                # Y-up coordinate system (camera looks down –Z).
                # The pipeline expects world-to-camera (W2C) in an OpenCV-
                # convention system (Y-down, Z-forward / camera looks down +Z).
                # Conversion: flip Y and Z columns of C2W, then invert → W2C.
                c2w = torch.tensor(
                    meta["transform"], dtype=torch.float32
                ).reshape(4, 4)
                c2w[:3, 1] *= -1  # flip Y column
                c2w[:3, 2] *= -1  # flip Z column
                Rt = torch.linalg.inv(c2w).cuda()  # W2C

                # Focal length, scaled for downsampling
                f = torch.tensor(meta["fx"] / ds, dtype=torch.float32).cuda()

                info = {
                    "is_test": False,
                    "Rt": Rt,
                    "focal": f,
                }

                # Keep only the latest frame (drop stale one)
                if not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put((image, info))

        except Exception as e:
            print(f"ARKit client disconnected: {e}")

    # ------------------------------------------------------------------
    # Dataset interface (mirrors StreamDataset)
    # ------------------------------------------------------------------

    def getnext(self) -> tuple[torch.Tensor, dict]:
        image, info = self._frame_queue.get(block=True)
        self.num_frames += 1
        return image, info

    def get_image_size(self) -> tuple[int, int]:
        """Blocks until the first frame arrives, then returns (height, width)."""
        self._size_event.wait()
        return self._height, self._width

    def __len__(self) -> int:
        return 100_000_000
