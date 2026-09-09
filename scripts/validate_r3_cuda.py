"""Validate pretrained streaming geometry on CUDA without the 3DGS backend.

Run from the repository root with python -m scripts.validate_r3_cuda.
Results measure this capture and configuration, not general reconstruction quality.
"""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from geometry.r3_provider import R3GeometryProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=504)
    parser.add_argument("--bank-size", type=int, default=8)
    parser.add_argument("--recent-frames", type=int, default=3)
    parser.add_argument("--downsampling", type=float, default=1.0)
    args = parser.parse_args()
    if args.limit < 1 or args.downsampling <= 0:
        parser.error("limit and downsampling must be positive")
    if not 112 <= args.resolution <= 1008 or args.bank_size < 2 or args.recent_frames < 1:
        parser.error("invalid R3 resolution or cache limits")
    paths = sorted(path for path in args.images.iterdir()
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[:args.limit]
    if not paths:
        parser.error("no input images")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    provider = None
    reference = None
    records = []
    initialization_seconds = None
    try:
        for index, path in enumerate(paths):
            array = cv2.imread(str(path))
            if array is None:
                raise ValueError(f"Cannot decode {path}")
            if args.downsampling != 1:
                array = cv2.resize(array, (0, 0), fx=1 / args.downsampling,
                                   fy=1 / args.downsampling, interpolation=cv2.INTER_AREA)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(array).permute(2, 0, 1).cuda().float() / 255
            height, width = image.shape[-2:]
            if provider is None:
                options = SimpleNamespace(
                    use_colmap_poses=False, enable_reboot=False, init_focal=-1,
                    init_fov=-1, r3_checkpoint="r3", r3_resolution=args.resolution,
                    r3_bank_size=args.bank_size, r3_recent_frames=args.recent_frames,
                    r3_min_confidence=1.02, r3_min_pose_confidence=1.0,
                )
                start = time.perf_counter()
                provider = R3GeometryProvider(width, height, options)
                torch.cuda.synchronize()
                initialization_seconds = time.perf_counter() - start
            info = {"name": path.name, "is_test": False}
            torch.cuda.synchronize()
            start = time.perf_counter()
            rgb = provider.observe(image, info, index)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            geometry = info["_r3_geometry"]
            for value in (rgb, geometry.Rt, geometry.idepth, geometry.pointmap):
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite geometry at {path.name}")
            valid = geometry.depth_confidence[0, 0].bool()
            if valid.any():
                torch.testing.assert_close(
                    geometry.pointmap[..., 2][valid], geometry.idepth[0, 0][valid].reciprocal(),
                )
            rotation = geometry.Rt[:3, :3]
            torch.testing.assert_close(rotation.T @ rotation, torch.eye(3), atol=1e-3, rtol=1e-3)
            admitted = provider.should_add_keyframe(info, reference, max(.05 * width, 30), False)
            if admitted:
                reference = {key: value for key, value in info.items() if key != "_r3_geometry"}
                reference.pop("mask", None)
            state = provider.backend.model.online_state
            if len(state.frame_order) > args.bank_size + args.recent_frames + 1:
                raise ValueError("R3 resident frame count exceeded the configured bound")
            record = {
                "frame": path.name, "seconds": elapsed,
                "valid_fraction": float(valid.float().mean()),
                "pose_confidence": geometry.metadata["pose_confidence"],
                "usable": bool(info["_r3_usable"]), "admitted": bool(admitted),
                "resident_frames": len(state.frame_order),
                "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
                "w2c": geometry.Rt.tolist(),
            }
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != "w2c"}), flush=True)
            del rgb, image, geometry, info
        seconds = [record["seconds"] for record in records[3:] or records]
        report = {
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "configuration": vars(args) | {"images": str(args.images),
                                                                         "report": str(args.report)},
            "initialization_seconds": initialization_seconds,
            "steady_p50_seconds": float(np.percentile(seconds, 50)),
            "steady_p95_seconds": float(np.percentile(seconds, 95)),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "usable_frames": sum(record["usable"] for record in records),
            "admitted_frames": sum(record["admitted"] for record in records),
            "frames": records,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "frames"}), flush=True)
        if not any(record["usable"] for record in records):
            raise RuntimeError("No frame passed the geometry validity gate; inspect the report")
    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    main()
