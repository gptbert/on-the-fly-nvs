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

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from poses.matcher import Matcher
    from poses.triangulator import Triangulator


def fov2focal(fov: float, pixels: int) -> float:
    return pixels / (2 * math.tan(fov / 2))


@dataclass
class GeometryFrame:
    """Geometry priors for one frame, independent from their source model."""

    Rt: torch.Tensor | None = None
    focal: torch.Tensor | float | None = None
    idepth: torch.Tensor | None = None
    depth_confidence: torch.Tensor | None = None
    pointmap: torch.Tensor | None = None
    tracks: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_pose(self) -> bool:
        return self.Rt is not None

    def has_depth(self) -> bool:
        return self.idepth is not None and self.depth_confidence is not None


class GeometryProvider:
    """
    Base interface for geometry sources.

    Implementations can wrap classic matching + BA, MASt3R-SLAM, VGGT, CUT3R,
    ARKit/ARCore exports, or any future source that can provide pose, depth,
    point maps, or tracks.
    """

    def __call__(self, image: torch.Tensor, info: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self.estimate_frame_geometry(image, info or {})
        if not geometry.has_depth():
            raise NotImplementedError("GeometryProvider did not provide inverse depth and confidence.")
        return geometry.idepth, geometry.depth_confidence

    def estimate_frame_geometry(self, image: torch.Tensor, info: dict | None = None) -> GeometryFrame:
        raise NotImplementedError

    def initialize_bootstrap(
        self,
        desc_kpts_list,
        frame_dicts: list[dict] | None = None,
        rebooting: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | float, torch.Tensor | float]:
        raise NotImplementedError

    def initialize_incremental(
        self,
        keyframes,
        curr_desc_kpts,
        index: int,
        is_test: bool,
        curr_img: torch.Tensor,
        info: dict | None = None,
    ) -> GeometryFrame:
        raise NotImplementedError


class DefaultGeometryProvider(GeometryProvider):
    """Compatibility provider that preserves the original project behavior."""

    def __init__(
        self,
        width: int,
        height: int,
        triangulator: Triangulator,
        matcher: Matcher,
        max_pnp_error: float,
        args,
    ):
        from poses.pose_initializer import PoseInitializer
        from scene.mono_depth import MonoDepthEstimator

        self.pose_initializer = PoseInitializer(
            width, height, triangulator, matcher, max_pnp_error, args
        )
        self.depth_estimator = MonoDepthEstimator(width, height)
        self.use_colmap_poses = args.use_colmap_poses

    @property
    def f_init(self):
        return self.pose_initializer.f_init

    @property
    def f(self):
        return self.pose_initializer.f

    def estimate_frame_geometry(self, image: torch.Tensor, info: dict | None = None) -> GeometryFrame:
        idepth, confidence = self.depth_estimator(image)
        return GeometryFrame(idepth=idepth, depth_confidence=confidence)

    def initialize_bootstrap(
        self,
        desc_kpts_list,
        frame_dicts: list[dict] | None = None,
        rebooting: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | float, torch.Tensor | float]:
        Rts, focal, residual = self.pose_initializer.initialize_bootstrap(
            desc_kpts_list, rebooting=rebooting
        )
        if self.use_colmap_poses and frame_dicts is not None:
            Rts = torch.stack([frame_dict["info"]["Rt"] for frame_dict in frame_dicts])
            focal = frame_dicts[0]["info"]["focal"]
        return Rts, focal, residual

    def initialize_incremental(
        self,
        keyframes,
        curr_desc_kpts,
        index: int,
        is_test: bool,
        curr_img: torch.Tensor,
        info: dict | None = None,
    ) -> GeometryFrame:
        Rt = self.pose_initializer.initialize_incremental(
            keyframes, curr_desc_kpts, index, is_test, curr_img
        )
        if Rt is not None and self.use_colmap_poses and info is not None:
            Rt = info["Rt"]
        return GeometryFrame(Rt=Rt)


class ExternalGeometryProvider(GeometryProvider):
    """
    Adapter for external geometry attached to the dataset info dict.

    This is the integration point for MASt3R-SLAM, VGGT, CUT3R, ARKit, or ARCore
    pipelines while the repo still uses SceneModel for 3DGS optimization.
    Missing fields fall back to the default provider.
    """

    def __init__(
        self,
        width: int,
        height: int,
        triangulator: Triangulator,
        matcher: Matcher,
        max_pnp_error: float,
        args,
    ):
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.matcher = matcher
        self.max_pnp_error = max_pnp_error
        self.args = args
        self._fallback_provider = None
        if args.init_focal > 0:
            self._f_init = args.init_focal
        elif args.init_fov > 0:
            self._f_init = fov2focal(args.init_fov * math.pi / 180, width)
        else:
            self._f_init = 0.7 * width

    @property
    def fallback_provider(self) -> DefaultGeometryProvider:
        if self._fallback_provider is None:
            self._fallback_provider = DefaultGeometryProvider(
                self.width,
                self.height,
                self.triangulator,
                self.matcher,
                self.max_pnp_error,
                self.args,
            )
        return self._fallback_provider

    @property
    def f_init(self):
        return self.fallback_provider.f_init if self._fallback_provider is not None else self._f_init

    def _from_info(self, info: dict | None) -> GeometryFrame:
        info = info or {}
        return GeometryFrame(
            Rt=info.get("geometry_Rt"),
            focal=info.get("geometry_focal"),
            idepth=info.get("geometry_idepth"),
            depth_confidence=info.get("geometry_depth_confidence"),
            pointmap=info.get("geometry_pointmap"),
            tracks=info.get("geometry_tracks"),
            metadata=info.get("geometry_metadata", {}),
        )

    def estimate_frame_geometry(self, image: torch.Tensor, info: dict | None = None) -> GeometryFrame:
        external = self._from_info(info)
        if not external.has_depth():
            fallback = self.fallback_provider.estimate_frame_geometry(image, info)
            if external.idepth is None:
                external.idepth = fallback.idepth
            if external.depth_confidence is None:
                external.depth_confidence = fallback.depth_confidence
        return external

    def initialize_bootstrap(
        self,
        desc_kpts_list,
        frame_dicts: list[dict] | None = None,
        rebooting: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | float, torch.Tensor | float]:
        if frame_dicts and all(self._from_info(frame_dict["info"]).has_pose() for frame_dict in frame_dicts):
            frames = [self._from_info(frame_dict["info"]) for frame_dict in frame_dicts]
            Rts = torch.stack([frame.Rt for frame in frames])
            focal = next((frame.focal for frame in frames if frame.focal is not None), self.f_init)
            return Rts, focal, torch.tensor(0.0, device=Rts.device)
        return self.fallback_provider.initialize_bootstrap(desc_kpts_list, frame_dicts, rebooting)

    def initialize_incremental(
        self,
        keyframes,
        curr_desc_kpts,
        index: int,
        is_test: bool,
        curr_img: torch.Tensor,
        info: dict | None = None,
    ) -> GeometryFrame:
        external = self._from_info(info)
        if external.has_pose():
            return external
        return self.fallback_provider.initialize_incremental(
            keyframes, curr_desc_kpts, index, is_test, curr_img, info
        )


def make_geometry_provider(
    name: str,
    width: int,
    height: int,
    triangulator: Triangulator,
    matcher: Matcher,
    max_pnp_error: float,
    args,
) -> GeometryProvider:
    if name == "default":
        return DefaultGeometryProvider(width, height, triangulator, matcher, max_pnp_error, args)
    if name in {"external", "arkit", "arcore", "mast3r", "mast3r_slam", "vggt", "cut3r"}:
        return ExternalGeometryProvider(width, height, triangulator, matcher, max_pnp_error, args)
    raise ValueError(f"Unknown geometry provider: {name}")
