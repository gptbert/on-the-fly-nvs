"""Native R3 streaming geometry, with a fixed virtual camera for SceneModel."""

from dataclasses import replace
import math

import torch
import torch.nn.functional as F

from geometry.provider import GeometryFrame, GeometryProvider, fov2focal
from model_store import get_model_store


class R3Backend:
    """Small boundary around the pinned upstream API; no sequence preloading."""

    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("R3 reconstruction requires a CUDA GPU.")
        if tuple(int(part) for part in torch.__version__.split('.')[:2]) < (2, 5):
            raise RuntimeError("R3 requires PyTorch >=2.5; rebuild the CUDA environment.")
        self.model = get_model_store().load_r3(
            args.r3_checkpoint, args.r3_recent_frames, args.r3_bank_size,
        )
        from R3.utils.pose_enc import pose_encoding_to_extri_intri

        self.decode_pose = pose_encoding_to_extri_intri
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model.clear_online_state()

    @torch.no_grad()
    def infer(self, image):
        # forward_online_step expects ImageNet normalization, unlike model.forward.
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        normalized = ((image - mean) / std)[None, None]
        with torch.autocast("cuda", dtype=self.dtype):
            prediction = self.model.forward_online_step(normalized, use_ray_pose=False)
        state = self.model.online_state
        frame_id = state.frame_count - 1
        pose, intrinsics = self.decode_pose(prediction["pose_enc"].float(), image.shape[-2:])
        result = {
            "Rt": pose[0, -1], "K": intrinsics[0, -1],
            "depth": prediction["depth"][0, -1, ..., 0].float(),
            "confidence": prediction["depth_conf"][0, -1].float(),
            "pose_confidence": state.frame_post_scores.get(frame_id, 0.0),
        }
        # Upstream retains diagnostic histories forever for offline export. This
        # adapter exports the scene instead, so keep only resident-frame scores.
        keep = set(state.frame_order)
        for scores in (state.frame_post_scores, state.frame_score_history,
                       self.model._persistent_post_scores):
            for old_id in list(scores):
                if old_id not in keep:
                    del scores[old_id]
        return result

    def close(self):
        self.model.clear_online_state()
        self.model = None
        torch.cuda.empty_cache()


def rectify_geometry(image, depth, confidence, intrinsics, focal, threshold, mask=None):
    """Resample RGB and Z-depth along identical rays into a centered pinhole camera.

    intrinsics belongs to the resized model input. Pixel-center transforms match
    interpolate/grid_sample with align_corners=False, including nonuniform resize.
    """
    height, width = image.shape[-2:]
    model_h, model_w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=torch.float32),
        torch.arange(width, device=image.device, dtype=torch.float32), indexing="ij",
    )
    rays = torch.stack(((xs - (width - 1) / 2) / focal,
                        (ys - (height - 1) / 2) / focal, torch.ones_like(xs)), dim=-1)
    projected = rays @ intrinsics.T
    uv = projected[..., :2] / projected[..., 2:]
    grid = 2 * (uv + 0.5) / image.new_tensor([model_w, model_h]) - 1
    grid = grid[None]
    # One normalized grid addresses both the resized depth and original RGB.
    rgb = F.grid_sample(image[None], grid, align_corners=False, padding_mode="border")[0]
    valid_source = torch.isfinite(depth) & (depth > 1e-6)
    valid_source &= torch.isfinite(confidence) & (confidence >= threshold)
    clean_depth = torch.where(valid_source, depth, torch.zeros_like(depth))
    z = F.grid_sample(clean_depth[None, None], grid, align_corners=False)[0, 0]
    coverage = F.grid_sample(valid_source[None, None].float(), grid,
                             align_corners=False)[0, 0]
    valid = (coverage > 1 - 1e-5) & (z > 1e-6)
    # Exclude borders that lack full source-image support after resampling.
    image_uv = (uv + 0.5) * image.new_tensor([width / model_w, height / model_h]) - 0.5
    valid &= (image_uv[..., 0] >= 0) & (image_uv[..., 0] <= width - 1)
    valid &= (image_uv[..., 1] >= 0) & (image_uv[..., 1] <= height - 1)
    if mask is not None:
        source_mask = mask.to(image).reshape(1, 1, height, width)
        valid &= F.grid_sample(source_mask, grid, align_corners=False)[0, 0] > 1 - 1e-5
    z = torch.where(valid, z, torch.zeros_like(z))
    idepth = torch.where(valid, z.clamp_min(1e-6).reciprocal(), torch.zeros_like(z))
    return rgb, idepth[None, None], valid[None, None].float(), rays * z[..., None]


class R3GeometryProvider(GeometryProvider):
    """Observe once per captured frame; accepted frames retain CPU geometry only."""

    requires_observation = True

    def __init__(self, width, height, args, backend=None):
        if args.use_colmap_poses or args.enable_reboot:
            raise ValueError("R3 cannot mix COLMAP/reboot poses with its streaming gauge.")
        self.width, self.height = width, height
        self.args = args
        self.backend = backend if backend is not None else R3Backend(args)
        scale = args.r3_resolution / max(height, width)
        self.input_size = tuple(max(14, round(d * scale / 14) * 14) for d in (height, width))
        self._f = args.init_focal if args.init_focal > 0 else None
        if self._f is None and args.init_fov > 0:
            self._f = fov2focal(math.radians(args.init_fov), width)
        self.last_frame_id = -1

    @property
    def f_init(self):
        return self._f if self._f is not None else 0.7 * self.width

    @torch.no_grad()
    def observe(self, image, info, frame_id):
        if frame_id <= self.last_frame_id:
            raise ValueError("R3 frames must be observed exactly once in increasing order.")
        if image.shape != (3, self.height, self.width):
            raise ValueError("Capture resolution changed; start a new reconstruction.")
        resized = F.interpolate(image[None], self.input_size, mode="bilinear",
                                align_corners=False, antialias=True)[0]
        output = self.backend.infer(resized)
        self.last_frame_id = frame_id
        K = output["K"].to(image).float()
        Rt = torch.eye(4, device=image.device)
        Rt[:3] = output["Rt"].to(image).float()[:3]
        if not torch.isfinite(K).all() or not torch.isfinite(Rt).all():
            raise RuntimeError("R3 returned non-finite camera parameters; refusing a corrupt scene.")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise RuntimeError("R3 returned invalid focal lengths.")
        if self._f is None:
            self._f = max(float(K[0, 0]) * self.width / self.input_size[1],
                          float(K[1, 1]) * self.height / self.input_size[0])
        rgb, idepth, confidence, pointmap = rectify_geometry(
            image, output["depth"], output["confidence"], K, self._f,
            self.args.r3_min_confidence, info.get("mask"),
        )
        valid = confidence.bool()
        pose_confidence = float(output["pose_confidence"])
        usable = math.isfinite(pose_confidence) and pose_confidence >= self.args.r3_min_pose_confidence
        usable &= float(valid.float().mean()) >= 0.1
        median_depth = float(idepth[valid].reciprocal().median()) if valid.any() else 0.0
        info["_r3_raw_Rt"] = Rt.cpu()
        info["_r3_median_depth"] = median_depth
        info["_r3_usable"] = usable
        info["mask"] = confidence[0].cpu()
        info["_r3_geometry"] = GeometryFrame(
            Rt=Rt.cpu(), focal=self._f, idepth=idepth.cpu(),
            depth_confidence=confidence.cpu(), pointmap=pointmap.cpu(),
            metadata={"provider": "r3", "frame_id": frame_id,
                      "pointmap_space": "camera", "scale_consistent": True,
                      "pose_confidence": pose_confidence},
        )
        return rgb

    def should_add_keyframe(self, info, reference_info, min_displacement, fallback):
        if not info.get("_r3_usable", False):
            return False
        if reference_info is None or info.get("is_test", False):
            return True
        relative = info["_r3_raw_Rt"] @ torch.linalg.inv(reference_info["_r3_raw_Rt"])
        angle = torch.acos(((torch.trace(relative[:3, :3]) - 1) / 2).clamp(-1, 1))
        distance = torch.linalg.vector_norm(relative[:3, 3])
        scene_depth = max(reference_info["_r3_median_depth"], 1e-6)
        displacement = self._f * (float(angle) + float(distance) / scene_depth)
        return displacement > min_displacement

    def estimate_frame_geometry(self, image, info=None):
        if info is None or "_r3_geometry" not in info:
            raise RuntimeError("R3 geometry must be observed before initialization.")
        geometry = info["_r3_geometry"]
        return replace(geometry, Rt=geometry.Rt.to(image.device),
                       focal=torch.tensor(geometry.focal, device=image.device),
                       idepth=geometry.idepth.to(image.device),
                       depth_confidence=geometry.depth_confidence.to(image.device),
                       pointmap=geometry.pointmap.to(image.device))

    def initialize_bootstrap(self, desc_kpts_list, frame_dicts=None, rebooting=False):
        if rebooting or not frame_dicts:
            raise ValueError("R3 bootstrap requires observed frames, without legacy reboot.")
        geometries = [frame["info"]["_r3_geometry"] for frame in frame_dicts]
        device = frame_dicts[0]["image"].device
        return (torch.stack([g.Rt for g in geometries]).to(device),
                torch.tensor(geometries[0].focal, device=device), 0.0)

    def initialize_incremental(self, keyframes, curr_desc_kpts, index, is_test, curr_img, info=None):
        geometry = self.estimate_frame_geometry(curr_img, info)
        references = [kf for kf in keyframes if "_r3_raw_Rt" in kf.info and not kf.is_test]
        if references:
            reference = max(references, key=lambda kf: kf.index)
            raw_reference = reference.info["_r3_raw_Rt"].to(curr_img.device)
            # Transfer the latest backend pose correction, without rescaling Z-depth.
            geometry.Rt = geometry.Rt @ torch.linalg.inv(raw_reference) @ reference.get_Rt().detach()
        return geometry

    def close(self):
        self.backend.close()
