"""Shared storage and loading for pretrained models and derived JIT caches."""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
from importlib.resources import as_file, files
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_DEPTH_DIR = Path("/cache/models")
DEPTH_VARIANTS = {"vits": "Small", "vitb": "Base", "vitl": "Large", "vitg": "Giant"}
R3_REVISION = "c1f2aeccfa14d035a0e7b18f188253003a8417f0"
R3_CHECKSUMS = {
    "r3": "887ad839eb2725c683bde55e7d378b5b3b6b629d8363a08db9a5bce60c7570e1",
    "r3_long": "a5e14c7aa751450f8a9e2e78f1ac5e980060230af6dd0b6af86f1e6dd0859f7e",
}


@contextmanager
def _atomic_destination(destination):
    """Publish only complete files, keeping any existing file on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        if temporary.stat().st_size == 0:
            raise ValueError(f"Empty model file: {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class ModelStore:
    """One root for all model assets; construction performs no downloads."""

    def __init__(self, root=None):
        configured = root if root is not None else os.environ.get("MODELS_DIR")
        self.root = Path(configured or PROJECT_ROOT / "models").expanduser().resolve()

    def directory(self, name):
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def configure_caches(self):
        """Call before importing libraries that capture cache env vars on import."""
        torch_home = self.directory("torch")
        hf_home = self.directory("huggingface")
        os.environ["TORCH_HOME"] = str(torch_home)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")

        import torch

        # TORCH_HOME alone does not override a previous torch.hub.set_dir call.
        torch.hub.set_dir(str(torch_home / "hub"))

    @staticmethod
    def _has_file(path):
        return path.is_file() and path.stat().st_size > 0

    @staticmethod
    def _import_file(source, destination):
        source = Path(source).expanduser().resolve()
        if not ModelStore._has_file(source):
            raise FileNotFoundError(f"Model file is missing or empty: {source}")
        if source == destination:
            return destination
        if destination.is_file():
            src_stat, dst_stat = source.stat(), destination.stat()
            if (src_stat.st_size, src_stat.st_mtime_ns) == (dst_stat.st_size, dst_stat.st_mtime_ns):
                return destination
        with _atomic_destination(destination) as temporary:
            shutil.copy2(source, temporary)
        return destination

    def depth_checkpoint(self, encoder):
        if encoder not in DEPTH_VARIANTS:
            raise ValueError(f"Unknown DEPTH_MODEL {encoder!r}; choose {', '.join(DEPTH_VARIANTS)}")
        filename = f"depth_anything_v2_{encoder}.pth"
        destination = self.root / "depth_anything_v2" / filename

        # Legacy overrides are import sources, so inference still uses one root.
        override = os.environ.get("DEPTH_MODEL_PATH")
        if override:
            return self._import_file(override, destination)
        if self._has_file(destination):
            return destination
        for legacy in (self.root / filename, LEGACY_DEPTH_DIR / filename):
            if self._has_file(legacy):
                return self._import_file(legacy, destination)

        endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        url = (
            f"{endpoint}/depth-anything/Depth-Anything-V2-{DEPTH_VARIANTS[encoder]}"
            f"/resolve/main/{filename}"
        )
        print(f"Downloading Depth Anything V2 ({encoder}) to {destination}")
        with _atomic_destination(destination) as temporary:
            urlretrieve(url, temporary)
        return destination

    def jit_path(self, filename):
        return self.directory("cache") / filename

    def r3_checkpoint(self, variant="r3"):
        """Download immutable, checksum-verified R3 weights into the shared root."""
        if variant not in R3_CHECKSUMS:
            raise ValueError(f"Unknown R3 checkpoint: {variant}")
        destination = self.root / "r3" / R3_REVISION / f"{variant}.safetensors"

        def verify(path):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != R3_CHECKSUMS[variant]:
                raise ValueError(f"R3 checkpoint checksum mismatch: {path}")

        if self._has_file(destination):
            verify(destination)
            return destination
        endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        url = f"{endpoint}/KevinXu02/R3/resolve/{R3_REVISION}/{destination.name}"
        print(f"Downloading R3 ({variant}, CC BY-NC 4.0) to {destination}")
        with _atomic_destination(destination) as temporary:
            urlretrieve(url, temporary)
            verify(temporary)
        return destination

    def load_r3(self, variant="r3", recent_frames=3, bank_size=8, device="cuda"):
        if recent_frames < 1 or bank_size < 2:
            raise ValueError("R3 needs at least one recent frame and two bank keyframes.")
        self.configure_caches()
        try:
            from R3.models.r3 import R3
            from omegaconf import OmegaConf
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError(
                "R3 inference dependencies are missing. Install requirements-r3.txt "
                "and the pinned source in requirements-r3-source.txt with --no-deps."
            ) from error

        checkpoint = self.r3_checkpoint(variant)
        with as_file(files("R3").joinpath("configs", "r3-large.yaml")) as config:
            model = R3(
                da3_cfg=OmegaConf.load(config), online_mode=True,
                online_kv_cache_mode="dynamic", online_kv_backend="dense",
                online_recent_frames=recent_frames, bank_initial_frames=1,
                keyframe_max_keyframes=bank_size, keyframe_interval=10,
                online_verbose=False, online_fallback_enabled=False,
                online_finalize_pose_reconstruction=False, metric_scale_enabled=False,
                disable_segment_pgo=True, max_segment_frames=0,
            )
        # Load on CPU first, so weight loading does not double peak CUDA allocation.
        state = load_file(str(checkpoint), device="cpu")
        expected = model.state_dict()
        normalized = {}
        for key, value in state.items():
            if key in {"train_total_images", "train_total_samples", "epoch_fraction"}:
                continue
            for prefix in ("module.", "net."):
                key = key.removeprefix(prefix)
            if key.startswith("model."):
                key = "da3." + key.removeprefix("model.")
            if key not in expected and "da3." + key in expected:
                key = "da3." + key
            normalized[key] = value
        model.load_state_dict(normalized, strict=True)
        del state, normalized, expected
        return model.eval().requires_grad_(False).to(device)

    def save_jit(self, model, filename):
        import torch

        destination = self.jit_path(filename)
        with _atomic_destination(destination) as temporary:
            torch.jit.save(model, str(temporary))
        return destination

    def load_xfeat(self, top_k):
        self.configure_caches()
        import torch

        return torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k
        )

    def load_lpips(self):
        self.configure_caches()
        import lpips

        destination = self.root / "lpips" / "vgg_v0.1.pth"
        if not self._has_file(destination):
            # LPIPS ships its learned linear weights inside the installed package.
            resource = files("lpips").joinpath("weights", "v0.1", "vgg.pth")
            with as_file(resource) as source:
                self._import_file(source, destination)
        # The ImageNet VGG backbone is cached by torchvision in torch/hub/checkpoints.
        return lpips.LPIPS(net="vgg", version="0.1", model_path=str(destination), verbose=False)


@lru_cache(maxsize=1)
def get_model_store():
    """Resolve MODELS_DIR once per process, before the first model is loaded."""
    return ModelStore()
