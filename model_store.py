"""Shared storage and loading for pretrained models and derived JIT caches."""

from contextlib import contextmanager
from functools import lru_cache
from importlib.resources import as_file, files
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_DEPTH_DIR = Path("/cache/models")
DEPTH_VARIANTS = {"vits": "Small", "vitb": "Base", "vitl": "Large", "vitg": "Giant"}


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
