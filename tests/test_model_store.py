"""Model storage tests: no GPU, third-party packages, or network required."""

from contextlib import chdir
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from model_store import DEPTH_VARIANTS, PROJECT_ROOT, R3_REVISION, ModelStore, get_model_store


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.store = ModelStore(self.base / "models")
        self.addCleanup(get_model_store.cache_clear)
        get_model_store.cache_clear()

        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in ("MODELS_DIR", "DEPTH_MODEL_PATH", "HF_ENDPOINT"):
            os.environ.pop(name, None)

        legacy = patch("model_store.LEGACY_DEPTH_DIR", self.base / "old-cache")
        legacy.start()
        self.addCleanup(legacy.stop)
        download = patch("model_store.urlretrieve", side_effect=AssertionError("Unexpected download"))
        self.download = download.start()
        self.addCleanup(download.stop)

        self.torch = SimpleNamespace(hub=SimpleNamespace(set_dir=Mock(), load=Mock()),
                                     jit=SimpleNamespace(save=Mock()))
        self.lpips = SimpleNamespace(LPIPS=Mock())
        packages = patch.dict(sys.modules, {"torch": self.torch, "lpips": self.lpips})
        packages.start()
        self.addCleanup(packages.stop)

    def write(self, path, data=b"weights"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def depth_path(self, encoder="vitb"):
        return self.store.root / "depth_anything_v2" / f"depth_anything_v2_{encoder}.pth"

    def test_module_import_needs_no_model_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", "-c", "import model_store, sys; "
             "assert not {'torch', 'lpips', 'cv2', 'cupy'} & sys.modules.keys()"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_root_does_not_depend_on_working_directory(self):
        with chdir(self.base):
            self.assertEqual(ModelStore().root, PROJECT_ROOT / "models")

    def test_custom_root_and_environment_precedence(self):
        os.environ["MODELS_DIR"] = str(self.base / "shared")
        self.assertEqual(ModelStore().root, self.base / "shared")
        self.assertEqual(ModelStore(self.base / "explicit").root, self.base / "explicit")

    def test_relative_root_is_resolved_once(self):
        with chdir(self.base):
            store = ModelStore("relative-models")
        self.assertEqual(store.root, self.base / "relative-models")

    def test_user_home_is_expanded(self):
        self.assertEqual(ModelStore("~/nvs-models").root, Path.home() / "nvs-models")

    def test_construction_has_no_filesystem_or_network_side_effects(self):
        self.assertFalse(self.store.root.exists())
        self.download.assert_not_called()
        self.torch.hub.load.assert_not_called()

    def test_process_store_is_shared(self):
        os.environ["MODELS_DIR"] = str(self.store.root)
        first = get_model_store()
        os.environ["MODELS_DIR"] = str(self.base / "different")
        self.assertIs(get_model_store(), first)
        self.assertEqual(first.root, self.store.root)

    def test_existing_depth_is_reused(self):
        path = self.write(self.depth_path())
        self.assertEqual(self.store.depth_checkpoint("vitb"), path)
        self.download.assert_not_called()

    def test_r3_pinned_download_is_atomic_verified_and_reused(self):
        payload = b"test r3 weights"
        digest = hashlib.sha256(payload).hexdigest()
        os.environ['HF_ENDPOINT'] = 'https://models.example.test/'
        self.download.side_effect = lambda _url, path: path.write_bytes(payload)
        with patch.dict('model_store.R3_CHECKSUMS', {'r3': digest}):
            path = self.store.r3_checkpoint()
            self.assertEqual(path, self.store.root / 'r3' / R3_REVISION / 'r3.safetensors')
            self.assertEqual(self.download.call_args.args[0],
                             f'https://models.example.test/KevinXu02/R3/resolve/{R3_REVISION}/r3.safetensors')
            self.assertEqual(self.store.r3_checkpoint(), path)
            self.assertEqual(self.download.call_count, 1)
            self.assertEqual(list(path.parent.glob('.*')), [])

    def test_r3_bad_download_is_not_published(self):
        self.download.side_effect = lambda _url, path: path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.store.r3_checkpoint()
        self.assertEqual(list((self.store.root / 'r3' / R3_REVISION).iterdir()), [])

    def test_r3_corrupt_existing_checkpoint_is_preserved_and_reported(self):
        path = self.write(self.store.root / 'r3' / R3_REVISION / 'r3.safetensors', b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.store.r3_checkpoint()
        self.assertEqual(path.read_bytes(), b'corrupt')
        self.download.assert_not_called()

    def test_r3_unknown_variant_does_not_download(self):
        with self.assertRaisesRegex(ValueError, 'Unknown R3'):
            self.store.r3_checkpoint('../unknown')
        self.download.assert_not_called()

    def test_r3_loader_uses_bounded_inference_and_strict_cpu_weights(self):
        config_path = self.write(self.base / 'package' / 'configs' / 'r3-large.yaml')
        checkpoint = self.write(self.base / 'r3.safetensors')
        model = Mock()
        model.state_dict.return_value = {'da3.weight': 'expected'}
        model.eval.return_value = model
        model.requires_grad_.return_value = model
        model.to.return_value = model
        constructor = Mock(return_value=model)
        load_file = Mock(return_value={'net.da3.weight': 'loaded',
                                       'train_total_images': 1, 'train_total_samples': 2,
                                       'epoch_fraction': 0.5})
        packages = {'R3': SimpleNamespace(), 'R3.models': SimpleNamespace(),
                    'R3.models.r3': SimpleNamespace(R3=constructor),
                    'omegaconf': SimpleNamespace(OmegaConf=SimpleNamespace(load=Mock(return_value={}))),
                    'safetensors': SimpleNamespace(),
                    'safetensors.torch': SimpleNamespace(load_file=load_file)}
        with patch.dict(sys.modules, packages), \
             patch('model_store.files', return_value=config_path.parent.parent), \
             patch.object(self.store, 'r3_checkpoint', return_value=checkpoint):
            self.assertIs(self.store.load_r3(recent_frames=3, bank_size=8), model)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs['online_kv_cache_mode'], 'dynamic')
        self.assertEqual(kwargs['keyframe_max_keyframes'], 8)
        self.assertEqual(kwargs['online_recent_frames'], 3)
        self.assertFalse(kwargs['metric_scale_enabled'])
        self.assertFalse(kwargs['online_fallback_enabled'])
        model.load_state_dict.assert_called_once_with({'da3.weight': 'loaded'}, strict=True)
        load_file.assert_called_once_with(str(checkpoint), device='cpu')
        model.to.assert_called_once_with('cuda')

    def test_unknown_encoder_fails_without_download(self):
        with self.assertRaisesRegex(ValueError, "Unknown DEPTH_MODEL"):
            self.store.depth_checkpoint("invalid")
        self.download.assert_not_called()
        self.assertFalse(self.store.root.exists())

    def test_all_depth_variants_use_selected_endpoint(self):
        os.environ["HF_ENDPOINT"] = "https://models.example.test/"
        for encoder, size in DEPTH_VARIANTS.items():
            with self.subTest(encoder=encoder):
                expected = self.depth_path(encoder)

                def download(url, temporary):
                    self.assertEqual(url, f"https://models.example.test/depth-anything/"
                                     f"Depth-Anything-V2-{size}/resolve/main/{expected.name}")
                    self.assertEqual(temporary.parent, expected.parent)
                    self.assertNotEqual(temporary, expected)
                    self.assertFalse(expected.exists())
                    temporary.write_bytes(b"complete model")

                self.download.side_effect = download
                self.assertEqual(self.store.depth_checkpoint(encoder), expected)
                self.assertEqual(expected.read_bytes(), b"complete model")
                self.assertEqual(list(expected.parent.glob(".*")), [])

    def test_default_endpoint_is_huggingface(self):
        self.download.side_effect = lambda _url, path: path.write_bytes(b"model")
        self.store.depth_checkpoint("vitb")
        self.assertTrue(self.download.call_args.args[0].startswith("https://huggingface.co/"))

    def test_interrupted_depth_download_is_cleaned_up_and_can_retry(self):
        def fail(_url, path):
            path.write_bytes(b"partial")
            raise OSError("connection interrupted")

        self.download.side_effect = fail
        with self.assertRaisesRegex(OSError, "connection interrupted"):
            self.store.depth_checkpoint("vitb")
        self.assertFalse(self.depth_path().exists())
        self.assertEqual(list(self.depth_path().parent.iterdir()), [])
        self.download.side_effect = lambda _url, path: path.write_bytes(b"complete")
        self.assertEqual(self.store.depth_checkpoint("vitb").read_bytes(), b"complete")

    def test_empty_download_is_not_published(self):
        self.download.side_effect = lambda _url, path: path.touch()
        with self.assertRaisesRegex(ValueError, "Empty model file"):
            self.store.depth_checkpoint("vitb")
        self.assertEqual(list(self.depth_path().parent.iterdir()), [])

    def test_empty_cached_depth_is_downloaded_again(self):
        path = self.write(self.depth_path(), b"")
        self.download.side_effect = lambda _url, path: path.write_bytes(b"complete")
        self.assertEqual(self.store.depth_checkpoint("vitb"), path)
        self.assertEqual(path.read_bytes(), b"complete")

    def test_override_is_imported_and_source_is_preserved(self):
        source = self.write(self.base / "custom.pth", b"custom weights")
        self.write(self.depth_path(), b"old weights")
        os.environ["DEPTH_MODEL_PATH"] = str(source)
        destination = self.store.depth_checkpoint("vitb")
        self.assertEqual(destination, self.depth_path())
        self.assertEqual(destination.read_bytes(), b"custom weights")
        self.assertEqual(source.read_bytes(), b"custom weights")
        self.download.assert_not_called()

    def test_unchanged_import_is_not_copied_again(self):
        source = self.write(self.base / "custom.pth")
        os.environ["DEPTH_MODEL_PATH"] = str(source)
        self.store.depth_checkpoint("vitb")
        with patch("model_store.shutil.copy2") as copy:
            self.store.depth_checkpoint("vitb")
        copy.assert_not_called()

    def test_override_can_be_the_canonical_checkpoint(self):
        source = self.write(self.depth_path())
        os.environ["DEPTH_MODEL_PATH"] = str(source)
        self.assertEqual(self.store.depth_checkpoint("vitb"), source)
        self.download.assert_not_called()

    def test_bad_explicit_override_does_not_fall_back_to_cache_or_network(self):
        self.write(self.depth_path())
        for source in (self.base / "missing", self.write(self.base / "empty", b"")):
            with self.subTest(source=source):
                os.environ["DEPTH_MODEL_PATH"] = str(source)
                with self.assertRaisesRegex(FileNotFoundError, "missing or empty"):
                    self.store.depth_checkpoint("vitb")
        self.download.assert_not_called()

    def test_failed_import_preserves_existing_checkpoint(self):
        source = self.write(self.base / "custom.pth", b"new weights")
        self.write(self.depth_path(), b"old")
        os.environ["DEPTH_MODEL_PATH"] = str(source)

        def fail(_source, temporary):
            temporary.write_bytes(b"partial")
            raise OSError("disk full")

        with patch("model_store.shutil.copy2", side_effect=fail):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.store.depth_checkpoint("vitb")
        self.assertEqual(self.depth_path().read_bytes(), b"old")
        self.assertEqual(list(self.depth_path().parent.glob(".*")), [])

    def test_flat_legacy_depth_is_imported(self):
        source = self.write(self.store.root / self.depth_path().name)
        self.assertEqual(self.store.depth_checkpoint("vitb").read_bytes(), source.read_bytes())
        self.assertTrue(source.exists())
        self.download.assert_not_called()

    def test_old_docker_depth_is_imported(self):
        source = self.write(self.base / "old-cache" / self.depth_path().name)
        self.assertEqual(self.store.depth_checkpoint("vitb").read_bytes(), source.read_bytes())
        self.assertTrue(source.exists())
        self.download.assert_not_called()

    def test_all_library_caches_are_under_the_selected_root(self):
        for key in ("TORCH_HOME", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            os.environ[key] = "/old/cache"
        self.store.configure_caches()
        self.assertEqual(os.environ["TORCH_HOME"], str(self.store.root / "torch"))
        self.assertEqual(os.environ["HF_HOME"], str(self.store.root / "huggingface"))
        for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            self.assertEqual(os.environ[key], str(self.store.root / "huggingface" / "hub"))
        self.torch.hub.set_dir.assert_called_once_with(str(self.store.root / "torch" / "hub"))

    def test_xfeat_callers_share_storage_not_model_instances(self):
        def load(_repo, _model, **_kwargs):
            self.torch.hub.set_dir.assert_called_with(str(self.store.root / "torch" / "hub"))
            return object()

        self.torch.hub.load.side_effect = load
        detector_model = self.store.load_xfeat(top_k=1024)
        self.torch.hub.load.assert_called_with(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=1024)
        dense_model = self.store.load_xfeat(top_k=4096)
        self.torch.hub.load.assert_called_with(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096)
        self.assertIsNot(detector_model, dense_model)

    def test_lpips_imports_bundled_weights_and_uses_explicit_path(self):
        package = self.base / "lpips-package"
        self.write(package / "weights" / "v0.1" / "vgg.pth", b"linear weights")
        with patch("model_store.files", return_value=package) as resources:
            metric = self.store.load_lpips()
        resources.assert_called_once_with("lpips")
        destination = self.store.root / "lpips" / "vgg_v0.1.pth"
        self.assertEqual(destination.read_bytes(), b"linear weights")
        self.lpips.LPIPS.assert_called_once_with(
            net="vgg", version="0.1", model_path=str(destination), verbose=False)
        self.assertIs(metric, self.lpips.LPIPS.return_value)
        self.assertEqual(os.environ["TORCH_HOME"], str(self.store.root / "torch"))

    def test_lpips_reuses_imported_weights(self):
        self.write(self.store.root / "lpips" / "vgg_v0.1.pth")
        with patch("model_store.files") as resources:
            self.store.load_lpips()
        resources.assert_not_called()

    def test_jit_cache_names_remain_compatible(self):
        for name in ("xfeat_640_480_4096.pt", "dense_extractor_640_480.pt"):
            with self.subTest(name=name):
                self.assertEqual(self.store.jit_path(name), self.store.root / "cache" / name)

    def test_jit_save_publishes_complete_file(self):
        model = object()
        destination = self.store.jit_path("model.pt")

        def save(value, temporary):
            self.assertIs(value, model)
            self.assertFalse(destination.exists())
            self.assertEqual(Path(temporary).parent, destination.parent)
            Path(temporary).write_bytes(b"compiled model")

        self.torch.jit.save.side_effect = save
        self.assertEqual(self.store.save_jit(model, "model.pt"), destination)
        self.assertEqual(destination.read_bytes(), b"compiled model")
        self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_failed_jit_save_keeps_previous_cache(self):
        destination = self.write(self.store.jit_path("model.pt"), b"old model")

        def fail(_model, temporary):
            Path(temporary).write_bytes(b"partial")
            raise OSError("serialization failed")

        self.torch.jit.save.side_effect = fail
        with self.assertRaisesRegex(OSError, "serialization failed"):
            self.store.save_jit(object(), "model.pt")
        self.assertEqual(destination.read_bytes(), b"old model")
        self.assertEqual(list(destination.parent.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
