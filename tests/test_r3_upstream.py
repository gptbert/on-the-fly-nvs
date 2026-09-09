"""Optional real upstream API smoke test with random weights, on CPU.

Install the pinned R3 source and inference dependencies to enable this test.
This validates the API, not pretrained-model quality or CUDA performance.
"""

from contextlib import nullcontext
from importlib.resources import as_file, files
import importlib.util
import unittest
from unittest.mock import patch

import torch

from geometry.r3_provider import R3Backend


@unittest.skipUnless(importlib.util.find_spec('R3'), 'Pinned R3 source not installed')
class UpstreamContractTests(unittest.TestCase):
    def test_real_model_single_frame_output_and_cache_lifecycle(self):
        from R3.models.r3 import R3
        from R3.utils.pose_enc import pose_encoding_to_extri_intri
        from omegaconf import OmegaConf

        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        torch.set_num_threads(2)
        with as_file(files('R3').joinpath('configs', 'r3-large.yaml')) as config:
            model = R3(OmegaConf.load(config), online_kv_cache_mode='dynamic',
                       online_recent_frames=1, keyframe_max_keyframes=2,
                       bank_initial_frames=1, keyframe_interval=1, online_verbose=False,
                       metric_scale_enabled=False, online_fallback_enabled=False).eval()
        model.requires_grad_(False)
        backend = R3Backend.__new__(R3Backend)
        backend.model = model
        backend.decode_pose = pose_encoding_to_extri_intri
        backend.dtype = torch.bfloat16
        for index in range(6):
            with patch('torch.autocast', return_value=nullcontext()):
                result = backend.infer(torch.rand(3, 112, 112))
            self.assertEqual(result['Rt'].shape, (3, 4))
            self.assertEqual(result['K'].shape, (3, 3))
            self.assertEqual(result['depth'].shape, (112, 112))
            self.assertEqual(result['confidence'].shape, (112, 112))
            self.assertTrue(torch.isfinite(result['Rt']).all())
            self.assertTrue((result['depth'] > 0).all())
            self.assertEqual(model.online_state.frame_count, index + 1)
            self.assertLessEqual(len(model.online_state.cache_frame_ids), 4)
            self.assertLessEqual(len(model._persistent_post_scores), 4)
            for cache in model.online_state.kv_cache_list:
                if cache is not None and cache[0] is not None:
                    self.assertLessEqual(cache[0].shape[2], model.online_state.tokens_per_frame * 5)
        model.clear_online_state()
        self.assertIsNone(model.online_state)


if __name__ == '__main__':
    unittest.main()
