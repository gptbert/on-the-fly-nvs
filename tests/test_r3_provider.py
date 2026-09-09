"""CPU-only adapter contracts; no pretrained weights or CUDA extensions needed."""

from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from args import get_args
from geometry.provider import GeometryProvider, make_geometry_provider
from geometry.r3_provider import R3Backend, R3GeometryProvider, rectify_geometry


def options(**overrides):
    values = dict(use_colmap_poses=False, enable_reboot=False, init_focal=-1,
                  init_fov=-1, r3_resolution=112, r3_checkpoint="r3",
                  r3_recent_frames=3, r3_bank_size=8, r3_min_confidence=1.02,
                  r3_min_pose_confidence=1.0)
    return SimpleNamespace(**(values | overrides))


class FakeBackend:
    def __init__(self):
        self.calls = 0
        self.confidence = 2.0
        self.pose_confidence = 10.0
        self.closed = False
        self.tx = 0.0

    def infer(self, image):
        self.calls += 1
        height, width = image.shape[-2:]
        Rt = torch.eye(4)[:3]
        Rt[0, 3] = self.tx
        return dict(Rt=Rt, K=torch.tensor([[112., 0., width / 2],
                                         [0., 112., height / 2], [0., 0., 1.]]),
                    depth=torch.full((height, width), 2.),
                    confidence=torch.full((height, width), self.confidence),
                    pose_confidence=self.pose_confidence)

    def close(self):
        self.closed = True


class RectificationTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.arange(3 * 8 * 12).float().reshape(3, 8, 12) / (3 * 8 * 12)
        self.K = torch.tensor([[10., 0., 5.5], [0., 10., 3.5], [0., 0., 1.]])
        self.depth = torch.full((8, 12), 2.)
        self.conf = torch.full((8, 12), 2.)

    def test_identity_camera_and_z_depth_pointmap(self):
        rgb, idepth, conf, points = rectify_geometry(
            self.image, self.depth, self.conf, self.K, 10., 1.02)
        torch.testing.assert_close(rgb, self.image)
        torch.testing.assert_close(idepth, torch.full((1, 1, 8, 12), .5))
        self.assertTrue(conf.bool().all())
        torch.testing.assert_close(points[3, 5], torch.tensor([-.1, -.1, 2.]))

    def test_nonuniform_resize_preserves_pixel_centers(self):
        # Half width, full height: map model K back without averaging fx/fy.
        K = self.K.clone()
        K[0, 0] /= 2
        K[0, 2] = (K[0, 2] + .5) / 2 - .5
        rgb, idepth, confidence, _ = rectify_geometry(
            self.image, self.depth[:, :6], self.conf[:, :6], K, 10., 1.02)
        torch.testing.assert_close(rgb, self.image)
        torch.testing.assert_close(idepth[..., 1:-1], torch.full((1, 1, 8, 10), .5))
        self.assertTrue(confidence[..., 1:-1].bool().all())

    def test_anisotropic_offcenter_intrinsics_warp_rgb_and_depth_together(self):
        K = self.K.clone()
        K[0, 0] = 5
        K[0, 2] = 6
        depth = torch.arange(12).float()[None].expand(8, -1) + 1
        image = (depth / 20)[None].expand(3, -1, -1)
        rgb, idepth, conf, points = rectify_geometry(image, depth, self.conf, K, 10, 1.02)
        valid = conf[0, 0].bool()
        torch.testing.assert_close(rgb[0][valid] * 20, idepth[0, 0][valid].reciprocal())
        torch.testing.assert_close(points[..., 2][valid], idepth[0, 0][valid].reciprocal())

    def test_bad_depth_and_confidence_are_not_gaussians(self):
        self.depth[1, :4] = torch.tensor([0, -1, float('nan'), float('inf')])
        self.conf[2, :3] = torch.tensor([1.01, float('nan'), float('inf')])
        _, idepth, confidence, points = rectify_geometry(
            self.image, self.depth, self.conf, self.K, 10., 1.02)
        self.assertTrue(torch.isfinite(idepth).all())
        self.assertTrue(torch.isfinite(points).all())
        self.assertEqual(confidence[0, 0, 1, :4].sum(), 0)
        self.assertEqual(confidence[0, 0, 2, :3].sum(), 0)

    def test_user_mask_is_preserved_and_reprojected(self):
        mask = torch.ones(1, 8, 12)
        mask[:, 3, 5] = 0
        _, idepth, confidence, points = rectify_geometry(
            self.image, self.depth, self.conf, self.K, 10., 1.02, mask)
        self.assertEqual(confidence[0, 0, 3, 5], 0)
        self.assertEqual(idepth[0, 0, 3, 5], 0)
        self.assertEqual(points[3, 5].sum(), 0)


class R3ProviderTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.provider = R3GeometryProvider(112, 84, options(), self.backend)
        self.image = torch.rand(3, 84, 112)

    def observe(self, frame_id=0, **info):
        info.setdefault("is_test", False)
        image = self.provider.observe(self.image, info, frame_id)
        return image, info

    def test_frame_retrieval_does_not_advance_stream(self):
        image, info = self.observe()
        first = self.provider.estimate_frame_geometry(image, info)
        second = self.provider.estimate_frame_geometry(image, info)
        self.assertEqual(self.backend.calls, 1)
        torch.testing.assert_close(first.Rt, second.Rt)
        self.assertTrue(first.metadata["scale_consistent"])
        self.assertEqual(first.metadata["pointmap_space"], "camera")
        self.assertEqual(info["_r3_geometry"].pointmap.device.type, "cpu")
        self.assertFalse(first.idepth.requires_grad)

    def test_repeated_or_out_of_order_observation_is_rejected(self):
        self.observe(3)
        for frame_id in (3, 2):
            with self.assertRaisesRegex(ValueError, 'exactly once'):
                self.observe(frame_id)
        self.assertEqual(self.backend.calls, 1)

    def test_unobserved_frame_fails_without_legacy_model(self):
        with self.assertRaisesRegex(RuntimeError, 'observed'):
            self.provider.estimate_frame_geometry(self.image, {})
        self.assertEqual(self.backend.calls, 0)

    def test_bootstrap_uses_cached_geometry_and_fixed_focal(self):
        image0, info0 = self.observe(0)
        self.backend.tx = 1
        image1, info1 = self.observe(1)
        Rts, focal, residual = self.provider.initialize_bootstrap(
            [], [dict(image=image0, info=info0), dict(image=image1, info=info1)])
        self.assertEqual(self.backend.calls, 2)
        self.assertEqual(Rts.shape, (2, 4, 4))
        self.assertEqual(Rts[1, 0, 3], 1)
        self.assertEqual(focal, 112)
        self.assertEqual(residual, 0)

    def test_no_sparse_matches_still_allows_geometry_motion(self):
        _, reference = self.observe(0)
        self.backend.tx = 1
        _, current = self.observe(1)
        self.assertTrue(self.provider.should_add_keyframe(current, reference, 30, False))
        self.assertFalse(self.provider.should_add_keyframe(reference, reference, 30, True))

    def test_low_confidence_is_rejected_even_for_test_frames(self):
        _, reference = self.observe(0)
        self.backend.tx = 1
        self.backend.confidence = 1.001
        _, current = self.observe(1, is_test=True)
        self.assertFalse(self.provider.should_add_keyframe(current, reference, 30, True))
        self.backend.confidence = 2
        self.backend.pose_confidence = float('nan')
        _, current = self.observe(2)
        self.assertFalse(self.provider.should_add_keyframe(current, reference, 30, True))

    def test_backend_pose_correction_does_not_change_camera_depth(self):
        _, reference_info = self.observe(0)
        corrected = torch.eye(4)
        corrected[:3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        corrected[1, 3] = 3
        reference = SimpleNamespace(info=reference_info, is_test=False, index=0,
                                    get_Rt=lambda: corrected)
        self.backend.tx = 1
        image, info = self.observe(1)
        before = self.provider.estimate_frame_geometry(image, info)
        after = self.provider.initialize_incremental([reference], None, 1, False, image, info)
        torch.testing.assert_close(after.Rt, before.Rt @ corrected)
        torch.testing.assert_close(after.idepth, before.idepth)
        torch.testing.assert_close(after.pointmap, before.pointmap)
        torch.testing.assert_close(info["_r3_raw_Rt"], before.Rt)
        self.assertEqual(self.backend.calls, 2)

    def test_resolution_and_reboot_changes_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'resolution'):
            self.provider.observe(torch.zeros(3, 56, 112), {}, 0)
        with self.assertRaisesRegex(ValueError, 'bootstrap'):
            self.provider.initialize_bootstrap([], rebooting=True)
        with self.assertRaisesRegex(ValueError, 'COLMAP'):
            R3GeometryProvider(112, 84, options(use_colmap_poses=True), self.backend)

    def test_explicit_virtual_fov_and_resource_release(self):
        provider = R3GeometryProvider(112, 84, options(init_fov=90), self.backend)
        self.assertAlmostEqual(provider.f_init, 56)
        provider.close()
        self.assertTrue(self.backend.closed)

    def test_factory_uses_native_backend_without_importing_depth_v2(self):
        with patch('geometry.r3_provider.R3Backend', return_value=self.backend):
            provider = make_geometry_provider('r3', 112, 84, None, None, 2, options())
        self.assertIsInstance(provider, R3GeometryProvider)

    def test_legacy_observation_and_selection_remain_noops(self):
        provider = GeometryProvider()
        self.assertIs(provider.observe(self.image, {}, 0), self.image)
        self.assertFalse(provider.requires_observation)
        self.assertFalse(provider.should_add_keyframe({}, {}, 30, False))
        self.assertTrue(provider.should_add_keyframe({}, {}, 30, True))
        provider.close()


class ArgsTests(unittest.TestCase):
    @staticmethod
    def parse(*args):
        with patch('sys.argv', ['train.py', '-s', 'capture', *args]):
            return get_args()

    def test_default_is_native_r3_and_legacy_is_explicit(self):
        self.assertEqual(self.parse().geometry_provider, 'r3')
        self.assertEqual(self.parse().max_active_keyframes, 40)
        self.assertEqual(self.parse('--geometry_provider', 'default').max_active_keyframes, 200)
        self.assertEqual(self.parse('--geometry_provider', 'default').geometry_provider, 'default')

    def test_invalid_r3_settings_fail_before_gpu_loading(self):
        cases = [('--enable_reboot',), ('--use_colmap_poses',),
                 ('--r3_bank_size', '0'), ('--r3_bank_size', '1'), ('--r3_recent_frames', '0'),
                 ('--r3_resolution', '28'), ('--r3_min_confidence', 'nan'),
                 ('--r3_min_pose_confidence', '-1'), ('--num_keyframes_miniba_bootstrap', '1')]
        for case in cases:
            with self.subTest(case=case), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(*case)

    def test_gpu_keyframe_budget_preserves_anchor_and_swap_window(self):
        for provider in ('r3', 'default'):
            for budget in ('0', '8', '20'):
                with self.subTest(provider=provider, budget=budget):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        self.parse('--geometry_provider', provider, '--max_active_keyframes', budget)
            self.assertEqual(self.parse('--geometry_provider', provider,
                                        '--max_active_keyframes', '21').max_active_keyframes, 21)


class BackendTests(unittest.TestCase):
    def test_single_step_normalization_shapes_and_bounded_diagnostics(self):
        backend = R3Backend.__new__(R3Backend)
        state = SimpleNamespace(frame_count=0, frame_order=[2],
                                frame_post_scores={0: 10., 2: 4.},
                                frame_score_history={0: [10.], 2: [4.]})
        backend.dtype = torch.bfloat16
        backend.model = Mock(online_state=state, _persistent_post_scores={0: 10., 2: 4.})
        prediction = dict(pose_enc=torch.zeros(1, 1, 9),
                          depth=torch.ones(1, 1, 28, 42, 1),
                          depth_conf=torch.full((1, 1, 28, 42), 2.))

        def step(images, **kwargs):
            self.assertEqual(images.shape, (1, 1, 3, 28, 42))
            self.assertFalse(kwargs['use_ray_pose'])
            torch.testing.assert_close(images[0, 0, :, 0, 0],
                                       (torch.ones(3) - torch.tensor([.485, .456, .406])) /
                                       torch.tensor([.229, .224, .225]))
            state.frame_count = 3
            return prediction

        backend.model.forward_online_step.side_effect = step
        backend.decode_pose = Mock(return_value=(torch.eye(4)[:3][None, None],
                                                 torch.eye(3)[None, None]))
        from contextlib import nullcontext
        with patch('torch.autocast', return_value=nullcontext()):
            result = backend.infer(torch.ones(3, 28, 42))
        self.assertEqual(result['depth'].shape, (28, 42))
        self.assertEqual(result['Rt'].shape, (3, 4))
        self.assertEqual(result['pose_confidence'], 4)
        self.assertEqual(state.frame_post_scores, {2: 4.})
        self.assertEqual(backend.model._persistent_post_scores, {2: 4.})
        backend.model.forward_online_step.assert_called_once()


if __name__ == '__main__':
    unittest.main()
