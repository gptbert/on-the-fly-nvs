# 2026 Geometry Model Selection

Research cutoff: 2026-09-09. Repository baseline: `8b377c9`.

Status: native R3 integration implemented and selected as the runtime default at the user's request. The old BA + Depth Anything V2 frontend remains explicitly selectable. XFeat and LPIPS/VGG are retained. CPU adapter tests and upstream API smoke coverage are provided; full CUDA reconstruction, quality, latency, memory, and Docker validation are still pending. Only checkpoint headers were inspected locally, not the full pretrained weights.

## Decision

Select **R3** as the first experimental replacement for the combined depth-estimation and pose-initialization frontend. Keep `GeometryProvider` as the boundary and retain `SceneModel` for scene-specific Gaussian optimization. This is an engineering selection for phone capture and a single consumer GPU, not a claim that R3 wins every benchmark.

- Start with `r3.safetensors` for rooms and short captures; evaluate `r3_long.safetensors` separately for larger trajectories. The author released inference and weights on 2026-05-26 and training code on 2026-06-19. The inference model has about 372M parameters. [R3 repository](https://github.com/KevinXu02/R3)
- Use **LingBot-Map** as the long-sequence comparison, initially in a separate preprocessing pass. Its public pipeline includes windowing and CPU prediction offload. Do not assume it can share a consumer GPU with growing Gaussian state at its published standalone speed. [LingBot-Map repository](https://github.com/Robbyant/lingbot-map)
- Keep XFeat for the existing sparse/dense feature consumers and fallback path. Keep LPIPS/VGG as the same evaluation metric across experiments.
- The user requested an immediate switch: `r3` is now the default native provider. This changes runtime selection, not the validation status below. Use `--geometry_provider default` for the legacy baseline.

## Hardware And License Constraints

The stated server is an "RTX 4080 24G". NVIDIA lists the standard RTX 4080 and 4080 SUPER as **16GB**. Treat 24GB as a reported capacity, not verified hardware. Confirm on the reconstruction server before selecting runtime limits. [NVIDIA specifications](https://www.nvidia.com/en-gb/geforce/graphics-cards/40-series/rtx-4080-family/)

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Plan for both 16GB and 24GB. Parameter count and checkpoint size are not peak VRAM: activations, KV caches, retained input/output tensors, Gaussians, optimizer state, and viewer buffers must all be measured together. None of the published FPS figures establishes end-to-end performance for this repository.

R3's model card explicitly licenses its weights under **CC BY-NC 4.0**, despite the original wrapper code being Apache-2.0. The existing project's [license](../LICENSE.md) also restricts commercial use. This selection assumes research/evaluation; it is not commercial-license clearance. [R3 model card](https://huggingface.co/KevinXu02/R3)

## Candidate Review

Dates below distinguish actual releases from conference years. Judgments in the last column are specific to this project.

| Candidate | Verified progress | Release and integration considerations | Decision |
| --- | --- | --- | --- |
| [R3](https://github.com/KevinXu02/R3) | May/June 2026; relative-pose regression and bounded keyframe context | Inference, training code, and two checkpoints available; public evaluation code still listed as TODO | First experimental frontend |
| [LingBot-Map](https://github.com/Robbyant/lingbot-map) | April 2026 paper; May benchmark release; June KV-cache fix | Public code and weights; larger streaming state and explicit window/reset management | Long-sequence comparison |
| [VGGT-SLAM 2.0](https://github.com/MIT-SPARK/VGGT-SLAM) | January 2026 release; June online code | Complete SLAM pipeline, including submap optimization; integration overlaps more with the existing backend | Alternative if loop closure dominates |
| [ZipMap](https://github.com/Haian-Jin/ZipMap) | March 2026 paper; April code, weights, streaming release | Stateful test-time-training design; authors identify streaming as less explored; non-commercial weights | Research comparison, not first integration |
| [HyDen / MetaDepth](https://github.com/facebookresearch/metadepth) | April/May 2026 relative-depth and metric-point releases | High-resolution monocular geometry; no joint video pose estimator; FAIR non-commercial terms | Not the primary fix for trajectory drift |
| [ZipDepth](https://github.com/fabiotosi92/ZipDepth) | July 2026 code and pretrained release | Compact monocular depth model with mobile deployment focus | Consider only if phone-side inference becomes a requirement |
| [LoMa](https://github.com/davnords/LoMa) | April 2026 inference release; ECCV 2026 | Learned local matching, not a pose/depth frontend; changing descriptor consumers is a separate refactor | Later matching experiment |
| [M3](https://github.com/InternRobotics/M3) | March 2026 paper | Public main branch currently contains README and assets, not a runnable implementation | Do not select for immediate replacement |
| [Scal3R](https://github.com/NVlabs/scal3r) | ECCV 2026; relative-pose queries and pose-graph optimization | README simultaneously says weights are coming soon and gives a download command; checkpoint availability could not be verified | Hold until artifact availability is clear |
| [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) / [Pi3X](https://github.com/yyfz/Pi3) | Released November/December **2025**, not new 2026 releases | Useful maintained baselines; DA3 now recommends refreshed `-1.1` large/giant checkpoints | DA3-LARGE-1.1 as a bounded-window quality baseline |

DA3-LARGE-1.1 has conflicting license labels: the repository model table says CC BY-NC 4.0, while its Hugging Face card says Apache-2.0. Do not infer commercial permission from the card alone; resolve the conflict with the authors before commercial deployment. [Repository model table](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards), [checkpoint card](https://huggingface.co/depth-anything/DA3-LARGE-1.1)

## Why R3 First

The project's quality target is coherent geometry across moving phone frames, not simply sharper single-frame depth. R3 jointly supplies relative pose evidence and dense geometry through a DA3-based frontend. Its bounded keyframe-bank design is a better initial fit for shared-GPU integration than selecting a billion-parameter model solely from a point-cloud demo. This is an inference from architecture, not a measured quality or memory advantage in this project. R3 still reports long-stream drift and the need for resets; those limitations must remain visible. [R3 paper](https://arxiv.org/html/2605.26519v1)

LingBot-Map remains a meaningful comparison. Its paper reports 13.28GB and 20.29 FPS in one window-size-64 ablation, but those numbers describe that experiment, not this server or concurrent Gaussian training. They are not a VRAM guarantee. [LingBot-Map paper, Table 7](https://arxiv.org/html/2604.14141v1)

Do not replace LPIPS to make scores look better: keeping the same metric makes baseline comparisons interpretable. Likewise, removing XFeat before replacing its consumers would conflate geometry quality with a second, independent algorithm change.

## Implemented Integration

The native path is implemented in [r3_provider.py](../geometry/r3_provider.py); deployment options are documented in [README](../README.md#native-r3-geometry).

```text
phone RGB frames / recorded video
  -> R3GeometryProvider: observe each frame once
  -> pose + intrinsics + inverse depth + confidence + camera-space pointmap
  -> existing feature consumers and keyframe/geometry validation
  -> SceneModel: Gaussian initialization, optimization, rendering
```

1. **Central loading.** Source is pinned to `e345f1112fbc9c451d44c9c5aa22d9bfec2a954d`; weights use HF revision `c1f2aeccfa14d035a0e7b18f188253003a8417f0` and verified SHA-256. Assets reside in `models/r3/<revision>/`. Model tensors load strictly on CPU before transfer to CUDA; only three known training counters are excluded.
2. **Observation before admission.** Every received frame invokes the public `forward_online_step` API once with explicit ImageNet normalization. Provider pose motion can admit frames without sparse matches. Bootstrap and incremental initialization reuse the observed geometry.
3. **Bounded state.** Dynamic KV retention uses a first-frame anchor, 3 recent frames and 8 bank keyframes by default. Offline diagnostic scores are pruned. Cached predictions reside in each frame's CPU info until consumed, then are removed; active scene frames have a separate CPU-offload limit.
4. **Geometry convention.** OpenCV `w2c`, positive Z-depth, relative scale, and camera-space pointmaps are explicit. Depth confidence is thresholded in its original `exp(logit)+1` score domain; the adapter emits binary validity, not a calibrated probability.
5. **Camera rectification.** RGB, Z-depth and masks are sampled along the same rays into a fixed centered virtual camera. The full predicted intrinsic matrix participates in the warp, including nonuniform resize and pixel-center offsets.
6. **Scale preservation.** Coherent priors skip affine inverse-depth alignment and freeze the per-frame depth scale/offset. Camera-space pointmaps follow optimized keyframe poses. Gaussian and pose optimization remain enabled.
7. **Conservative feedback.** The most recent available optimized non-test reference supplies a rigid correction to incoming R3 poses. Automatic upstream re-anchoring, global pose rewriting and metric anchoring are disabled. Legacy reboot/COLMAP initialization flags fail explicitly; long-trajectory drift and recovery remain limitations.
8. **GPU scheduling.** Gaussian optimization is joined before each R3 observation. Defaults keep 40 active scene keyframes instead of the legacy 200. LPIPS loads only for requested test-view evaluation. Memory ceilings still require measurement with the actual capture and Gaussian count.

Local evidence: [provider interface](../geometry/provider.py), [training flow](../train.py), [camera/sidecar handling](../dataloaders/image_dataset.py), [depth alignment and pointmaps](../scene/keyframe.py).

### Upstream Runtime Traps

The published R3 CLI loads input views onto the GPU before invoking inference. A bounded model KV cache therefore does not, by itself, establish bounded end-to-end application memory. Native integration must stream inputs and release or offload outputs. The export code also maps outputs through `output_frame_ids`; preserve that mapping instead of assuming every input produces an output at the same index. [R3 inference implementation](https://github.com/KevinXu02/R3/blob/main/infer.py)

The demo defaults to `test`, which keeps all KV entries. Its `local` preset uses dynamic retention. The `long` preset enables a second `DA3METRIC-LARGE` model and applies preset values after CLI parsing. Do not assume `--mode long --no-metric_scale` disables that extra model. Configure the native adapter explicitly and leave metric anchoring off until it is deliberately tested. [R3 demo implementation](https://github.com/KevinXu02/R3/blob/main/demo.py)

Retain Python 3.12. Upstream packaging declares Python >=3.10, but that is not proof that the complete dependency stack works here. Validate a minimal inference dependency set with the existing CUDA/PyTorch stack; do not import upstream's training, viewer, or optional metric/export dependencies wholesale. [R3 packaging](https://github.com/KevinXu02/R3/blob/main/setup.py), [requirements](https://github.com/KevinXu02/R3/blob/main/requirements.txt)

## Validation Still Required

The runtime switch was requested before GPU acceptance testing. Validate the native path on a saved phone sequence before relying on a live capture. The initial integration serializes frontend inference and Gaussian updates to limit simultaneous allocation peaks.

Use the same extracted frames, held-out views, optimization iterations, resolution, and Gaussian budget across the original frontend, R3, and a bounded-window DA3/LingBot comparison. Do not compare PSNR from different datasets or copy a paper's improvement into this project.

R3 rectifies images into a fixed virtual camera; its exported RGB and validity masks differ from the original phone images. For cross-provider image metrics, use a common calibrated evaluation camera and mask, or explicitly report that results use different evaluation support. Raw scores from different rectification/masking policies are not directly comparable.

Required captures: a textured room, a weak-texture wall, a corridor with a return loop, outdoor vegetation, a portrait recording, and a clip with motion blur or people moving through the scene. Use a short clip for correctness, a 2-5 minute clip for resource measurements, and a longer capture for cache-growth/reset tests.

| Check | Acceptance condition |
| --- | --- |
| Pose/depth/pointmap contract | Projection round-trip, crop/intrinsic transform, positive finite depth, and frame-ID tests pass |
| Streaming lifecycle | One observation per frame; bounded caches; correct bootstrap, reset, dropped-frame, and re-anchoring behavior |
| Reconstruction completion | All validation captures finish without OOM or silent source/scale changes |
| Render quality | Comparable held-out PSNR/SSIM/LPIPS and visibly reduced target artifacts, with scene-by-scene results |
| Trajectory | ATE/RPE only where usable reference trajectories exist; otherwise report reprojection/loop consistency without calling it ground-truth accuracy |
| Memory | Combined geometry + 3DGS + viewer peak measured on the actual GPU, with a reserved margin |
| Latency | Report preprocessing, first-result delay, steady-state frame latency, Gaussian-update cost, and capture backlog separately |
| Provenance | Record input frame list, package versions, checkpoint hashes, inference settings, and capture settings |

The tests cover storage, frame lifecycle, rectification, geometry admission, rigid backend feedback, and an optional real upstream CPU API smoke test with random weights. Neither synthetic inputs nor random-weight inference establishes reconstruction quality. No quality, FPS, or VRAM improvement is claimed until these experiments have run.
