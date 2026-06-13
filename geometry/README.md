# Geometry Provider Sidecars

External geometry providers can attach per-frame priors through files in:

```text
${SOURCE_PATH}/${GEOMETRY_DIR}/${IMAGE_STEM}.npz
${SOURCE_PATH}/${GEOMETRY_DIR}/${IMAGE_STEM}.pt
${SOURCE_PATH}/${GEOMETRY_DIR}/${IMAGE_STEM}.json
```

`GEOMETRY_DIR` defaults to `geometry` and is used when `--geometry_provider` is not
`default`.

Supported fields:

- `Rt`, `w2c`, `T_cw`, or `world_to_camera`: 4x4 world-to-camera matrix.
- `c2w`, `T_wc`, or `camera_to_world`: 4x4 camera-to-world matrix. It is inverted on load.
- `focal`, `fx`, `K`, or `intrinsics`: focal length in pixels, or a 3x3 intrinsics matrix.
- `idepth`, `inverse_depth`, or `inv_depth`: inverse depth map.
- `depth` or `metric_depth`: depth map. It is converted to inverse depth.
- `depth_confidence`, `confidence`, or `conf`: depth confidence map.
- `pointmap`, `points`, or `xyz`: optional dense/sparse 3D points for future initialization paths.
- `metadata`: optional source-specific dictionary.

All tensor-like arrays are loaded on CPU and moved to CUDA by the dataset loader.
Depth and confidence maps may be `H x W`, `1 x H x W`, or `H x W x 1`.
