# rayst3r-depth-benchmark-isaacsim

This repo is used to generate synthetic RGB, depth, mask, camera intrinsics, and camera poses in Isaac Sim, feed that synthetic data into RaySt3R, and then benchmark RaySt3R's predicted depths against Isaac Sim ground truth at the same camera views.

The core idea is:

1. Build a synthetic scene in Isaac Sim.
2. Export scene observations in a format that RaySt3R can consume.
3. Let RaySt3R predict depth and confidence for sampled camera poses around the object or scene.
4. Re-render the same scene in Isaac Sim at those exact sampled poses.
5. Compare RaySt3R predicted depth against Isaac ground-truth depth and inspect how depth error relates to RaySt3R confidence.

## Pipeline Overview

The scripts in this repo cover two stages:

- Synthetic data generation for RaySt3R input.
- Ground-truth rendering and debugging for depth benchmarking after RaySt3R inference.

Typical flow:

1. Use Isaac Sim to capture an input RGB/depth/mask sample and camera metadata.
2. Run RaySt3R externally using those Isaac-generated inputs.
3. Use RaySt3R's exported `extrinsics_c2w.npy`, `intrinsics.npy`, `depths.npy`, and `confidence.npy`.
4. Render Isaac Sim ground-truth depth at those same poses.
5. Visualize and evaluate prediction-vs-ground-truth depth.

## Scripts

### `scripts/capture_rgb_depth_seg.py`

Captures a single view from an existing Isaac Sim stage for a single target object. It exports:

- `rgb.png`
- `depth.png`
- `mask.png`
- `intrinsics.pt`
- `cam2world.pt`

This is the single-object capture script. It can auto-assign semantic labels to one object prim and produce a target mask for that class. Use it when you want a minimal RaySt3R input sample for one object already loaded in the scene.

### `scripts/capture_rgb_depth_seg_scene.py`

Builds a small YCB multi-object scene from configured assets, places a fixed camera, and exports one scene sample with:

- RGB
- depth
- binary object mask
- camera intrinsics
- camera pose

This is the scene-level synthetic data generator. It is the script to use when the goal is to create a clean synthetic input scene that can later be given to RaySt3R.

### `scripts/render_gt_depth_from_rayst3r_poses.py`

Rebuilds the same YCB multi-object scene used for capture, loads RaySt3R sampled camera poses from `extrinsics_c2w.npy`, loads the matching intrinsics from `intrinsics.npy`, and renders Isaac Sim ground-truth depth for every sampled view.

It currently saves:

- `isaac_sim_gt/depths.npy`
  This is image-plane depth (`distance_to_image_plane`) when available.
- `isaac_sim_gt/depths_range.npy`
  This is radial depth (`distance_to_camera`).
- `isaac_sim_gt/masks.npy`
- `isaac_sim_gt/extrinsics_c2w.npy`
- `isaac_sim_gt/intrinsics.npy`

This is the main script that aligns Isaac Sim with RaySt3R's sampled views for benchmarking.