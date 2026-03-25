import os
import asyncio

import numpy as np
import torch
from PIL import Image

import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom, Usd

# ---- CONFIG ----
CAMERA_PATH = "/World/Camera"        # Must match your camera prim path exactly
OBJECT_PRIM_PATH = "/World/textured"   # Set to your object prim path for auto-labeling
TARGET_CLASS = "003_cracker_box"               # Class label for segmentation target
AUTO_ASSIGN_SEMANTIC_LABEL = True

RESOLUTION = (1280, 720)
OUT_DIR = "/home/shiva/Documents/isaac_sim_rendering"
DEPTH_MAX_METERS = 10.0


def _extract_semantic_payload(sem_raw):
    if isinstance(sem_raw, dict):
        sem = sem_raw.get("data")
        info = sem_raw.get("info", {})
        if sem is None and "buffer" in sem_raw:
            sem = sem_raw["buffer"]
        if not isinstance(info, dict):
            info = {}
        return sem, info
    return sem_raw, {}


def _label_info_contains_class(label_info, target_class: str) -> bool:
    if isinstance(label_info, str):
        return target_class in label_info
    if isinstance(label_info, dict):
        return any(target_class == str(v) for v in label_info.values())
    if isinstance(label_info, (list, tuple)):
        return any(target_class == str(v) for v in label_info)
    return False


def _try_assign_semantics(stage):
    if not AUTO_ASSIGN_SEMANTIC_LABEL:
        return

    obj_prim = stage.GetPrimAtPath(OBJECT_PRIM_PATH)
    if not obj_prim.IsValid():
        print(f"[capture] WARN: OBJECT_PRIM_PATH not found: {OBJECT_PRIM_PATH}")
        return

    try:
        from omni.isaac.core.utils.semantics import add_update_semantics

        add_update_semantics(obj_prim, TARGET_CLASS)
        print(f"[capture] semantic label assigned: {OBJECT_PRIM_PATH} -> '{TARGET_CLASS}'")
    except Exception as e:
        print(f"[capture] WARN: failed to auto-assign semantics: {e}")


def _get_camera_params_cv(stage, camera_path, width, height):
    cam_prim = stage.GetPrimAtPath(camera_path)
    cam = UsdGeom.Camera(cam_prim)

    focal = float(cam.GetFocalLengthAttr().Get())
    h_ap = float(cam.GetHorizontalApertureAttr().Get())
    v_ap = float(cam.GetVerticalApertureAttr().Get())

    fx = (focal / h_ap) * float(width)
    fy = (focal / v_ap) * float(height)
    cx = float(width) * 0.5
    cy = float(height) * 0.5

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # USD world transform (camera->world in USD camera coords)
    imageable = UsdGeom.Imageable(cam_prim)
    T_wc_usd = np.array(
        imageable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64
    )

    # Convert USD camera coords to OpenCV camera coords.
    # USD cam: +X right, +Y up, looks -Z
    # CV  cam: +X right, +Y down, looks +Z
    S = np.diag([1.0, -1.0, -1.0, 1.0])
    T_wc_cv = T_wc_usd @ S

    return K, T_wc_cv.astype(np.float32)


def _normalize_depth(depth_raw):
    depth = np.asarray(depth_raw)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth, 0.0, DEPTH_MAX_METERS)
    depth_mm = np.round(depth * 1000.0).astype(np.uint16)
    return depth_mm


def _build_mask(sem, sem_info, target_class: str, out_shape):
    target_ids = []
    id_to_labels = sem_info.get("idToLabels", {}) if isinstance(sem_info, dict) else {}

    for sid, label_info in id_to_labels.items():
        if _label_info_contains_class(label_info, target_class):
            try:
                target_ids.append(int(sid))
            except ValueError:
                pass

    sem_arr = np.asarray(sem)
    if sem_arr.ndim == 3 and sem_arr.shape[-1] == 1:
        sem_arr = sem_arr[..., 0]

    if target_ids and sem_arr.ndim == 2:
        mask = np.isin(sem_arr, target_ids).astype(np.uint8) * 255
    elif sem_arr.ndim == 2:
        mask = (sem_arr > 0).astype(np.uint8) * 255
    elif sem_arr.ndim == 3 and sem_arr.shape[-1] in (3, 4):
        mask = np.any(sem_arr[..., :3] != 0, axis=-1).astype(np.uint8) * 255
    else:
        h, w = out_shape
        mask = np.zeros((h, w), dtype=np.uint8)

    return mask, target_ids


async def capture():
    print("[capture] starting...")
    os.makedirs(OUT_DIR, exist_ok=True)

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not cam_prim.IsValid():
        print(f"[capture] ERROR: camera prim not found at {CAMERA_PATH}")
        return

    _try_assign_semantics(stage)

    render_product = rep.create.render_product(CAMERA_PATH, RESOLUTION)

    rgb_anno = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_anno = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    try:
        sem_anno = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False}
        )
    except Exception:
        sem_anno = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")

    rgb_anno.attach([render_product])
    depth_anno.attach([render_product])
    sem_anno.attach([render_product])

    for _ in range(8):
        await rep.orchestrator.step_async()
        await omni.kit.app.get_app().next_update_async()

    rgb = rgb_anno.get_data()
    depth_raw = depth_anno.get_data()
    sem_raw = sem_anno.get_data()
    sem, sem_info = _extract_semantic_payload(sem_raw)

    if rgb is None or depth_raw is None or sem is None:
        print("[capture] ERROR: annotator returned empty data")
        return

    rgb_np = np.asarray(rgb)
    if rgb_np.ndim != 3 or rgb_np.shape[-1] < 3:
        print(f"[capture] ERROR: unexpected rgb shape: {rgb_np.shape}")
        return
    rgb_img = rgb_np[..., :3].astype(np.uint8)

    depth_png = _normalize_depth(depth_raw)
    mask, target_ids = _build_mask(sem, sem_info, TARGET_CLASS, (RESOLUTION[1], RESOLUTION[0]))

    width, height = RESOLUTION
    K, T_wc_cv = _get_camera_params_cv(stage, CAMERA_PATH, width, height)

    rgb_path = os.path.join(OUT_DIR, "rgb.png")
    depth_path = os.path.join(OUT_DIR, "depth.png")
    mask_path = os.path.join(OUT_DIR, "mask.png")
    intr_path = os.path.join(OUT_DIR, "intrinsics.pt")
    c2w_path = os.path.join(OUT_DIR, "cam2world.pt")

    Image.fromarray(rgb_img).save(rgb_path)
    Image.fromarray(depth_png).save(depth_path)
    Image.fromarray(mask).save(mask_path)

    torch.save(torch.from_numpy(K), intr_path)
    torch.save(torch.from_numpy(T_wc_cv), c2w_path)

    print("[capture] done")
    print(f"[capture] saved: {rgb_path}")
    print(f"[capture] saved: {depth_path} (uint16, millimeters, clipped to {DEPTH_MAX_METERS}m)")
    print(f"[capture] saved: {mask_path}")
    print(f"[capture] saved: {intr_path} shape={tuple(K.shape)}")
    print(f"[capture] saved: {c2w_path} shape={tuple(T_wc_cv.shape)}")
    print(f"[capture] target class '{TARGET_CLASS}' -> semantic IDs: {target_ids}")
    if not target_ids:
        print("[capture] NOTE: No class-ID mapping found; mask uses non-background fallback.")


asyncio.ensure_future(capture())
