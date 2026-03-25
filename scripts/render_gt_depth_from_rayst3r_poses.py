import os
import asyncio

import numpy as np

import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf

# ---------- Scene config (matches capture_rgb_depth_seg_scene.py) ----------
HOME_DIR = os.environ.get("HOME", os.path.expanduser("~"))
ROOT = os.path.join(HOME_DIR, "Downloads", "datasets", "ycb")
SCENE_STAGE_PATH = os.path.join(HOME_DIR, "isaacsim", "ycb_multi_scene.usd")
CAMERA_PATH = "/World/CameraMain"
DEPTH_MAX_METERS = 10.0

ASSETS = [
    {
        "name": "CrackerBox",
        "label": "003_cracker_box",
        "rel_path": "003_cracker_box_google_16k/003_cracker_box/google_16k/textured.usd",
        "t": (-0.18, -0.10, 0.00),
        "r": (0.0, 0.0, 15.0),
        "s": (1.0, 1.0, 1.0),
    },
    {
        "name": "SugarBox",
        "label": "004_sugar_box",
        "rel_path": "004_sugar_box_google_16k/004_sugar_box/google_16k/textured.usd",
        "t": (0.00, -0.05, 0.00),
        "r": (0.0, 0.0, -10.0),
        "s": (1.0, 1.0, 1.0),
    },
    {
        "name": "TomatoSoupCan",
        "label": "005_tomato_soup_can",
        "rel_path": "005_tomato_soup_can_google_16k/005_tomato_soup_can/google_16k/textured.usd",
        "t": (0.18, -0.08, 0.00),
        "r": (0.0, 0.0, 20.0),
        "s": (1.0, 1.0, 1.0),
    },
    {
        "name": "MustardBottle",
        "label": "006_mustard_bottle",
        "rel_path": "006_mustard_bottle_google_16k/006_mustard_bottle/google_16k/textured.usd",
        "t": (-0.10, 0.12, 0.00),
        "r": (0.0, 0.0, -25.0),
        "s": (1.0, 1.0, 1.0),
    },
    {
        "name": "GelatinBox",
        "label": "009_gelatin_box",
        "rel_path": "009_gelatin_box_google_16k/009_gelatin_box/google_16k/textured.usd",
        "t": (0.10, 0.10, 0.00),
        "r": (0.0, 0.0, 5.0),
        "s": (1.0, 1.0, 1.0),
    },
    {
        "name": "Spoon",
        "label": "031_spoon",
        "rel_path": "031_spoon_google_16k/031_spoon/google_16k/textured.usd",
        "t": (0.00, 0.22, 0.01),
        "r": (0.0, 0.0, 45.0),
        "s": (1.0, 1.0, 1.0),
    },
]

# ---------- RaySt3R input/output ----------
RAYST3R_DIR = os.path.join(HOME_DIR, "Documents", "rayst3r", "outputs", "rayst3r_preds")
POSES_PATH = os.path.join(RAYST3R_DIR, "extrinsics_c2w.npy")
INTRINSICS_PATH = os.path.join(RAYST3R_DIR, "intrinsics.npy")
GT_OUT_DIR = os.path.join(RAYST3R_DIR, "isaac_sim_gt")


def _assign_semantics(parent_prim, label):
    try:
        from omni.isaac.core.utils.semantics import add_update_semantics

        add_update_semantics(parent_prim, label)
    except Exception as e:
        print(f"[WARN] semantics assignment failed for {parent_prim.GetPath()}: {e}")


def _add_obj(stage, item):
    usd_path = os.path.join(ROOT, item["rel_path"])
    if not os.path.exists(usd_path):
        print(f"[MISSING] {usd_path}")
        return

    parent_path = f"/World/Objects/{item['name']}"
    parent = UsdGeom.Xform.Define(stage, parent_path)
    pprim = parent.GetPrim()

    child = UsdGeom.Xform.Define(stage, f"{parent_path}/Asset")
    child.GetPrim().GetReferences().AddReference(usd_path)

    x = UsdGeom.XformCommonAPI(pprim)
    x.SetTranslate(Gf.Vec3d(*item["t"]))
    x.SetRotate(Gf.Vec3f(*item["r"]), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    x.SetScale(Gf.Vec3f(*item["s"]))

    _assign_semantics(pprim, item["label"])


def _setup_camera(stage):
    cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    cam.GetFocalLengthAttr().Set(35.0)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))


def _setup_stage():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Objects")

    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.GetIntensityAttr().Set(2500.0)

    for item in ASSETS:
        _add_obj(stage, item)

    _setup_camera(stage)
    ctx.save_as_stage(SCENE_STAGE_PATH)
    return omni.usd.get_context().get_stage()


def _set_camera_intrinsics_from_K(cam_prim, K, width, height):
    fx = float(K[0, 0])
    fy = float(K[1, 1])

    cam = UsdGeom.Camera(cam_prim)
    h_ap = float(cam.GetHorizontalApertureAttr().Get())
    if h_ap == 0.0:
        h_ap = 20.955
        cam.GetHorizontalApertureAttr().Set(h_ap)

    focal = fx * h_ap / float(width)
    if focal <= 0.0:
        raise ValueError(f"Invalid focal computed from intrinsics: {focal}")

    v_ap = focal * float(height) / fy

    cam.GetFocalLengthAttr().Set(float(focal))
    cam.GetVerticalApertureAttr().Set(float(v_ap))


def _set_camera_pose_from_c2w_cv(stage, camera_path, T_wc_cv):
    cam_prim = stage.GetPrimAtPath(camera_path)
    if not cam_prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {camera_path}")

    # USD cam: +X right, +Y up, looks -Z
    # CV  cam: +X right, +Y down, looks +Z
    S = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    T_wc_usd = np.asarray(T_wc_cv, dtype=np.float64) @ S

    xform = UsdGeom.Xformable(cam_prim)
    ops = xform.GetOrderedXformOps()
    op = None
    for existing in ops:
        if existing.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op = existing
            break
    if op is None:
        op = xform.AddTransformOp()

    op.Set(Gf.Matrix4d(T_wc_usd.tolist()))


def _normalize_depth_meters(depth_raw):
    depth = np.asarray(depth_raw)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth, 0.0, DEPTH_MAX_METERS)
    return depth.astype(np.float32)


def _extract_semantic_payload(sem_raw):
    if isinstance(sem_raw, dict):
        sem = sem_raw.get("data")
        if sem is None and "buffer" in sem_raw:
            sem = sem_raw["buffer"]
        return sem
    return sem_raw


async def render_gt_depth_stack():
    os.makedirs(GT_OUT_DIR, exist_ok=True)

    c2ws = np.load(POSES_PATH).astype(np.float32)
    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses shape (N,4,4), got {c2ws.shape}")

    intrinsics = np.load(INTRINSICS_PATH).astype(np.float32)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape (N,3,3), got {intrinsics.shape}")

    if intrinsics.shape[0] != c2ws.shape[0]:
        raise ValueError("intrinsics and poses must have same number of views")

    pred_depth_path = os.path.join(RAYST3R_DIR, "depths.npy")
    if os.path.exists(pred_depth_path):
        pred_depth = np.load(pred_depth_path)
        if pred_depth.ndim != 3:
            raise ValueError(f"Expected predicted depth shape (N,H,W), got {pred_depth.shape}")
        if pred_depth.shape[0] != c2ws.shape[0]:
            raise ValueError("Pose count and depth view count mismatch")
        height, width = int(pred_depth.shape[1]), int(pred_depth.shape[2])
    else:
        height, width = 480, 640
        print("[WARN] depths.npy not found; defaulting resolution to 640x480")

    print(f"[GT] views={c2ws.shape[0]} resolution=({width},{height})")

    stage = _setup_stage()
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)

    render_product = rep.create.render_product(CAMERA_PATH, (width, height))

    depth_range_anno = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    try:
        depth_z_anno = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    except Exception:
        depth_z_anno = None
        print("[WARN] distance_to_image_plane annotator unavailable; falling back to distance_to_camera for depths.npy")
    try:
        sem_anno = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False}
        )
    except Exception:
        sem_anno = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")

    depth_range_anno.attach([render_product])
    if depth_z_anno is not None:
        depth_z_anno.attach([render_product])
    sem_anno.attach([render_product])

    depths = []
    depths_range = []
    masks = []

    for idx in range(c2ws.shape[0]):
        _set_camera_intrinsics_from_K(cam_prim, intrinsics[idx], width, height)
        _set_camera_pose_from_c2w_cv(stage, CAMERA_PATH, c2ws[idx])

        for _ in range(4):
            await rep.orchestrator.step_async()
            await omni.kit.app.get_app().next_update_async()

        depth_range_m = _normalize_depth_meters(depth_range_anno.get_data())
        if depth_z_anno is not None:
            depth_z_m = _normalize_depth_meters(depth_z_anno.get_data())
        else:
            depth_z_m = depth_range_m

        depths.append(depth_z_m)
        depths_range.append(depth_range_m)

        sem = _extract_semantic_payload(sem_anno.get_data())
        sem_arr = np.asarray(sem)
        if sem_arr.ndim == 3 and sem_arr.shape[-1] == 1:
            sem_arr = sem_arr[..., 0]
        masks.append((sem_arr > 0).astype(bool))

        if idx % 5 == 0 or idx == c2ws.shape[0] - 1:
            print(f"[GT] rendered {idx + 1}/{c2ws.shape[0]}")

    depths = np.stack(depths, axis=0).astype(np.float32)
    depths_range = np.stack(depths_range, axis=0).astype(np.float32)
    masks = np.stack(masks, axis=0).astype(bool)

    np.save(os.path.join(GT_OUT_DIR, "depths.npy"), depths)
    np.save(os.path.join(GT_OUT_DIR, "depths_range.npy"), depths_range)
    np.save(os.path.join(GT_OUT_DIR, "masks.npy"), masks)
    np.save(os.path.join(GT_OUT_DIR, "extrinsics_c2w.npy"), c2ws)
    np.save(os.path.join(GT_OUT_DIR, "intrinsics.npy"), intrinsics)

    print("[GT] done")
    print(f"[GT] saved: {os.path.join(GT_OUT_DIR, 'depths.npy')} shape={depths.shape} (z-depth, distance_to_image_plane)")
    print(f"[GT] saved: {os.path.join(GT_OUT_DIR, 'depths_range.npy')} shape={depths_range.shape} (radial, distance_to_camera)")
    print(f"[GT] saved: {os.path.join(GT_OUT_DIR, 'masks.npy')} shape={masks.shape}")


asyncio.ensure_future(render_gt_depth_stack())
