import os
import asyncio

import numpy as np
import torch
from PIL import Image

import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Usd, Gf

# ---------- Config ----------
HOME_DIR = os.environ.get("HOME", os.path.expanduser("~"))
ROOT = os.path.join(HOME_DIR, "Downloads", "datasets", "ycb")
OUT_DIR = os.path.join(HOME_DIR, "Documents", "isaac_sim_rendering")
SCENE_STAGE_PATH = os.path.join(HOME_DIR, "isaacsim", "ycb_multi_scene.usd")

RESOLUTION = (1280, 720)
DEPTH_MAX_METERS = 10.0
CAMERA_PATH = "/World/CameraMain"

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


def _normalize_depth(depth_raw):
    depth = np.asarray(depth_raw)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth, 0.0, DEPTH_MAX_METERS)
    return np.round(depth * 1000.0).astype(np.uint16)


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

    imageable = UsdGeom.Imageable(cam_prim)
    T_wc_usd = np.array(
        imageable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64
    )

    # USD cam: +X right, +Y up, looks -Z
    # CV  cam: +X right, +Y down, looks +Z
    S = np.diag([1.0, -1.0, -1.0, 1.0])
    T_wc_cv = T_wc_usd @ S

    return K, T_wc_cv.astype(np.float32)


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
    print(f"[OK] {item['name']} -> {usd_path}")


def _setup_camera(stage):
    cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    cam.GetFocalLengthAttr().Set(35.0)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

    # Fixed oblique viewpoint for a nice full-scene view.
    x = UsdGeom.XformCommonAPI(cam.GetPrim())
    x.SetTranslate(Gf.Vec3d(0.75, -0.85, 0.55))
    x.SetRotate(Gf.Vec3f(65.0, 0.0, 40.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


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


async def capture_scene_once():
    print("[scene-capture] starting single-frame capture...")
    os.makedirs(OUT_DIR, exist_ok=True)

    stage = _setup_stage()

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

    rgb = np.asarray(rgb_anno.get_data())
    depth_mm = _normalize_depth(depth_anno.get_data())

    sem_raw = sem_anno.get_data()
    sem, _ = _extract_semantic_payload(sem_raw)
    sem_arr = np.asarray(sem)
    if sem_arr.ndim == 3 and sem_arr.shape[-1] == 1:
        sem_arr = sem_arr[..., 0]

    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        print(f"[scene-capture] ERROR: unexpected RGB shape {rgb.shape}")
        return

    # Binary scene mask: all objects=1, background=0
    mask = (sem_arr > 0).astype(np.uint8) * 255

    K, T_wc_cv = _get_camera_params_cv(stage, CAMERA_PATH, RESOLUTION[0], RESOLUTION[1])

    rgb_path = os.path.join(OUT_DIR, "rgb.png")
    depth_path = os.path.join(OUT_DIR, "depth.png")
    mask_path = os.path.join(OUT_DIR, "mask.png")
    intr_path = os.path.join(OUT_DIR, "intrinsics.pt")
    c2w_path = os.path.join(OUT_DIR, "cam2world.pt")

    Image.fromarray(rgb[..., :3].astype(np.uint8)).save(rgb_path)
    Image.fromarray(depth_mm).save(depth_path)
    Image.fromarray(mask).save(mask_path)

    torch.save(torch.from_numpy(K), intr_path)
    torch.save(torch.from_numpy(T_wc_cv), c2w_path)

    print("[scene-capture] done")
    print(f"[scene-capture] saved: {rgb_path}")
    print(f"[scene-capture] saved: {depth_path} (uint16 mm, clipped to {DEPTH_MAX_METERS}m)")
    print(f"[scene-capture] saved: {mask_path} (binary 0/255)")
    print(f"[scene-capture] saved: {intr_path} shape={tuple(K.shape)}")
    print(f"[scene-capture] saved: {c2w_path} shape={tuple(T_wc_cv.shape)}")


asyncio.ensure_future(capture_scene_once())
