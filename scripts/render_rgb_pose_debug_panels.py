import os
import asyncio

import numpy as np
from PIL import Image, ImageDraw

import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf


ROOT = "/home/shiva/Downloads/datasets/ycb"
SCENE_STAGE_PATH = "/home/shiva/isaacsim/ycb_multi_scene.usd"
CAMERA_PATH = "/World/CameraMain"

RAYST3R_DIR = "/home/shiva/Documents/rayst3r/outputs/rayst3r_preds"
POSES_PATH = os.path.join(RAYST3R_DIR, "extrinsics_c2w.npy")
INTRINSICS_PATH = os.path.join(RAYST3R_DIR, "intrinsics.npy")
OUT_DIR = os.path.join(RAYST3R_DIR, "rgb_pose_debug")

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


def _assign_semantics(parent_prim, label):
    try:
        from omni.isaac.core.utils.semantics import add_update_semantics

        add_update_semantics(parent_prim, label)
    except Exception:
        pass


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


def _set_camera_pose(stage, T_wc, apply_cv_to_usd):
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not cam_prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {CAMERA_PATH}")

    T = np.asarray(T_wc, dtype=np.float64)
    if apply_cv_to_usd:
        s = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
        T = T @ s

    xform = UsdGeom.Xformable(cam_prim)
    op = None
    for existing in xform.GetOrderedXformOps():
        if existing.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op = existing
            break
    if op is None:
        op = xform.AddTransformOp()
    op.Set(Gf.Matrix4d(T.tolist()))


def _to_uint8_rgb(rgb_raw):
    rgb = np.asarray(rgb_raw)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"Unexpected RGB shape: {rgb.shape}")
    return rgb[..., :3].astype(np.uint8)


def _add_title(rgb, text):
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 24), fill=(0, 0, 0))
    draw.text((8, 5), text, fill=(255, 255, 255))
    return np.asarray(img)


async def render_rgb_debug():
    os.makedirs(OUT_DIR, exist_ok=True)

    c2ws = np.load(POSES_PATH).astype(np.float32)
    intrinsics = np.load(INTRINSICS_PATH).astype(np.float32)
    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses shape (N,4,4), got {c2ws.shape}")
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape (N,3,3), got {intrinsics.shape}")
    if c2ws.shape[0] != intrinsics.shape[0]:
        raise ValueError("Pose and intrinsics view count mismatch")

    pred_depth = np.load(os.path.join(RAYST3R_DIR, "depths.npy"))
    if pred_depth.ndim != 3 or pred_depth.shape[0] != c2ws.shape[0]:
        raise ValueError("Could not infer consistent resolution from depths.npy")
    height, width = int(pred_depth.shape[1]), int(pred_depth.shape[2])

    stage = _setup_stage()
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)

    render_product = rep.create.render_product(CAMERA_PATH, (width, height))
    rgb_anno = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_anno.attach([render_product])

    rgb_converted = []
    rgb_raw = []

    for i in range(c2ws.shape[0]):
        _set_camera_intrinsics_from_K(cam_prim, intrinsics[i], width, height)

        _set_camera_pose(stage, c2ws[i], apply_cv_to_usd=True)
        for _ in range(3):
            await rep.orchestrator.step_async()
            await omni.kit.app.get_app().next_update_async()
        left = _to_uint8_rgb(rgb_anno.get_data())

        _set_camera_pose(stage, c2ws[i], apply_cv_to_usd=False)
        for _ in range(3):
            await rep.orchestrator.step_async()
            await omni.kit.app.get_app().next_update_async()
        right = _to_uint8_rgb(rgb_anno.get_data())

        rgb_converted.append(left)
        rgb_raw.append(right)

        left_t = _add_title(left, f"cv->usd converted pose view {i:03d}")
        right_t = _add_title(right, f"raw c2w as usd view {i:03d}")
        sep = np.full((height, 8, 3), 16, dtype=np.uint8)
        panel = np.concatenate([left_t, sep, right_t], axis=1)
        Image.fromarray(panel).save(os.path.join(OUT_DIR, f"view_{i:03d}.png"))

        if i % 5 == 0 or i == c2ws.shape[0] - 1:
            print(f"[RGB-DEBUG] rendered {i + 1}/{c2ws.shape[0]}")

    np.save(os.path.join(OUT_DIR, "rgb_cv_to_usd.npy"), np.stack(rgb_converted, axis=0))
    np.save(os.path.join(OUT_DIR, "rgb_raw_c2w.npy"), np.stack(rgb_raw, axis=0))
    print(f"[RGB-DEBUG] done, saved panels to: {OUT_DIR}")


asyncio.ensure_future(render_rgb_debug())
