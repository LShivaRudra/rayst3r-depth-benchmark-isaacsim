import os
import numpy as np
from PIL import Image, ImageDraw

HOME_DIR = os.environ.get("HOME", os.path.expanduser("~"))
RAYST3R_DIR = os.path.join(HOME_DIR, "Documents", "rayst3r", "outputs", "rayst3r_preds")
PRED_DEPTH_PATH = os.path.join(RAYST3R_DIR, "depths.npy")
GT_DEPTH_PATH = os.path.join(RAYST3R_DIR, "isaac_sim_gt", "depths.npy")
OUT_DIR = os.path.join(RAYST3R_DIR, "depth_side_by_side")

PRED_MASK_PATH = os.path.join(RAYST3R_DIR, "masks.npy")
GT_MASK_PATH = os.path.join(RAYST3R_DIR, "isaac_sim_gt", "masks.npy")

USE_PER_VIEW_SHARED_RANGE = True
GLOBAL_CLIP = None  # example: (0.0, 2.0)


def _load(path, name):
    if os.path.exists(path):
        return np.load(path)
    raise FileNotFoundError(f"Missing {name}: {path}")


def _calc_clip(values, fallback=(0.0, 1.0)):
    if values.size == 0:
        return fallback
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi > lo:
        return (lo, hi)
    return (lo, lo + 1e-6)


def _norm_to_rgb(norm01):
    # high-contrast pseudo-color: blue -> cyan -> yellow -> red
    n = np.clip(norm01, 0.0, 1.0)
    r = np.clip(1.5 * n - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * n - 1.5), 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * n, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def _depth_to_rgb(depth_m, valid_mask, clip_range):
    out = np.zeros(depth_m.shape + (3,), dtype=np.uint8)
    out[...] = (18, 18, 18)

    lo, hi = float(clip_range[0]), float(clip_range[1])
    denom = max(hi - lo, 1e-6)
    norm = (depth_m - lo) / denom
    norm = np.clip(norm, 0.0, 1.0)

    color = _norm_to_rgb(norm)
    out[valid_mask] = color[valid_mask]
    return out


def _error_to_rgb(abs_err, valid_mask):
    out = np.zeros(abs_err.shape + (3,), dtype=np.uint8)
    out[...] = (18, 18, 18)

    vals = abs_err[valid_mask]
    if vals.size == 0:
        return out

    lo = 0.0
    hi = float(np.percentile(vals, 99.0))
    if hi <= lo:
        hi = lo + 1e-6

    norm = np.clip((abs_err - lo) / (hi - lo), 0.0, 1.0)

    # dark -> red -> yellow for error magnitude
    r = np.clip(1.2 * norm, 0.0, 1.0)
    g = np.clip(2.0 * norm - 0.6, 0.0, 1.0)
    b = np.clip(0.35 - norm, 0.0, 0.35)
    color = np.stack([r, g, b], axis=-1)
    color_u8 = (color * 255.0).astype(np.uint8)

    out[valid_mask] = color_u8[valid_mask]
    return out


def _mask_edges(mask):
    up = np.roll(mask, 1, axis=0)
    down = np.roll(mask, -1, axis=0)
    left = np.roll(mask, 1, axis=1)
    right = np.roll(mask, -1, axis=1)
    edge = mask & (~(up & down & left & right))
    edge[0, :] = False
    edge[-1, :] = False
    edge[:, 0] = False
    edge[:, -1] = False
    return edge


def _overlay_edges(rgb, edge_mask, color=(255, 255, 255)):
    out = rgb.copy()
    out[edge_mask] = np.array(color, dtype=np.uint8)
    return out


def _panel_with_title(rgb, title):
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 24), fill=(0, 0, 0))
    draw.text((8, 5), title, fill=(255, 255, 255))
    return np.asarray(img)


def _shape_is_nhw(arr):
    return arr.ndim in (3,)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pred = _load(PRED_DEPTH_PATH, "pred depth").astype(np.float32)
    gt = _load(GT_DEPTH_PATH, "gt depth").astype(np.float32)

    if _shape_is_nhw(pred) and _shape_is_nhw(gt):
        pass
    else:
        raise ValueError("pred and gt depth must have shape (N,H,W)")

    if pred.shape == gt.shape:
        pass
    else:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")

    pred_mask = np.load(PRED_MASK_PATH).astype(bool) if os.path.exists(PRED_MASK_PATH) else None
    gt_mask = np.load(GT_MASK_PATH).astype(bool) if os.path.exists(GT_MASK_PATH) else None

    if pred_mask is not None:
        if pred_mask.shape == pred.shape:
            pass
        else:
            raise ValueError(f"pred mask shape mismatch: {pred_mask.shape} vs {pred.shape}")

    if gt_mask is not None:
        if gt_mask.shape == gt.shape:
            pass
        else:
            raise ValueError(f"gt mask shape mismatch: {gt_mask.shape} vs {gt.shape}")

    n, h, w = pred.shape

    for i in range(n):
        p = pred[i]
        g = gt[i]

        p_valid = np.isfinite(p) & (p > 0.0)
        g_valid = np.isfinite(g) & (g > 0.0)

        if pred_mask is not None:
            p_valid &= pred_mask[i]
        if gt_mask is not None:
            g_valid &= gt_mask[i]

        both_valid = p_valid & g_valid

        if USE_PER_VIEW_SHARED_RANGE:
            vals = []
            if p_valid.any():
                vals.append(p[p_valid])
            if g_valid.any():
                vals.append(g[g_valid])
            if len(vals) > 0:
                clip = _calc_clip(np.concatenate(vals))
            else:
                clip = (0.0, 1.0)
        else:
            if GLOBAL_CLIP is None:
                vals = []
                if p_valid.any():
                    vals.append(p[p_valid])
                if g_valid.any():
                    vals.append(g[g_valid])
                if len(vals) > 0:
                    clip = _calc_clip(np.concatenate(vals))
                else:
                    clip = (0.0, 1.0)
            else:
                clip = GLOBAL_CLIP

        p_rgb = _depth_to_rgb(p, p_valid, clip)
        g_rgb = _depth_to_rgb(g, g_valid, clip)

        p_edges = _mask_edges(p_valid)
        g_edges = _mask_edges(g_valid)

        p_rgb = _overlay_edges(p_rgb, p_edges, color=(255, 255, 255))
        g_rgb = _overlay_edges(g_rgb, g_edges, color=(255, 255, 255))

        abs_err = np.abs(p - g)
        e_rgb = _error_to_rgb(abs_err, both_valid)
        e_edges = _mask_edges(both_valid)
        e_rgb = _overlay_edges(e_rgb, e_edges, color=(255, 255, 255))

        p_panel = _panel_with_title(p_rgb, f"RaySt3R depth {i:03d}")
        g_panel = _panel_with_title(g_rgb, f"Isaac GT depth {i:03d}")
        e_panel = _panel_with_title(e_rgb, f"|pred-gt| error {i:03d}")

        sep = np.full((h, 8, 3), 16, dtype=np.uint8)
        out = np.concatenate([p_panel, sep, g_panel, sep, e_panel], axis=1)

        out_path = os.path.join(OUT_DIR, f"view_{i:03d}.png")
        Image.fromarray(out).save(out_path)

    print("[VIS] done")
    print(f"[VIS] saved {n} images to: {OUT_DIR}")


if __name__ == "__main__":
    main()
