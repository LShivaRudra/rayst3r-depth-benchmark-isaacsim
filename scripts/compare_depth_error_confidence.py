import os
import numpy as np

RAYST3R_DIR = "/home/shiva/Documents/rayst3r/outputs/rayst3r_preds"
PRED_DEPTH_PATH = os.path.join(RAYST3R_DIR, "depths.npy")
CONF_PATH = os.path.join(RAYST3R_DIR, "confidence.npy")
PRED_MASK_PATH = os.path.join(RAYST3R_DIR, "masks.npy")
GT_DEPTH_PATH = os.path.join(RAYST3R_DIR, "isaac_sim_gt", "depths.npy")
GT_MASK_PATH = os.path.join(RAYST3R_DIR, "isaac_sim_gt", "masks.npy")
OUT_DIR = os.path.join(RAYST3R_DIR, "depth_error_eval")

NUM_BINS = 20
CONF_PERCENTILE_CLIP = (1.0, 99.0)


def _load(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
    return np.load(path)


def _validate_shapes(pred_depth, conf, gt_depth):
    if pred_depth.ndim != 3 or gt_depth.ndim != 3 or conf.ndim != 3:
        raise ValueError("pred_depth, conf, gt_depth must all have shape (N,H,W)")
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(f"pred_depth shape {pred_depth.shape} != gt_depth shape {gt_depth.shape}")
    if pred_depth.shape != conf.shape:
        raise ValueError(f"pred_depth shape {pred_depth.shape} != conf shape {conf.shape}")


def _build_valid_mask(pred_depth, gt_depth, pred_mask=None, gt_mask=None):
    valid = np.isfinite(pred_depth) & np.isfinite(gt_depth)
    valid &= pred_depth > 0.0
    valid &= gt_depth > 0.0

    if pred_mask is not None:
        valid &= pred_mask.astype(bool)
    if gt_mask is not None:
        valid &= gt_mask.astype(bool)

    return valid


def _bin_error_vs_confidence(abs_err, conf, valid_mask, num_bins):
    conf_valid = conf[valid_mask]
    err_valid = abs_err[valid_mask]

    cmin, cmax = np.percentile(conf_valid, CONF_PERCENTILE_CLIP)
    if cmax <= cmin:
        cmax = cmin + 1e-6

    edges = np.linspace(cmin, cmax, num_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows = []
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == num_bins - 1:
            sel = valid_mask & (conf >= lo) & (conf <= hi)
        else:
            sel = valid_mask & (conf >= lo) & (conf < hi)

        cnt = int(sel.sum())
        if cnt == 0:
            rows.append((centers[i], np.nan, np.nan, cnt))
            continue

        vals = abs_err[sel]
        rows.append((centers[i], float(np.mean(vals)), float(np.median(vals)), cnt))

    return np.asarray(rows, dtype=np.float64)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pred_depth = _load(PRED_DEPTH_PATH, "pred depth").astype(np.float32)
    conf = _load(CONF_PATH, "confidence").astype(np.float32)
    gt_depth = _load(GT_DEPTH_PATH, "GT depth").astype(np.float32)

    _validate_shapes(pred_depth, conf, gt_depth)

    pred_mask = np.load(PRED_MASK_PATH) if os.path.exists(PRED_MASK_PATH) else None
    gt_mask = np.load(GT_MASK_PATH) if os.path.exists(GT_MASK_PATH) else None

    if pred_mask is not None and pred_mask.shape != pred_depth.shape:
        raise ValueError(f"pred mask shape {pred_mask.shape} != depth shape {pred_depth.shape}")
    if gt_mask is not None and gt_mask.shape != gt_depth.shape:
        raise ValueError(f"gt mask shape {gt_mask.shape} != depth shape {gt_depth.shape}")

    abs_err = np.abs(pred_depth - gt_depth).astype(np.float32)
    valid = _build_valid_mask(pred_depth, gt_depth, pred_mask=pred_mask, gt_mask=gt_mask)

    if valid.sum() == 0:
        raise RuntimeError("No valid pixels for evaluation after masking/filtering")

    mae = float(np.mean(abs_err[valid]))
    medae = float(np.median(abs_err[valid]))
    rmse = float(np.sqrt(np.mean((abs_err[valid] ** 2))))

    rows = _bin_error_vs_confidence(abs_err, conf, valid, NUM_BINS)

    np.save(os.path.join(OUT_DIR, "abs_error.npy"), abs_err)
    np.save(os.path.join(OUT_DIR, "valid_mask.npy"), valid)
    np.save(os.path.join(OUT_DIR, "error_vs_confidence_bins.npy"), rows)

    csv_path = os.path.join(OUT_DIR, "error_vs_confidence_bins.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("conf_bin_center,mean_abs_err_m,median_abs_err_m,count\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{int(r[3])}\n")

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"views={pred_depth.shape[0]}\n")
        f.write(f"resolution={pred_depth.shape[2]}x{pred_depth.shape[1]}\n")
        f.write(f"valid_pixels={int(valid.sum())}\n")
        f.write(f"mae_m={mae:.6f}\n")
        f.write(f"median_ae_m={medae:.6f}\n")
        f.write(f"rmse_m={rmse:.6f}\n")

    print("[EVAL] done")
    print(f"[EVAL] MAE={mae:.6f}m  MedAE={medae:.6f}m  RMSE={rmse:.6f}m")
    print(f"[EVAL] saved: {summary_path}")
    print(f"[EVAL] saved: {csv_path}")

    try:
        import matplotlib.pyplot as plt

        valid_conf = conf[valid]
        valid_err = abs_err[valid]

        fig = plt.figure(figsize=(7, 4.5), dpi=150)
        plt.scatter(valid_conf, valid_err, s=1, alpha=0.08)

        non_empty = rows[:, 3] > 0
        plt.plot(rows[non_empty, 0], rows[non_empty, 1], linewidth=2.0)

        plt.xlabel("RaySt3R confidence")
        plt.ylabel("Absolute depth error (m)")
        plt.title("Depth Error vs Confidence")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        plot_path = os.path.join(OUT_DIR, "error_vs_confidence.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"[EVAL] saved: {plot_path}")
    except Exception as e:
        print(f"[EVAL] plot skipped: {e}")


if __name__ == "__main__":
    main()
