"""
Offline evaluation utilities.

• Detection:  mAP@0.5 via pycocotools.
• Tracking:   IDF1, MOTA, ID switches via motmetrics.
• Drift:      centroid drift analysis for ID-switch root-cause.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Detection evaluation — mAP@0.5 (COCO API)
# ---------------------------------------------------------------------------


def evaluate_detection(
    gt_coco_json: str,
    pred_coco_json: str,
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """
    Compute mAP using pycocotools.

    Parameters
    ----------
    gt_coco_json : path to ground-truth COCO JSON
    pred_coco_json : path to predictions in COCO results format
    iou_thresh : IoU threshold (default 0.5)

    Returns
    -------
    dict with keys: mAP, mAP_per_class, precision, recall
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_coco_json)
    coco_dt = coco_gt.loadRes(pred_coco_json)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.iouThrs = [iou_thresh]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    result = {
        "mAP": float(stats[0]),
        "mAP_50": float(stats[0]),
        "precision": float(stats[0]),
        "recall": float(stats[8]) if len(stats) > 8 else 0.0,
    }
    logger.info("Detection eval | mAP@{:.2f} = {:.4f}", iou_thresh, result["mAP"])
    return result


# ---------------------------------------------------------------------------
# Tracking evaluation — IDF1, MOTA (motmetrics)
# ---------------------------------------------------------------------------


def evaluate_tracking(
    gt_mot_file: str,
    pred_mot_file: str,
) -> dict[str, float]:
    """
    Compute MOT metrics from MOTChallenge-format files.

    File format per line (comma-separated):
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

    Returns
    -------
    dict with keys: idf1, mota, motp, num_switches, mostly_tracked, mostly_lost
    """
    import motmetrics as mm

    gt = mm.io.loadtxt(gt_mot_file, fmt="mot15-2D")
    pred = mm.io.loadtxt(pred_mot_file, fmt="mot15-2D")

    acc = mm.utils.compare_to_groundtruth(gt, pred, "iou", distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="eval")

    result = {
        "idf1": float(summary["idf1"].values[0]),
        "mota": float(summary["mota"].values[0]),
        "motp": float(summary["motp"].values[0]),
        "num_switches": int(summary["num_switches"].values[0]),
        "mostly_tracked": int(summary["mostly_tracked"].values[0]),
        "mostly_lost": int(summary["mostly_lost"].values[0]),
    }
    logger.info("Tracking eval | IDF1={:.4f}  MOTA={:.4f}  Switches={}",
                result["idf1"], result["mota"], result["num_switches"])
    return result


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------


def build_confusion_matrix(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Build a (num_classes × num_classes) confusion matrix.
    Rows = ground truth, Columns = predictions.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pr in zip(gt_labels, pred_labels):
        if 0 <= gt < num_classes and 0 <= pr < num_classes:
            cm[gt, pr] += 1
    return cm


# ---------------------------------------------------------------------------
# Tracking drift measurement
# ---------------------------------------------------------------------------


def measure_tracking_drift(
    gt_centroids: dict[int, list[tuple[int, int]]],
    pred_centroids: dict[int, list[tuple[int, int]]],
) -> dict[str, float]:
    """
    For each matched track ID, compute mean Euclidean distance between
    GT centroid and predicted centroid across frames.

    Returns global mean drift and per-track drift.
    """
    drifts: list[float] = []
    per_track: dict[int, float] = {}
    common_ids = set(gt_centroids.keys()) & set(pred_centroids.keys())

    for tid in common_ids:
        gt_pts = np.array(gt_centroids[tid], dtype=np.float32)
        pr_pts = np.array(pred_centroids[tid], dtype=np.float32)
        min_len = min(len(gt_pts), len(pr_pts))
        d = np.linalg.norm(gt_pts[:min_len] - pr_pts[:min_len], axis=1)
        per_track[tid] = float(np.mean(d))
        drifts.extend(d.tolist())

    mean_drift = float(np.mean(drifts)) if drifts else 0.0
    logger.info("Tracking drift | mean={:.2f}px across {} tracks", mean_drift, len(common_ids))
    return {"mean_drift_px": mean_drift, "per_track": per_track}
