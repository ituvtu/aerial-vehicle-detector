from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from src.optics import CameraOpticsConfig, assign_distance_band, estimate_distance_m_from_px


def audit_model_architecture(
    yaml_or_weights_path: str | Path,
    max_params: int = 500_000,
) -> dict[str, Any]:
    """
    Validates model architecture against Rule 5 (< 500k params) and P2-P3-P4 strides [4, 8, 16].
    """
    from ultralytics import YOLO

    path_str = str(yaml_or_weights_path)
    model = YOLO(path_str)

    unfused_params = sum(p.numel() for p in model.model.parameters())

    fused_model = copy.deepcopy(model.model)
    fused_model.fuse(verbose=False)
    fused_params = sum(p.numel() for p in fused_model.parameters())

    detect_layer = model.model.model[-1]
    if hasattr(detect_layer, "stride") and detect_layer.stride is not None:
        strides = [float(s) for s in detect_layer.stride.tolist()]
    else:
        strides = [4.0, 8.0, 16.0]

    if unfused_params >= max_params:
        raise ValueError(f"Unfused params {unfused_params} exceeds limit of {max_params}")
    if fused_params >= max_params:
        raise ValueError(f"Fused params {fused_params} exceeds limit of {max_params}")
    if strides != [4.0, 8.0, 16.0]:
        raise ValueError(f"Expected strides [4.0, 8.0, 16.0], got {strides}")

    return {
        "unfused_params": unfused_params,
        "fused_params": fused_params,
        "max_params_limit": max_params,
        "strides": strides,
        "num_layers": len(list(model.model.model)),
    }


def compute_box_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """Computes IoU between two xyxy bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def match_frame_detections(
    gt_boxes_xyxy: list[tuple[float, float, float, float]],
    pred_boxes_xyxy_conf: list[tuple[float, float, float, float, float]],
    orig_w: int,
    orig_h: int,
    iou_threshold: float = 0.50,
    fov_deg: float = 84.0,
    car_length_m: float = 4.5,
    method: str = "max_side",
) -> dict[str, Any]:
    """
    Performs global greedy IoU matching of predictions to ground truth on a single frame.
    Assigns each GT and prediction to its true optical distance band.
    """
    optics_cfg = CameraOpticsConfig(
        fov_long_deg=fov_deg,
        car_length_m=car_length_m,
        estimation_method=method,  # type: ignore[arg-type]
    )

    gt_info = []
    for g in gt_boxes_xyxy:
        w_px = max(0.0, g[2] - g[0])
        h_px = max(0.0, g[3] - g[1])
        d_m = estimate_distance_m_from_px(w_px, h_px, orig_w, orig_h, optics_cfg)
        band = assign_distance_band(d_m, optics_cfg)
        gt_info.append({"box": g, "distance_m": d_m, "band": band, "matched": False})

    sorted_preds = sorted(pred_boxes_xyxy_conf, key=lambda x: x[4], reverse=True)

    matched_tps: list[dict[str, Any]] = []
    fps: list[dict[str, Any]] = []

    for p in sorted_preds:
        p_box = (p[0], p[1], p[2], p[3])
        p_conf = p[4]
        best_iou = 0.0
        best_gt_idx = -1

        for idx, g in enumerate(gt_info):
            if g["matched"]:
                continue
            iou = compute_box_iou(p_box, g["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            gt_info[best_gt_idx]["matched"] = True
            matched_tps.append({
                "pred_box": p_box,
                "conf": p_conf,
                "gt_box": gt_info[best_gt_idx]["box"],
                "gt_band": gt_info[best_gt_idx]["band"],
                "gt_distance_m": gt_info[best_gt_idx]["distance_m"],
                "iou": best_iou,
            })
        else:
            w_px = max(0.0, p[2] - p[0])
            h_px = max(0.0, p[3] - p[1])
            d_m = estimate_distance_m_from_px(w_px, h_px, orig_w, orig_h, optics_cfg)
            band = assign_distance_band(d_m, optics_cfg)
            fps.append({
                "pred_box": p_box,
                "conf": p_conf,
                "band": band,
                "distance_m": d_m,
            })

    fns: list[dict[str, Any]] = [
        {"gt_box": g["box"], "band": g["band"], "distance_m": g["distance_m"]}
        for g in gt_info if not g["matched"]
    ]

    return {
        "matched_tps": matched_tps,
        "fps": fps,
        "fns": fns,
        "total_gt": len(gt_boxes_xyxy),
        "total_preds": len(pred_boxes_xyxy_conf),
    }


def compute_ap50(tp_flags: list[int], confs: list[float], total_gt: int) -> float:
    """Computes AP@0.5 area under precision-recall envelope."""
    if total_gt <= 0 or not tp_flags:
        return 0.0

    order = sorted(range(len(confs)), key=lambda k: confs[k], reverse=True)
    sorted_tp = [tp_flags[i] for i in order]

    tp_cum = np.cumsum(sorted_tp)
    fp_cum = np.cumsum([1 - x for x in sorted_tp])

    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / (tp_cum + fp_cum)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_change = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i_change + 1] - mrec[i_change]) * mpre[i_change + 1]))
    return max(0.0, min(1.0, ap))


def evaluate_split_metrics(
    frame_records: list[dict[str, Any]],
    gt_boxes_by_frame: dict[str, list[tuple[float, float, float, float]]],
    raw_preds_by_frame: dict[str, list[tuple[float, float, float, float, float]]],
    conf_threshold: float,
    fps: float = 1.0,
    iou_threshold: float = 0.50,
    fov_deg: float = 84.0,
    car_length_m: float = 4.5,
    method: str = "max_side",
    mean_inference_latency_s: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluates detection rate, precision, F1, false alarms/min, time to first detection,
    and mAP@0.5 across distance bands ('0-200m', '200-400m', 'overall').
    """
    bands_list = ["0-200m", "200-400m", "overall"]

    counts: dict[str, dict[str, Any]] = {
        b: {"tp": 0, "fp": 0, "fn": 0, "num_gt": 0, "num_frames": len(frame_records)}
        for b in bands_list
    }
    ap_pool: dict[str, dict[str, list[Any]]] = {
        b: {"tp_flags": [], "confs": [], "total_gt": 0} for b in bands_list
    }

    tps_by_frame_and_band: dict[str, dict[str, int]] = {
        r["frame_id"]: {b: 0 for b in bands_list} for r in frame_records
    }
    gts_by_frame_and_band: dict[str, dict[str, int]] = {
        r["frame_id"]: {b: 0 for b in bands_list} for r in frame_records
    }

    for record in frame_records:
        fid = record["frame_id"]
        w = int(record["width"])
        h = int(record["height"])

        gt_boxes = gt_boxes_by_frame.get(fid, [])
        all_preds = raw_preds_by_frame.get(fid, [])
        filtered_preds = [p for p in all_preds if p[4] >= conf_threshold]

        res_filtered = match_frame_detections(
            gt_boxes, filtered_preds, w, h, iou_threshold, fov_deg, car_length_m, method
        )
        res_all = match_frame_detections(
            gt_boxes, all_preds, w, h, iou_threshold, fov_deg, car_length_m, method
        )

        for tp in res_filtered["matched_tps"]:
            b = tp["gt_band"]
            if b in counts:
                counts[b]["tp"] += 1
                tps_by_frame_and_band[fid][b] += 1
            counts["overall"]["tp"] += 1
            tps_by_frame_and_band[fid]["overall"] += 1

        for fp in res_filtered["fps"]:
            b = fp["band"]
            if b in counts:
                counts[b]["fp"] += 1
            counts["overall"]["fp"] += 1

        for fn in res_filtered["fns"]:
            b = fn["band"]
            if b in counts:
                counts[b]["fn"] += 1
            counts["overall"]["fn"] += 1

        optics_cfg = CameraOpticsConfig(fov_long_deg=fov_deg, car_length_m=car_length_m, estimation_method=method)  # type: ignore[arg-type]
        for g in gt_boxes:
            box_w = max(0.0, g[2] - g[0])
            box_h = max(0.0, g[3] - g[1])
            dist = estimate_distance_m_from_px(box_w, box_h, w, h, optics_cfg)
            b = assign_distance_band(dist, optics_cfg)
            if b in counts:
                counts[b]["num_gt"] += 1
                gts_by_frame_and_band[fid][b] += 1
            counts["overall"]["num_gt"] += 1
            gts_by_frame_and_band[fid]["overall"] += 1

        for tp in res_all["matched_tps"]:
            b = tp["gt_band"]
            conf = tp["conf"]
            if b in ap_pool:
                ap_pool[b]["tp_flags"].append(1)
                ap_pool[b]["confs"].append(conf)
            ap_pool["overall"]["tp_flags"].append(1)
            ap_pool["overall"]["confs"].append(conf)

        for fp in res_all["fps"]:
            b = fp["band"]
            conf = fp["conf"]
            if b in ap_pool:
                ap_pool[b]["tp_flags"].append(0)
                ap_pool[b]["confs"].append(conf)
            ap_pool["overall"]["tp_flags"].append(0)
            ap_pool["overall"]["confs"].append(conf)

    for b in bands_list:
        ap_pool[b]["total_gt"] = counts[b]["num_gt"]

    # Calculate Time to First Detection per video clip and average
    videos_map: dict[str, list[dict[str, Any]]] = {}
    for r in frame_records:
        vid = str(r["video_id"])
        videos_map.setdefault(vid, []).append(r)

    ttfd_results: dict[str, dict[str, Any]] = {b: {"mean_s": None, "per_video": {}} for b in bands_list}

    for b in bands_list:
        clip_delays: list[float] = []
        for vid, records in videos_map.items():
            sorted_rec = sorted(records, key=lambda x: float(x.get("timestamp_sec", x.get("timestamp_s", 0.0))))
            first_gt_ts = None
            first_tp_ts = None

            for r in sorted_rec:
                ts = float(r.get("timestamp_sec", r.get("timestamp_s", 0.0)))
                fid = r["frame_id"]
                if first_gt_ts is None and gts_by_frame_and_band[fid][b] > 0:
                    first_gt_ts = ts
                if first_gt_ts is not None and first_tp_ts is None and tps_by_frame_and_band[fid][b] > 0:
                    first_tp_ts = ts

            if first_gt_ts is not None:
                if first_tp_ts is not None:
                    delay = max(0.0, first_tp_ts - first_gt_ts) + mean_inference_latency_s
                    clip_delays.append(delay)
                    ttfd_results[b]["per_video"][vid] = round(delay, 3)
                else:
                    ttfd_results[b]["per_video"][vid] = None
            else:
                ttfd_results[b]["per_video"][vid] = None

        if clip_delays:
            ttfd_results[b]["mean_s"] = round(float(np.mean(clip_delays)), 3)

    final_bands_metrics: dict[str, Any] = {}
    for b in bands_list:
        tp = counts[b]["tp"]
        fp = counts[b]["fp"]
        fn = counts[b]["fn"]
        n_gt = counts[b]["num_gt"]
        n_frames = counts[b]["num_frames"]

        det_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 and tp == 0 else 0.0)
        f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0
        fa_per_min = (fp * 60.0 * fps / float(n_frames)) if n_frames > 0 else 0.0

        ap = compute_ap50(ap_pool[b]["tp_flags"], ap_pool[b]["confs"], ap_pool[b]["total_gt"])

        final_bands_metrics[b] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_gt": n_gt,
            "num_frames": n_frames,
            "detection_rate": round(det_rate, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "false_alarms_per_min": round(fa_per_min, 4),
            "time_to_first_detection_s": ttfd_results[b]["mean_s"],
            "time_to_first_detection_breakdown": ttfd_results[b]["per_video"],
            "map50": round(ap, 4),
        }

    return {
        "conf_threshold": conf_threshold,
        "num_frames": len(frame_records),
        "bands": final_bands_metrics,
    }


def calibrate_confidence_threshold(
    val_records: list[dict[str, Any]],
    gt_boxes_by_frame: dict[str, list[tuple[float, float, float, float]]],
    raw_preds_by_frame: dict[str, list[tuple[float, float, float, float, float]]],
    candidate_thresholds: list[float] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Calibrates confidence threshold tau* maximizing J(tau) = 0.5 * F1_overall + 0.5 * F1_macro.
    Tie-breaking favors higher tau to suppress false alarms.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [round(0.10 + 0.01 * i, 2) for i in range(76)]

    best_score = -1.0
    best_tau = 0.25
    best_metrics: dict[str, Any] = {}

    for tau in candidate_thresholds:
        m = evaluate_split_metrics(
            frame_records=val_records,
            gt_boxes_by_frame=gt_boxes_by_frame,
            raw_preds_by_frame=raw_preds_by_frame,
            conf_threshold=tau,
            **kwargs,
        )
        bands = m["bands"]
        f1_overall = bands["overall"]["f1"]

        valid_band_f1s = [
            bands[b]["f1"] for b in ("0-200m", "200-400m") if bands[b]["num_gt"] > 0
        ]
        f1_macro = float(np.mean(valid_band_f1s)) if valid_band_f1s else f1_overall

        score = 0.5 * f1_overall + 0.5 * f1_macro

        if score > best_score or (abs(score - best_score) < 1e-6 and tau > best_tau):
            best_score = score
            best_tau = tau
            best_metrics = m

    return {
        "best_conf_threshold": best_tau,
        "best_objective_score": round(best_score, 4),
        "val_metrics_at_best": best_metrics,
        "num_candidates_tested": len(candidate_thresholds),
    }
