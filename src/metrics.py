from __future__ import annotations

import copy
import io
import json
import sys
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

    layers = list(model.model.model)
    backbone_params = sum(sum(p.numel() for p in l.parameters()) for l in layers[0:9])
    neck_params = sum(sum(p.numel() for p in l.parameters()) for l in layers[9:21])
    head_params = sum(sum(p.numel() for p in l.parameters()) for l in layers[21:])

    return {
        "unfused_params": unfused_params,
        "fused_params": fused_params,
        "max_params_limit": max_params,
        "strides": strides,
        "num_layers": len(layers),
        "backbone_params": backbone_params,
        "neck_params": neck_params,
        "head_params": head_params,
    }


def inspect_and_verify_architecture(
    yaml_path: str | Path,
    pretrained_weights: str = "yolo26n.pt",
    output_report_path: str | Path | None = "results/model_architecture_report.json",
    max_params: int = 500_000,
) -> dict[str, Any]:
    """
    Detailed inspection of YOLO26-Pico-P2 architecture before training:
    - Builds model from yaml and loads pretrained stem weights if available.
    - Captures detailed layer information.
    - Verifies unfused_params and fused_params are strictly < 500_000 (Rule 5).
    - Splits params across Backbone (0..8), Neck (9..20), Detect Head (21).
    - Saves report to output_report_path.
    """
    from ultralytics import YOLO

    yaml_p = Path(yaml_path)
    if not yaml_p.exists():
        raise FileNotFoundError(f"Model architecture YAML not found at: {yaml_p}")

    model = YOLO(str(yaml_p))

    # Capture model info
    try:
        model.info(detailed=True, verbose=True)
    except Exception:
        pass

    transferred_items = 0
    total_pretrained_items = 0
    transfer_msg = "none"

    if pretrained_weights:
        try:
            # Capture stdout of transfer
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                model.load(pretrained_weights)
            finally:
                sys.stdout = old_stdout
            out_str = buf.getvalue()
            transfer_msg = out_str.strip()
            # Parse 'Transferred X/Y items'
            if "Transferred " in out_str:
                part = out_str.split("Transferred ")[1].split()[0]
                if "/" in part:
                    transferred_items = int(part.split("/")[0])
                    total_pretrained_items = int(part.split("/")[1])
        except Exception as exc:
            transfer_msg = f"Transfer skipped: {exc}"

    unfused_params = sum(p.numel() for p in model.model.parameters())

    fused_model = copy.deepcopy(model.model)
    fused_model.fuse(verbose=False)
    fused_params = sum(p.numel() for p in fused_model.parameters())

    assert unfused_params < max_params, f"Unfused params {unfused_params} >= {max_params}"
    assert fused_params < max_params, f"Fused params {fused_params} >= {max_params}"

    layers = list(model.model.model)
    backbone_params = sum(sum(p.numel() for p in l.parameters()) for l in layers[0:9])
    neck_params = sum(sum(p.numel() for p in l.parameters()) for l in layers[9:21])
    head_params = sum(sum(p.numel() for p in l.parameters()) for l in layers[21:])

    detect_layer = layers[-1]
    strides = (
        [float(s) for s in detect_layer.stride.tolist()]
        if hasattr(detect_layer, "stride") and detect_layer.stride is not None
        else [4.0, 8.0, 16.0]
    )

    report = {
        "model_name": "YOLO26-Pico-P2",
        "yaml_path": str(yaml_p),
        "unfused_params": unfused_params,
        "fused_params": fused_params,
        "max_params_limit": max_params,
        "strides": strides,
        "num_layers": len(layers),
        "blocks": {
            "backbone_layers_0_8": backbone_params,
            "neck_layers_9_20": neck_params,
            "detect_head_layer_21": head_params,
        },
        "pretrained_transfer": {
            "source": pretrained_weights,
            "transferred_items": transferred_items,
            "total_items": total_pretrained_items,
            "message": transfer_msg[:200],
        },
    }

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[Architecture Inspection] Saved report to {out_p}")

    return report


def transfer_coco_stem_weights(model: Any, weights_path: str = "yolo26n.pt") -> dict[str, Any]:
    """
    Deterministically transfers COCO pretrained weights from yolo26n.pt into compatible
    Pico-P2 stem and backbone layers (model.0..model.4: stem convolutions and C3k2 blocks).
    Validates strict tensor shape equality before loading.
    """
    from ultralytics import YOLO

    src_model = YOLO(weights_path)
    src_state = src_model.model.state_dict()
    target_state = model.model.state_dict()

    matched = {}
    transferred_params = 0
    for k, v in src_state.items():
        if k in target_state and v.shape == target_state[k].shape:
            if any(k.startswith(f"model.{i}.") for i in range(5)):
                matched[k] = v
                transferred_params += v.numel()

    model.model.load_state_dict(matched, strict=False)
    print(
        f"[Pretrained Transfer] Successfully transferred {len(matched)} tensors "
        f"({transferred_params:,} params) from {weights_path} into Pico-P2 backbone (layers 0..4)."
    )
    return {
        "source_weights": weights_path,
        "matched_tensors": len(matched),
        "transferred_params": transferred_params,
        "layers": "model.0..model.4",
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
    Matches against ALL GT boxes (0-200m, 200-400m, >400m) BEFORE distance categorization.
    - Matched GT -> TP in band D_gt.
    - Unmatched GT -> FN in band D_gt.
    - Unmatched Pred -> FP in band D_pred.
    - CRITICAL: Pred matching >400m GT is a TP in >400m and NOT an FP in near/far bands.
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
    """
    Computes standard 101-point interpolated AP@0.5 (COCO standard: recall in 0.00..1.00 step 0.01).
    """
    if total_gt <= 0 or not tp_flags:
        return 0.0

    order = sorted(range(len(confs)), key=lambda k: confs[k], reverse=True)
    tps = np.array(tp_flags, dtype=float)[order]
    fps = 1.0 - tps

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)

    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / (tp_cum + fp_cum)

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    p_interp = []
    for r in recall_thresholds:
        p_at_r = precisions[recalls >= r]
        p_interp.append(float(np.max(p_at_r)) if len(p_at_r) > 0 else 0.0)

    return float(np.mean(p_interp))


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
    and mAP@0.5 across distance bands ('0-200m', '200-400m', '>400m', 'overall_0_400m', 'overall_all').
    """
    bands_list = ["0-200m", "200-400m", ">400m", "overall_0_400m", "overall_all", "overall"]

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

    optics_cfg = CameraOpticsConfig(
        fov_long_deg=fov_deg,
        car_length_m=car_length_m,
        estimation_method=method,  # type: ignore[arg-type]
    )

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

        # 1. Process True Positives at conf >= conf_threshold
        for tp in res_filtered["matched_tps"]:
            b = tp["gt_band"]
            if b in counts:
                counts[b]["tp"] += 1
                tps_by_frame_and_band[fid][b] += 1
            if b in ("0-200m", "200-400m"):
                counts["overall_0_400m"]["tp"] += 1
                counts["overall"]["tp"] += 1
                tps_by_frame_and_band[fid]["overall_0_400m"] += 1
                tps_by_frame_and_band[fid]["overall"] += 1
            counts["overall_all"]["tp"] += 1
            tps_by_frame_and_band[fid]["overall_all"] += 1

        # 2. Process False Positives at conf >= conf_threshold
        for fp in res_filtered["fps"]:
            b = fp["band"]
            if b in counts:
                counts[b]["fp"] += 1
            if b in ("0-200m", "200-400m"):
                counts["overall_0_400m"]["fp"] += 1
                counts["overall"]["fp"] += 1
            counts["overall_all"]["fp"] += 1

        # 3. Process False Negatives at conf >= conf_threshold
        for fn in res_filtered["fns"]:
            b = fn["band"]
            if b in counts:
                counts[b]["fn"] += 1
            if b in ("0-200m", "200-400m"):
                counts["overall_0_400m"]["fn"] += 1
                counts["overall"]["fn"] += 1
            counts["overall_all"]["fn"] += 1

        # 4. Count Ground Truth boxes
        for g in gt_boxes:
            box_w = max(0.0, g[2] - g[0])
            box_h = max(0.0, g[3] - g[1])
            dist = estimate_distance_m_from_px(box_w, box_h, w, h, optics_cfg)
            b = assign_distance_band(dist, optics_cfg)
            if b in counts:
                counts[b]["num_gt"] += 1
                gts_by_frame_and_band[fid][b] += 1
            if b in ("0-200m", "200-400m"):
                counts["overall_0_400m"]["num_gt"] += 1
                counts["overall"]["num_gt"] += 1
                gts_by_frame_and_band[fid]["overall_0_400m"] += 1
                gts_by_frame_and_band[fid]["overall"] += 1
            counts["overall_all"]["num_gt"] += 1
            gts_by_frame_and_band[fid]["overall_all"] += 1

        # 5. Populate PR curves across all candidate predictions
        for tp in res_all["matched_tps"]:
            b = tp["gt_band"]
            conf = tp["conf"]
            if b in ap_pool:
                ap_pool[b]["tp_flags"].append(1)
                ap_pool[b]["confs"].append(conf)
            if b in ("0-200m", "200-400m"):
                ap_pool["overall_0_400m"]["tp_flags"].append(1)
                ap_pool["overall_0_400m"]["confs"].append(conf)
                ap_pool["overall"]["tp_flags"].append(1)
                ap_pool["overall"]["confs"].append(conf)
            ap_pool["overall_all"]["tp_flags"].append(1)
            ap_pool["overall_all"]["confs"].append(conf)

        for fp in res_all["fps"]:
            b = fp["band"]
            conf = fp["conf"]
            if b in ap_pool:
                ap_pool[b]["tp_flags"].append(0)
                ap_pool[b]["confs"].append(conf)
            if b in ("0-200m", "200-400m"):
                ap_pool["overall_0_400m"]["tp_flags"].append(0)
                ap_pool["overall_0_400m"]["confs"].append(conf)
                ap_pool["overall"]["tp_flags"].append(0)
                ap_pool["overall"]["confs"].append(conf)
            ap_pool["overall_all"]["tp_flags"].append(0)
            ap_pool["overall_all"]["confs"].append(conf)

    for b in bands_list:
        ap_pool[b]["total_gt"] = counts[b]["num_gt"]

    # Calculate Time to First Detection per video clip
    videos_map: dict[str, list[dict[str, Any]]] = {}
    for r in frame_records:
        vid = str(r["video_id"])
        videos_map.setdefault(vid, []).append(r)

    ttfd_results: dict[str, dict[str, Any]] = {
        b: {"mean_s": None, "per_video": {}} for b in bands_list
    }

    for b in bands_list:
        clip_delays: list[float] = []
        for vid, records in videos_map.items():
            sorted_rec = sorted(
                records, key=lambda x: float(x.get("timestamp_sec", x.get("timestamp_s", 0.0)))
            )
            first_gt_ts = None
            first_tp_ts = None

            for r in sorted_rec:
                ts = float(r.get("timestamp_sec", r.get("timestamp_s", 0.0)))
                fid = r["frame_id"]
                if first_gt_ts is None and gts_by_frame_and_band[fid][b] > 0:
                    first_gt_ts = ts
                if first_tp_ts is None and tps_by_frame_and_band[fid][b] > 0:
                    first_tp_ts = ts
                if first_gt_ts is not None and first_tp_ts is not None:
                    break

            if first_gt_ts is not None:
                if first_tp_ts is not None:
                    # Delay in seconds from appearance to detection
                    delay_s = max(0.0, first_tp_ts - first_gt_ts) + mean_inference_latency_s
                    clip_delays.append(delay_s)
                    ttfd_results[b]["per_video"][vid] = {
                        "delay_s": round(delay_s, 3),
                        "first_gt_ts": round(first_gt_ts, 3),
                        "first_tp_ts": round(first_tp_ts, 3),
                        "detected": True,
                    }
                else:
                    ttfd_results[b]["per_video"][vid] = {
                        "delay_s": None,
                        "first_gt_ts": round(first_gt_ts, 3),
                        "first_tp_ts": None,
                        "detected": False,
                    }

        if clip_delays:
            ttfd_results[b]["mean_s"] = round(float(np.mean(clip_delays)), 3)

    final_bands_metrics: dict[str, Any] = {}
    total_frames = len(frame_records)
    total_duration_min = (total_frames / float(fps)) / 60.0 if fps > 0 and total_frames > 0 else 1.0

    for b in bands_list:
        tp = counts[b]["tp"]
        fp = counts[b]["fp"]
        fn = counts[b]["fn"]
        n_gt = counts[b]["num_gt"]
        n_frames = counts[b]["num_frames"]

        det_rate = tp / float(tp + fn) if (tp + fn) > 0 else (1.0 if n_gt == 0 else 0.0)
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2.0 * prec * det_rate) / (prec + det_rate) if (prec + det_rate) > 0 else 0.0

        fa_per_min = fp / total_duration_min if total_duration_min > 0 else 0.0
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
    Calibrates confidence threshold tau* maximizing J(tau) = 0.5 * (F1_0_200m + F1_200_400m).
    Tie-breaking favors higher tau to suppress false alarms.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [round(0.10 + 0.05 * i, 2) for i in range(16)]  # 0.10 .. 0.85

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

        f1_near = bands["0-200m"]["f1"]
        f1_far = bands["200-400m"]["f1"]

        valid_f1s = []
        if bands["0-200m"]["num_gt"] > 0:
            valid_f1s.append(f1_near)
        if bands["200-400m"]["num_gt"] > 0:
            valid_f1s.append(f1_far)

        macro_f1 = float(np.mean(valid_f1s)) if valid_f1s else bands["overall"]["f1"]
        score = macro_f1

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
