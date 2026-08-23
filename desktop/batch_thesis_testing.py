# -*- coding: utf-8 -*-
"""
Batch Thesis Testing for SpectraMatch.

Runs the 100 reference/sample thesis dataset through the existing alignment,
color, and pattern analysis code, then writes only:
  1. one structured JSON file with numeric results
  2. one 100-page heatmap PDF
"""

import copy
import json
import os
import time
import traceback
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from modules.ColorUnitBackend import analyze_color, _deltaE2000_array, _bgr_pixels_to_lab
from modules.ImageAlignmentBackend import apply_alignment
from modules.PatternUnitBackend import (
    DEFAULT_CONFIG as PATTERN_DEFAULT_CONFIG,
    method1_structural_ssim,
    method3_gradient_similarity,
    method6_phase_correlation,
    create_gradient_red_boundaries,
    create_phase_red_boundaries,
    structural_difference_analysis,
    fourier_domain_analysis,
    glcm_texture_analysis,
)


TOTAL_PAIRS = 100


def run_batch_thesis_test(project_dir, current_settings=None, current_region_data=None, progress_callback=None):
    """Run the 100-pair batch test and return output paths."""
    current_settings = current_settings or {}
    current_region_data = current_region_data or {}
    dataset_dir = os.path.join(project_dir, "100TESTKLASORU")
    ref_dir = _first_existing_dir(
        dataset_dir,
        ("referens", "REFERENS", "REFERENS GORUNTULERI", "Referens Goruntuleri"),
    )
    sample_dir = _first_existing_dir(
        dataset_dir,
        ("numune", "NUMUNE", "NUMUNE GORUNTULERI", "Numune Goruntuleri"),
    )
    results_root = _first_existing_or_create_dir(
        dataset_dir,
        ("sonuclar", "SONUCLAR", "SONUCLAR", "sonuçlar"),
        preferred="sonuclar",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"Batch_Results_{timestamp}_Sonuc_Ozeti")
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, f"Batch_Results_{timestamp}.json")
    pdf_path = os.path.join(run_dir, f"Batch_Results_{timestamp}_isi_haritalari.pdf")

    report_lang = current_settings.get("report_lang", "en")
    settings = _build_batch_settings(current_settings)
    alignment_mode = _normalize_alignment_mode(settings.get("alignment_mode", "direct"))
    settings["alignment_mode"] = alignment_mode

    if progress_callback:
        progress_callback(0, "Initializing 100-sample batch test...")

    batch_doc = {
        "Thesis_Metadata": _build_thesis_metadata(timestamp, dataset_dir, ref_dir, sample_dir),
        "Runtime_Configuration": _build_runtime_configuration(
            timestamp,
            settings,
            current_region_data,
            dataset_dir,
            ref_dir,
            sample_dir,
            results_root,
        ),
        "Batch_Run": {
            "timestamp": timestamp,
            "output_folder": run_dir,
            "json_file": os.path.basename(json_path),
            "heatmap_pdf_file": os.path.basename(pdf_path),
            "alignment_mode": alignment_mode,
            "report_language": report_lang,
            "operator": settings.get("operator", "Operator"),
            "software": "SpectraMatch",
            "expected_pair_count": TOTAL_PAIRS,
            "region_data": current_region_data,
        },
        "Test_Results": [],
    }

    with PdfPages(pdf_path) as pdf:
        for pair_no in range(1, TOTAL_PAIRS + 1):
            status = "PASS" if pair_no <= 50 else "FAIL"
            alignment_tag = "Alignment_Issue_Tested" if pair_no >= 90 else None
            if progress_callback:
                progress_callback(
                    pair_no - 1,
                    f"Processing pair {pair_no} of 100... Generating Heatmap...",
                )

            ref_path = os.path.join(ref_dir, f"{pair_no}.png")
            sample_path = os.path.join(sample_dir, f"{pair_no}.png")
            started = time.time()

            try:
                result, heatmap = _process_pair(
                    pair_no,
                    ref_path,
                    sample_path,
                    settings,
                    current_region_data,
                    status,
                    alignment_tag,
                )
            except Exception as exc:
                traceback.print_exc()
                result = {
                    "sample_no": pair_no,
                    "ground_truth": {"status": status, "tag": alignment_tag},
                    "reference_image": {"file_name": f"{pair_no}.png", "path": ref_path},
                    "sample_image": {"file_name": f"{pair_no}.png", "path": sample_path},
                    "success": False,
                    "error": str(exc),
                    "processing_time_seconds": round(time.time() - started, 4),
                }
                heatmap = np.zeros((80, 80), dtype=np.float32)

            batch_doc["Test_Results"].append(result)
            _append_heatmap_page(pdf, heatmap, pair_no, status, alignment_tag)

            if progress_callback:
                progress_callback(pair_no, f"Completed pair {pair_no} of 100")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(batch_doc), f, indent=2, ensure_ascii=False)

    return {
        "success": True,
        "message": "100-sample batch thesis test completed successfully.",
        "output_folder": run_dir,
        "json_path": json_path,
        "pdf_path": pdf_path,
        "pairs_processed": TOTAL_PAIRS,
    }


def _process_pair(pair_no, ref_path, sample_path, settings, region_data, status, alignment_tag):
    started = time.time()
    started_iso = datetime.now().isoformat(timespec="seconds")
    ref_img = _imread_unicode(ref_path)
    sample_img = _imread_unicode(sample_path)
    if ref_img is None:
        raise FileNotFoundError(f"Cannot read reference image: {ref_path}")
    if sample_img is None:
        raise FileNotFoundError(f"Cannot read sample image: {sample_path}")

    ref_img = _ensure_bgr(ref_img)
    sample_img = _ensure_bgr(sample_img)
    ref_original = ref_img.copy()
    sample_original = sample_img.copy()

    analysis_ref = _crop_for_batch(ref_img, region_data)
    analysis_sample = _crop_for_batch(sample_img, region_data)

    alignment_mode = _normalize_alignment_mode(settings.get("alignment_mode", "direct"))
    alignment = apply_alignment(
        analysis_ref.copy(),
        analysis_sample.copy(),
        mode=alignment_mode,
        region_data=region_data,
    )
    aligned_sample = _ensure_bgr(alignment.get("aligned_sample", analysis_sample))
    analysis_ref = _ensure_bgr(alignment.get("ref_cropped", analysis_ref))
    if aligned_sample.shape[:2] != analysis_ref.shape[:2]:
        aligned_sample = cv2.resize(aligned_sample, (analysis_ref.shape[1], analysis_ref.shape[0]))

    color_results = analyze_color(analysis_ref, aligned_sample, settings)
    pattern_results = _run_pattern_numbers(analysis_ref, aligned_sample, settings)
    heatmap = _delta_e_heatmap(analysis_ref, aligned_sample)

    result = {
        "sample_no": pair_no,
        "ground_truth": {"status": status, "tag": alignment_tag},
        "success": True,
        "reference_image": _image_metadata(ref_path, ref_original),
        "sample_image": _image_metadata(sample_path, sample_original),
        "analysis_image_metadata": {
            "reference_after_alignment": _array_metadata(analysis_ref),
            "sample_after_alignment": _array_metadata(aligned_sample),
            "region_data_applied": region_data if region_data else {"type": "full"},
        },
        "alignment": {
            "mode": alignment_mode,
            "method": alignment.get("method"),
            "transform_matrix": alignment.get("transform_matrix"),
            "metrics": alignment.get("metrics", {}),
        },
        "color_analysis": _summarize_color(color_results),
        "pattern_analysis": pattern_results,
        "heatmap_statistics": _map_stats(heatmap),
        "algorithm_decisions": {
            "ground_truth_status": status,
            "color_status": color_results.get("overall_status"),
            "csi_status": color_results.get("csi_details", {}).get("status"),
            "pattern_status": pattern_results.get("final_status"),
            "batch_alignment_tag": alignment_tag,
        },
        "runtime": {
            "operator": settings.get("operator", "Operator"),
            "report_language": settings.get("report_lang", "en"),
            "sampling_mode": settings.get("sampling_mode"),
            "sampling_count": settings.get("region_count"),
            "started_at": started_iso,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
        "processing_time_seconds": round(time.time() - started, 4),
    }
    return result, heatmap


def _run_pattern_numbers(ref_img, sample_img, settings):
    cfg = copy.deepcopy(PATTERN_DEFAULT_CONFIG)
    cfg.update(settings or {})
    sections = cfg.get("sections", {})

    ssim_score, ssim_map = method1_structural_ssim(ref_img, sample_img)
    grad_score, _grad_img, grad_data = method3_gradient_similarity(
        ref_img,
        sample_img,
        diff_sensitivity=cfg.get("gradient_diff_sensitivity", 7),
    )
    phase_score, _phase_img, phase_data = method6_phase_correlation(
        ref_img,
        sample_img,
        diff_sensitivity=cfg.get("phase_diff_sensitivity", 7),
    )
    grad_boundary = create_gradient_red_boundaries(
        sample_img,
        grad_data,
        sensitivity=cfg.get("gradient_boundary_sensitivity", 4),
    )
    phase_boundary = create_phase_red_boundaries(
        sample_img,
        phase_data,
        sensitivity=cfg.get("phase_boundary_sensitivity", 8),
    )
    structural = structural_difference_analysis(
        ref_img,
        sample_img,
        sensitivity=cfg.get("structural_sensitivity", 10),
    )

    fourier = {}
    try:
        fourier = {
            "reference": _strip_arrays(fourier_domain_analysis(ref_img)),
            "sample": _strip_arrays(fourier_domain_analysis(sample_img)),
        }
    except Exception as exc:
        fourier = {"error": str(exc)}

    glcm = {}
    if sections.get("glcm", True):
        try:
            ref_glcm = glcm_texture_analysis(ref_img)
            sam_glcm = glcm_texture_analysis(sample_img)
            glcm = {
                "reference_properties": ref_glcm.get("properties", {}),
                "sample_properties": sam_glcm.get("properties", {}),
                "reference_glcm_stats": _map_stats(ref_glcm.get("glcm_matrix")),
                "sample_glcm_stats": _map_stats(sam_glcm.get("glcm_matrix")),
            }
        except Exception as exc:
            glcm = {"error": str(exc)}

    scores = {
        "Structural SSIM": float(ssim_score),
        "Gradient Similarity": float(grad_score),
        "Phase Correlation": float(phase_score),
        "Structural Match": float(structural.get("similarity_score", 0.0)),
    }
    normalized = {
        "Structural SSIM": max(0.0, min(100.0, float(ssim_score) * 100.0)),
        "Gradient Similarity": float(grad_score),
        "Phase Correlation": float(phase_score),
        "Structural Match": float(structural.get("similarity_score", 0.0)),
    }
    composite = float(np.mean(list(normalized.values()))) if normalized else 0.0
    global_thr = float(cfg.get("global_threshold", 75.0))
    final_status = "PASS" if composite >= global_thr else ("CONDITIONAL" if composite >= global_thr - 15 else "FAIL")

    return {
        "raw_metrics": scores,
        "normalized_scores": normalized,
        "composite_score": composite,
        "final_status": final_status,
        "ssim_heatmap_stats": _image_channel_stats(ssim_map),
        "gradient_boundary_metrics": {
            "changed_area_pct": float(grad_boundary[2]),
            "weighted_severity": float(grad_boundary[3]),
            "region_count": int(grad_boundary[4]),
        },
        "phase_boundary_metrics": {
            "changed_area_pct": float(phase_boundary[2]),
            "weighted_severity": float(phase_boundary[3]),
            "region_count": int(phase_boundary[4]),
        },
        "structural_results": _strip_arrays(structural),
        "fourier_analysis": fourier,
        "glcm_analysis": glcm,
    }


def _delta_e_heatmap(ref_img, sample_img):
    if sample_img.shape[:2] != ref_img.shape[:2]:
        sample_img = cv2.resize(sample_img, (ref_img.shape[1], ref_img.shape[0]))
    ref_pixels = ref_img[:, :, :3].reshape(-1, 3)
    sample_pixels = sample_img[:, :, :3].reshape(-1, 3)
    ref_lab = _bgr_pixels_to_lab(ref_pixels, illuminant="D65")
    sample_lab = _bgr_pixels_to_lab(sample_pixels, illuminant="D65")
    values = _deltaE2000_array(ref_lab, sample_lab)
    return values.reshape(ref_img.shape[:2]).astype(np.float32)


def _append_heatmap_page(pdf, heatmap, pair_no, status, alignment_tag):
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=130)
    fig.patch.set_facecolor("white")
    ax.set_axis_off()

    title_color = "#1f2937"
    fig.text(0.5, 0.955, f"Numune No: {pair_no}", ha="center", va="top",
             fontsize=16, fontweight="bold", color=title_color)
    fig.text(0.5, 0.925, f"Durum: {status}", ha="center", va="top",
             fontsize=14, fontweight="bold", color=("#15803d" if status == "PASS" else "#b91c1c"))
    if alignment_tag:
        fig.text(0.5, 0.895, "Not: Hizalama Sorunu Test Edildi (Alignment Issue Tested)",
                 ha="center", va="top", fontsize=11, fontweight="bold", color="#dc2626")

    plot_ax = fig.add_axes([0.08, 0.18, 0.84, 0.65])
    im = plot_ax.imshow(heatmap, cmap="inferno")
    plot_ax.set_title("Delta E Difference Heatmap / Isı Haritası", fontsize=12, pad=10)
    plot_ax.set_axis_off()
    cbar = fig.colorbar(im, ax=plot_ax, fraction=0.046, pad=0.03)
    cbar.set_label("Delta E 2000", fontsize=9)

    fig.text(0.5, 0.08, "SpectraMatch Batch Thesis Testing", ha="center",
             fontsize=9, color="#64748b")
    pdf.savefig(fig)
    plt.close(fig)


def _summarize_color(color_results):
    return {
        "mean_de00": color_results.get("mean_de00"),
        "overall_status": color_results.get("overall_status"),
        "csi_value": color_results.get("csi_value"),
        "csi_details": color_results.get("csi_details", {}),
        "color_averages": color_results.get("color_averages", {}),
        "sampling_radius_px": color_results.get("r"),
        "sampling_points": color_results.get("points", []),
        "region_statistics": color_results.get("reg_stats", []),
    }


def _build_batch_settings(current_settings):
    settings = copy.deepcopy(current_settings or {})
    settings["alignment_mode"] = _normalize_alignment_mode(settings.get("alignment_mode", "direct"))
    settings.setdefault("report_lang", current_settings.get("report_lang", "en") if current_settings else "en")
    settings.setdefault("region_count", 5)
    sections = settings.setdefault("sections", {})
    sections.setdefault("glcm", True)
    return settings


def _normalize_alignment_mode(mode):
    mode = str(mode or "direct").strip()
    return mode if mode in {"direct", "ai_smart_match", "bestch"} else "direct"


def _build_runtime_configuration(timestamp, settings, region_data, dataset_dir, ref_dir, sample_dir, results_root):
    pair_count = _count_available_pairs(ref_dir, sample_dir)
    return {
        "generated_timestamp": timestamp,
        "operator": settings.get("operator", "Operator"),
        "report_language": settings.get("report_lang", "en"),
        "timezone_offset": settings.get("timezone_offset"),
        "alignment": {
            "mode": settings.get("alignment_mode", "direct"),
            "source": "Desktop runtime settings",
        },
        "sampling": {
            "mode": settings.get("sampling_mode"),
            "requested_count": settings.get("region_count"),
            "points": settings.get("sampling_points", []),
        },
        "region_data": region_data if region_data else {"type": "full"},
        "dataset": {
            "dataset_directory": dataset_dir,
            "reference_directory": ref_dir,
            "sample_directory": sample_dir,
            "results_directory": results_root,
            "expected_pair_count": TOTAL_PAIRS,
            "available_pair_count": pair_count,
        },
        "settings_snapshot": settings,
    }


def _count_available_pairs(ref_dir, sample_dir):
    count = 0
    for pair_no in range(1, TOTAL_PAIRS + 1):
        if os.path.exists(os.path.join(ref_dir, f"{pair_no}.png")) and os.path.exists(os.path.join(sample_dir, f"{pair_no}.png")):
            count += 1
    return count


def _crop_for_batch(image, region_data):
    if not region_data or region_data.get("type") == "full" or not region_data.get("use_crop"):
        return image
    try:
        from app import crop_image
        return _ensure_bgr(crop_image(image, region_data))
    except Exception as exc:
        print(f"[BatchThesis] WARNING: region crop skipped: {exc}")
        return image


def _build_thesis_metadata(timestamp, dataset_dir, ref_dir, sample_dir):
    return {
        "Experiment_Name": "SpectraMatch 100-Sample Batch Thesis Testing",
        "Generated_Timestamp": timestamp,
        "Dataset_Directory": dataset_dir,
        "Reference_Directory": ref_dir,
        "Sample_Directory": sample_dir,
        "Ground_Truth": {
            "Samples 1 to 50": {
                "Status": "PASS",
                "Description": "Perfect samples, identical to reference.",
            },
            "Samples 51 to 100": {
                "Status": "FAIL",
                "Description": "Defective in color, pattern, or both.",
            },
            "Samples 90 to 100": {
                "Tag": "Alignment_Issue_Tested",
                "Description": "These specifically test the AISMARTMATCH alignment capabilities.",
            },
        },
    }


def _first_existing_dir(parent, names):
    for name in names:
        path = os.path.join(parent, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"None of these folders were found in {parent}: {', '.join(names)}")


def _first_existing_or_create_dir(parent, names, preferred):
    for name in names:
        path = os.path.join(parent, name)
        if os.path.isdir(path):
            return path
    path = os.path.join(parent, preferred)
    os.makedirs(path, exist_ok=True)
    return path


def _ensure_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img[:, :, :3].copy()
    return img[:, :, :3].copy()


def _imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def _image_metadata(path, img):
    meta = _array_metadata(img)
    meta.update({"file_name": os.path.basename(path), "path": path})
    return meta


def _array_metadata(img):
    h, w = img.shape[:2]
    channels = int(img.shape[2]) if img.ndim == 3 else 1
    return {"width": int(w), "height": int(h), "channels": channels}


def _map_stats(arr):
    if arr is None:
        return {}
    arr = np.asarray(arr)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "shape": [int(x) for x in arr.shape],
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _image_channel_stats(img):
    arr = np.asarray(img)
    return {
        "shape": [int(x) for x in arr.shape],
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _strip_arrays(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if isinstance(item, np.ndarray):
                cleaned[key + "_stats"] = _map_stats(item)
            elif key.endswith("_img") or key.endswith("_path") or key in {"subplot_img", "diff_img"}:
                continue
            else:
                cleaned[key] = _strip_arrays(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_strip_arrays(v) for v in value]
    return value


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim <= 1 and value.size <= 16:
            return [_json_ready(v) for v in value.tolist()]
        return _map_stats(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (bytes, bytearray)):
        return {"byte_length": len(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
