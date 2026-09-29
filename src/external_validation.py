#!/usr/bin/env python3
"""
External validation for MSHF breast cancer molecular subtype classification.

Loads trained K-fold models and evaluates them on an independent external dataset.
Supports single-fold evaluation and multi-fold ensemble (soft voting).

Usage:
    python src/external_validation.py \
        --checkpoint_root /path/to/training/output \
        --config_path configs/external_config.yaml \
        --backbone ResNet50 \
        --modalities mg,us,clinical \
        --use_seg \
        --output_dir /path/to/results
"""

import argparse
import csv
import datetime
import glob
import json
import os
import re
import sys
from collections import Counter
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project src to path (works when running from repo root)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dataloader.load_data import MyDataset
from models.MSHF_roi_consistency_progressive_sparse import MSHF
from utils.calibration import (
    ConfidenceBinTemperatureCalibrator,
    brier_score,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
)

METRIC_NAMES = ["ACC", "AUC", "F1", "Rec", "SEN", "SPE", "PRE", "ECE", "MCE", "NLL", "Brier"]
THRESHOLD_STRATEGIES = ["argmax", "youden", "fixed"]


# ---------------------------------------------------------------------------
# Threshold strategies (class-imbalance-aware)
# ---------------------------------------------------------------------------

def predict_from_scores(scores, threshold, positive_label=0):
    scores = np.asarray(scores).astype(float)
    negative_label = 1 - positive_label
    return np.where(scores >= threshold, positive_label, negative_label).astype(int)


def find_best_youden_threshold(labels, scores, positive_label=0):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    labels_positive = (labels == positive_label).astype(int)

    if len(np.unique(labels_positive)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(labels_positive, scores)
    valid = np.isfinite(thresholds)
    if not np.any(valid):
        return 0.5

    youden = tpr[valid] - fpr[valid]
    valid_thresholds = thresholds[valid]
    best_idx = int(np.argmax(youden))
    return float(valid_thresholds[best_idx])


def apply_threshold_strategy(
    labels, preds, scores,
    positive_label=0, threshold_strategy="argmax",
    threshold_value=None, logits=None, calibration_bins=15,
):
    if threshold_strategy == "argmax":
        threshold = np.nan
        thresholded_preds = np.asarray(preds).astype(int)
    elif threshold_strategy == "youden":
        threshold = find_best_youden_threshold(labels, scores, positive_label=positive_label)
        thresholded_preds = predict_from_scores(scores, threshold, positive_label=positive_label)
    elif threshold_strategy == "fixed":
        if threshold_value is None:
            raise ValueError("--threshold_value is required when --threshold_strategy=fixed")
        threshold = float(threshold_value)
        thresholded_preds = predict_from_scores(scores, threshold, positive_label=positive_label)
    else:
        raise ValueError(f"Invalid threshold strategy: {threshold_strategy}")

    metrics = compute_binary_metrics(
        labels, thresholded_preds, scores,
        positive_label=positive_label,
        logits=logits,
        calibration_bins=calibration_bins,
    )
    metrics["Threshold_Strategy"] = threshold_strategy
    metrics["Decision_Threshold"] = threshold
    return metrics, thresholded_preds, threshold


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="External validation for MSHF multi-modal classification model"
    )
    # Required
    parser.add_argument(
        "--checkpoint_root", type=str, required=True,
        help="Path to the K-fold training output directory containing fold_*/best_model.pth"
    )
    # Data
    parser.add_argument(
        "--config_path", type=str, default="configs/external_config.yaml",
        help="Path to the external dataset config YAML"
    )
    # Model architecture (must match training)
    parser.add_argument("--backbone", type=str, default="ResNet50",
                        choices=["ResNet50", "DenseNet121", "InceptionV3", "VGG16", "ViT-B_16"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--modalities", type=str, default="mg,us,clinical",
                        help="Comma-separated: mg,us,clinical")
    parser.add_argument("--use_seg", action="store_true",
                        help="Enable mask-guided feature modulation (must match training)")
    parser.add_argument("--disable_cross_modal_fusion", action="store_true",
                        help="Use simple concat+Linear fusion instead of sparse cross-attention")
    # Inference
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1,
                        help="Single fold to evaluate when --no_ensemble is set")
    parser.add_argument("--no_ensemble", action="store_true",
                        help="Evaluate a single fold instead of ensembling all folds")
    # Evaluation
    parser.add_argument("--positive_label", type=int, default=0, choices=[0, 1],
                        help="Positive class for SEN/SPE/AUC. 0=Luminal, 1=non-Luminal")
    parser.add_argument("--calibration_bins", type=int, default=15,
                        help="Number of bins for ECE/MCE/reliability diagrams")
    # Threshold strategy for class-imbalance-aware evaluation
    parser.add_argument(
        "--threshold_strategy",
        type=str.lower,
        default="argmax",
        choices=THRESHOLD_STRATEGIES,
        help="Decision rule for binary metrics. 'youden' finds the ROC threshold maximizing TPR-FPR on the external set. "
             "'fixed' uses a user-specified --threshold_value (e.g., taken from training fold's Youden threshold).",
    )
    parser.add_argument(
        "--threshold_value", type=float, default=None,
        help="Fixed decision threshold on positive_score (only used when --threshold_strategy=fixed). "
             "When omitted, automatically reads fold_*/best_metrics.txt from checkpoint_root and "
             "uses the median Youden threshold across folds.",
    )
    parser.add_argument(
        "--threshold_fold",
        type=int,
        default=None,
        help="Use decision threshold from a specific training fold. "
         "If omitted, use the median threshold across all folds."
    )
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: checkpoint_root/external_val_<timestamp>)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility: metric computation (same logic as training)
# ---------------------------------------------------------------------------

def compute_binary_metrics(labels, preds, scores, positive_label=0, logits=None, calibration_bins=15):
    """Compute classification + calibration metrics. Gracefully handles degenerate cases."""
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    scores = np.asarray(scores).astype(float)

    labels_positive = (labels == positive_label).astype(int)
    preds_positive = (preds == positive_label).astype(int)
    n_classes = len(np.unique(labels))

    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(labels_positive, preds_positive, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    pre = tp / max(tp + fp, 1)
    f1 = f1_score(labels_positive, preds_positive, pos_label=1, zero_division=0)
    bacc = 0.5 * (sen + spe)
    gmean = float(np.sqrt(sen * spe))

    # AUC (degenerate if only one class)
    if n_classes < 2:
        auc = float("nan")
    else:
        try:
            auc = roc_auc_score(labels_positive, scores)
        except ValueError:
            auc = float("nan")

    # Calibration metrics
    if logits is not None:
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        ece = float(expected_calibration_error(logits_t, labels_t, bins=calibration_bins))
        mce = float(maximum_calibration_error(logits_t, labels_t, bins=calibration_bins))
        nll = float(F.cross_entropy(logits_t, labels_t).item())
        brier = float(brier_score(logits_t, labels_t))
    else:
        ece = mce = nll = brier = float("nan")

    return {
        "ACC": acc,
        "AUC": auc if not np.isnan(auc) else 0.0,
        "AUC_raw": auc,
        "F1": f1,
        "Rec": sen,
        "SEN": sen,
        "SPE": spe,
        "PRE": pre,
        "ECE": ece,
        "MCE": mce,
        "NLL": nll,
        "Brier": brier,
        "BACC": bacc,
        "GMEAN": gmean,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Positive_Label": positive_label,
        "N_Classes_Found": n_classes,
    }


def compute_reliability_arrays(logits, labels, positive_label=0):
    """Compute per-sample confidence/uncertainty arrays."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    probs = F.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    positive_scores = probs[:, positive_label]
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    max_entropy = np.log(max(probs.shape[1], 2))
    uncertainty = entropy / max_entropy
    margins = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    correctness = (preds == labels).astype(int)
    labels_positive = (labels == positive_label).astype(int)
    preds_positive = (preds == positive_label).astype(int)
    return {
        "probs": probs,
        "preds": preds,
        "confidences": confidences,
        "positive_scores": positive_scores,
        "entropy": entropy,
        "uncertainty": uncertainty,
        "margins": margins,
        "correct": correctness,
        "labels_positive": labels_positive,
        "preds_positive": preds_positive,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(args, device):
    """Instantiate MSHF model with same architecture as training."""
    modalities = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    model = MSHF(
        num_classes=args.num_classes,
        backbone=args.backbone,
        modalities=modalities,
        use_cross_modal_fusion=not args.disable_cross_modal_fusion,
    ).to(device)
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights, handling both DataParallel-wrapped and bare state_dict."""
    state = torch.load(checkpoint_path, map_location=device)
    if "module." in next(iter(state.keys())):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device, modalities, use_seg):
    """Run inference on a DataLoader, returning logits, labels, and metadata."""
    model.eval()
    all_logits = []
    all_labels = []
    all_patient_ids = []

    for batch in tqdm(loader, desc="Inference"):
        if use_seg:
            (img_cc, mask_cc), (img_mlo, mask_mlo), (img_us, mask_us), labels, clinical = batch
            mask_cc = mask_cc.to(device)
            mask_mlo = mask_mlo.to(device)
            mask_us = mask_us.to(device)
        else:
            img_cc, img_mlo, img_us, labels, clinical = batch
            mask_cc = mask_mlo = mask_us = None

        img_cc = img_cc.to(device)
        img_mlo = img_mlo.to(device)
        img_us = img_us.to(device)
        clinical_feat = clinical.to(device)

        # Model forward: (img_mlo, img_cc, img_us, clinical, mask_mlo, mask_cc, mask_us)
        outputs = model(img_mlo, img_cc, img_us, clinical_feat, mask_mlo, mask_cc, mask_us)

        all_logits.append(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.asarray(all_labels, dtype=int)

    return all_logits, all_labels


# ---------------------------------------------------------------------------
# Calibration & reliability output helpers
# ---------------------------------------------------------------------------

def save_calibration_outputs(logits, labels, save_dir, prefix="external", bins=15):
    """Save calibration tensors, metrics, and reliability reports."""
    os.makedirs(save_dir, exist_ok=True)
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    calibrator = ConfidenceBinTemperatureCalibrator(bins=bins)
    calibrator.fit(logits_t, labels_t)
    calibrated_logits = calibrator.transform(logits_t)

    before = calibration_report(logits_t, labels_t)
    after = calibration_report(logits_t, labels_t, after_logits=calibrated_logits)

    # Save tensors
    torch.save({
        "logits": logits_t,
        "labels": labels_t,
        "calibrated_logits": calibrated_logits,
        "temperatures": calibrator.temperatures,
        "before": before,
        "after": after,
        "bins": bins,
    }, os.path.join(save_dir, f"{prefix}_calibration.pt"))

    # Save metrics
    with open(os.path.join(save_dir, f"{prefix}_calibration_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "before", "after"])
        for metric in before:
            writer.writerow([metric, before[metric], after[metric]])


def save_reliability_csv(logits, labels, save_dir, prefix="external", positive_label=0):
    """Save per-sample confidence/uncertainty CSV."""
    os.makedirs(save_dir, exist_ok=True)
    arrays = compute_reliability_arrays(logits, labels, positive_label=positive_label)
    probs = arrays["probs"]
    labels = np.asarray(labels, dtype=int)

    csv_path = os.path.join(save_dir, f"{prefix}_sample_confidence_uncertainty.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "sample_index", "label", "pred_label", "correct",
            "confidence", "uncertainty", "entropy", "margin",
            "positive_label", "positive_score",
        ] + [f"prob_class_{idx}" for idx in range(probs.shape[1])]
        writer.writerow(header)
        for idx in range(len(labels)):
            writer.writerow([
                idx, labels[idx], arrays["preds"][idx], arrays["correct"][idx],
                arrays["confidences"][idx], arrays["uncertainty"][idx],
                arrays["entropy"][idx], arrays["margins"][idx],
                positive_label, arrays["positive_scores"][idx],
            ] + probs[idx].tolist())
    return csv_path


def save_metrics_txt(metrics, save_path):
    """Write metrics to a human-readable text file."""
    with open(save_path, "w") as f:
        f.write("External Validation Metrics\n")
        f.write("==========================\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")

def summarize_fold_metrics(fold_metrics_list, metric_names):
    """
    Compute Mean / Std / 95% CI across folds.
    """

    summary = []

    n = len(fold_metrics_list)

    for metric in metric_names:

        values = np.array(
            [m[metric] for m in fold_metrics_list],
            dtype=np.float64
        )

        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0.0

        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0

        summary.append({
            "Metric": metric,
            "Mean": mean,
            "Std": std,
            "CI95_Low": mean - ci95,
            "CI95_High": mean + ci95,
        })

    return summary



# ---------------------------------------------------------------------------
# ROC curve output helpers
# ---------------------------------------------------------------------------


def save_external_ensemble_roc(
    labels,
    ensemble_scores,
    save_dir,
    positive_label=0,
    num_points=101,
):
    """
    Save ROC points for the final external Ensemble prediction.

    Ensemble prediction:
        P_ensemble = mean(P_fold1, ..., P_foldN)

    Since this is a single final Ensemble ROC curve, std_tpr is zero.
    The CSV format is kept identical to training mean_fold_roc_points.csv:
        mean_fpr, mean_tpr, std_tpr, tpr_low, tpr_high
    """

    labels = np.asarray(labels).astype(int)
    ensemble_scores = np.asarray(ensemble_scores).astype(float)

    labels_positive = (labels == positive_label).astype(int)

    if len(np.unique(labels_positive)) < 2:
        print("Skipped external Ensemble ROC: labels contain only one class.")
        return

    # Original ROC
    fpr, tpr, thresholds = roc_curve(
        labels_positive,
        ensemble_scores
    )

    roc_auc = roc_auc_score(
        labels_positive,
        ensemble_scores
    )

    # Same 101-point interpolation strategy as training
    mean_fpr = np.linspace(
        0.0,
        1.0,
        num_points
    )

    mean_tpr = np.interp(
        mean_fpr,
        fpr,
        tpr
    )

    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    # Only ONE Ensemble ROC exists.
    # Therefore there is no Fold-to-Fold standard deviation here.
    std_tpr = np.zeros_like(mean_tpr)

    tpr_low = np.clip(
        mean_tpr - std_tpr,
        0.0,
        1.0
    )

    tpr_high = np.clip(
        mean_tpr + std_tpr,
        0.0,
        1.0
    )

    csv_path = os.path.join(
        save_dir,
        "external_ensemble_roc_points.csv"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "mean_fpr",
            "mean_tpr",
            "std_tpr",
            "tpr_low",
            "tpr_high"
        ])

        for fp, tp, std, low, high in zip(
            mean_fpr,
            mean_tpr,
            std_tpr,
            tpr_low,
            tpr_high
        ):
            writer.writerow([
                fp,
                tp,
                std,
                low,
                high
            ])

    print(
        f"Saved external Ensemble ROC points: {csv_path}"
    )

    print(
        f"External Ensemble AUC: {roc_auc:.6f}"
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": roc_auc,
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "tpr_low": tpr_low,
        "tpr_high": tpr_high,
    }


def save_external_mean_fold_roc(
    fold_logits_dict,
    folds_to_eval,
    save_dir,
    positive_label=0,
    num_points=101,
):
    """
    Calculate and save Mean ROC across external validation folds.

    Each trained Fold model is evaluated independently on the SAME
    external dataset.

    Fold 1 -> ROC_1
    Fold 2 -> ROC_2
    ...
    Fold N -> ROC_N

    Then interpolate all ROC curves to the same 101 FPR points and
    calculate Mean / Std exactly as in training.
    """

    mean_fpr = np.linspace(
        0.0,
        1.0,
        num_points
    )

    interp_tprs = []
    fold_aucs = []

    for fold_idx in folds_to_eval:

        if fold_idx not in fold_logits_dict:
            continue

        fold_data = fold_logits_dict[fold_idx]

        labels = np.asarray(
            fold_data["labels"]
        ).astype(int)

        probs = np.asarray(
            fold_data["probs"]
        ).astype(float)

        scores = probs[:, positive_label]

        labels_positive = (
            labels == positive_label
        ).astype(int)

        if len(np.unique(labels_positive)) < 2:
            print(
                f"Skipped fold {fold_idx} in external mean ROC: "
                f"labels contain only one class."
            )
            continue

        # Original Fold ROC
        fpr, tpr, thresholds = roc_curve(
            labels_positive,
            scores
        )

        fold_auc = roc_auc_score(
            labels_positive,
            scores
        )

        # Same interpolation method as training
        interp_tpr = np.interp(
            mean_fpr,
            fpr,
            tpr
        )

        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        interp_tprs.append(interp_tpr)
        fold_aucs.append(fold_auc)

    if not interp_tprs:
        print(
            "Skipped external mean Fold ROC: "
            "no valid fold ROC curves."
        )
        return

    interp_tprs = np.asarray(
        interp_tprs,
        dtype=float
    )

    fold_aucs = np.asarray(
        fold_aucs,
        dtype=float
    )

    # ---------------------------------------------------------------
    # Exactly the same calculation as training mean_fold_roc_points
    # ---------------------------------------------------------------

    mean_tpr = np.mean(
        interp_tprs,
        axis=0
    )

    if len(interp_tprs) > 1:
        std_tpr = np.std(
            interp_tprs,
            axis=0,
            ddof=1
        )
    else:
        std_tpr = np.zeros_like(
            mean_tpr
        )

    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    tpr_low = np.clip(
        mean_tpr - std_tpr,
        0.0,
        1.0
    )

    tpr_high = np.clip(
        mean_tpr + std_tpr,
        0.0,
        1.0
    )

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------

    csv_path = os.path.join(
        save_dir,
        "external_mean_fold_roc_points.csv"
    )

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "mean_fpr",
            "mean_tpr",
            "std_tpr",
            "tpr_low",
            "tpr_high"
        ])

        for fp, tp, std, low, high in zip(
            mean_fpr,
            mean_tpr,
            std_tpr,
            tpr_low,
            tpr_high
        ):
            writer.writerow([
                fp,
                tp,
                std,
                low,
                high
            ])

    # ---------------------------------------------------------------
    # Save Fold AUCs separately
    # ---------------------------------------------------------------

    auc_csv_path = os.path.join(
        save_dir,
        "external_fold_auc.csv"
    )

    with open(auc_csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "fold",
            "auc"
        ])

        valid_folds = [
            fi for fi in folds_to_eval
            if fi in fold_logits_dict
        ]

        for fold_idx, fold_auc in zip(
            valid_folds,
            fold_aucs
        ):
            writer.writerow([
                fold_idx,
                fold_auc
            ])

    print(
        f"Saved external Mean-Fold ROC points: {csv_path}"
    )

    print(
        f"Saved external Fold AUC values: {auc_csv_path}"
    )

    print(
        f"External Fold AUC Mean: "
        f"{np.mean(fold_aucs):.6f}"
    )

    if len(fold_aucs) > 1:
        print(
            f"External Fold AUC Std: "
            f"{np.std(fold_aucs, ddof=1):.6f}"
        )

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "tpr_low": tpr_low,
        "tpr_high": tpr_high,
        "fold_aucs": fold_aucs,
    }



# ---------------------------------------------------------------------------
# Auto-detect threshold from training fold outputs
# ---------------------------------------------------------------------------

def auto_detect_threshold(checkpoint_root, threshold_fold=None):#选择threshold_fold指定的阈值

    # ============================================================
    # Case 1: explicitly use one fold's threshold
    # ============================================================
    if threshold_fold is not None:

        metrics_file = os.path.join(
            checkpoint_root,
            f"fold_{threshold_fold}",
            "best_metrics.txt"
        )

        if not os.path.exists(metrics_file):
            print(
                f"WARNING: {metrics_file} not found. "
                f"Cannot read threshold from fold_{threshold_fold}."
            )
            return None
        with open(metrics_file, "r") as f:
            for line in f:
                m = re.search(
                    r"Decision threshold:\s*([\d.]+)",
                    line
                )
                if m:
                    threshold = float(m.group(1))
                    print(
                        f"Using threshold from fold_{threshold_fold}: "
                        f"{threshold:.4f}"
                    )
                    return threshold
        print(
            f"WARNING: Decision threshold not found in "
            f"{metrics_file}."
        )
        return None

    # ============================================================
    # Case 2: threshold_fold is None
    #         automatically use median across all folds
    # ============================================================
    pattern = os.path.join(
        checkpoint_root,
        "fold_*",
        "best_metrics.txt"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        print(
            f"WARNING: No fold_*/best_metrics.txt found in "
            f"{checkpoint_root}."
        )
        return None
    thresholds = []
    threshold_folds = []
    for fpath in files:
        # Extract fold number from path
        fold_match = re.search(
            r"fold_(\d+)",
            fpath
        )
        fold_name = (
            f"fold_{fold_match.group(1)}"
            if fold_match
            else os.path.basename(os.path.dirname(fpath))
        )
        with open(fpath, "r") as f:
            for line in f:
                m = re.search(
                    r"Decision threshold:\s*([\d.]+)",
                    line
                )
                if m:
                    thresholds.append(float(m.group(1)))
                    threshold_folds.append(fold_name)
                    break
    if not thresholds:
        print(
            "WARNING: Could not parse Decision threshold "
            "from any best_metrics.txt files."
        )
        return None
    med = median(thresholds)
    print(
        f"Auto-detected thresholds from "
        f"{len(thresholds)} folds: "
        f"{[f'{t:.4f}' for t in thresholds]}"
    )
    print(
        f"Threshold folds: {threshold_folds}"
    )
    print(
        f"Median threshold: {med:.4f}"
    )
    return med


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(f"DEBUG threshold_value    = {args.threshold_value}")
    # --- Auto-detect threshold if needed ---
    if args.threshold_strategy == "fixed" and args.threshold_value is None:
        detected = auto_detect_threshold(
            args.checkpoint_root,
            args.threshold_fold
        )
        if detected is not None:
            args.threshold_value = detected
        else:
            print("Falling back to threshold_strategy=argmax.")
            args.threshold_strategy = "argmax"

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Setup output directory ---
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.checkpoint_root, f"external_val_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save run arguments
    args_path = os.path.join(args.output_dir, "run_arguments.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"Output directory: {args.output_dir}")

    # --- Load external clinical data ---
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    clinical_path = config.get("clinical_dir", "")
    if not os.path.isabs(clinical_path):
        clinical_path = os.path.join(os.path.dirname(args.config_path), os.path.basename(clinical_path))
    with open(clinical_path, "r", encoding="utf-8") as f:
        clinical_infos = json.load(f)

    labels_all = [info["label"] for info in clinical_infos]
    print(f"External dataset: {len(clinical_infos)} samples")
    print(f"Label distribution: {dict(sorted(Counter(labels_all).items()))}")

    if len(np.unique(labels_all)) < 2:
        print("WARNING: External dataset contains only one class. "
              "AUC, sensitivity/specificity may be degenerate.")

    # --- DataLoader ---
    modalities_list = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    dataset = MyDataset(clinical_infos, args.config_path, use_seg=args.use_seg, is_train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"DataLoader: {len(dataset)} samples, batch_size={args.batch_size}")

    # --- Determine folds to evaluate ---
    if args.no_ensemble:
        folds_to_eval = [args.fold]
        print(f"Single-fold evaluation: fold {args.fold}")
    else:
        folds_to_eval = list(range(1, args.num_folds + 1))
        print(f"Ensemble evaluation: folds {folds_to_eval}")

    # --- Run inference per fold ---
    all_fold_probs = []
    fold_logits_dict = {}
    fold_metrics_all = []

    for fold_idx in folds_to_eval:
        fold_dir = os.path.join(args.checkpoint_root, f"fold_{fold_idx}")
        checkpoint_path = os.path.join(fold_dir, "best_model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: {checkpoint_path} not found, trying fallback_model.pth")
            checkpoint_path = os.path.join(fold_dir, "fallback_model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: No checkpoint found for fold {fold_idx}, skipping.")
            continue

        print(f"\n--- Fold {fold_idx} ---")
        print(f"Loading: {checkpoint_path}")

        model = build_model(args, device)
        model = load_checkpoint(model, checkpoint_path, device)

        logits, labels = run_inference(model, loader, device, modalities_list, args.use_seg)
        probs = F.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()

        all_fold_probs.append(probs)
        fold_logits_dict[fold_idx] = {
            "logits": logits,
            "probs": probs,
            "labels": labels,
        }

        # Per-fold metrics (informational)
        fold_preds_raw = np.argmax(probs, axis=1)
        fold_scores = probs[:, args.positive_label]
        fold_metrics, fold_preds, _ = apply_threshold_strategy(
            labels, fold_preds_raw, fold_scores,
            positive_label=args.positive_label,
            threshold_strategy=args.threshold_strategy,
            threshold_value=args.threshold_value,
            logits=logits,
            calibration_bins=args.calibration_bins,
        )
        # Store thresholded predictions for later use
        fold_metrics_all.append(fold_metrics)
        fold_logits_dict[fold_idx]["preds"] = fold_preds
        print(f"Fold {fold_idx} ACC={fold_metrics['ACC']:.4f}  "
              f"AUC={fold_metrics['AUC_raw']:.4f}  "
              f"SEN={fold_metrics['SEN']:.4f}  "
              f"SPE={fold_metrics['SPE']:.4f}  "
              f"ECE={fold_metrics['ECE']:.4f}")

    if not all_fold_probs:
        print("ERROR: No valid folds found. Aborting.")
        sys.exit(1)

    # --- Ensemble (soft voting) ---
    print(f"\n--- Ensemble ({len(all_fold_probs)} folds, soft voting) ---")
    ensemble_probs = np.mean(all_fold_probs, axis=0)
    ensemble_logits = np.log(np.clip(ensemble_probs, 1e-12, 1.0))  # approx inverse softmax for calibration
    ensemble_probs = np.mean(all_fold_probs,axis=0)
    ensemble_logits = np.log(np.clip(ensemble_probs,1e-12,1.0))

    # Use labels from any fold (they are the same)
    final_labels = fold_logits_dict[folds_to_eval[0]]["labels"]
    final_preds_raw = np.argmax(ensemble_probs, axis=1)
    final_scores = ensemble_probs[:, args.positive_label]

    final_labels = fold_logits_dict[
        folds_to_eval[0]
    ]["labels"]

    final_preds_raw = np.argmax(
        ensemble_probs,
        axis=1
    )

    final_scores = ensemble_probs[
        :, args.positive_label
    ]

    save_external_ensemble_roc(
        labels=final_labels,
        ensemble_scores=final_scores,
        save_dir=args.output_dir,
        positive_label=args.positive_label,
        num_points=101,
    )

    save_external_mean_fold_roc(
        fold_logits_dict=fold_logits_dict,
        folds_to_eval=folds_to_eval,
        save_dir=args.output_dir,
        positive_label=args.positive_label,
        num_points=101,
    )

    # --- Apply threshold strategy (argmax or youden) ---
    final_metrics, final_preds, decision_threshold = apply_threshold_strategy(
        final_labels, final_preds_raw, final_scores,
        positive_label=args.positive_label,
        threshold_strategy=args.threshold_strategy,
        threshold_value=args.threshold_value,
        logits=ensemble_logits,
        calibration_bins=args.calibration_bins,
    )

    print(f"\n=== Ensemble Results (strategy={args.threshold_strategy}) ===")
    if args.threshold_strategy == "youden":
        print(f"  Optimal threshold: {decision_threshold:.4f}")
    for m in METRIC_NAMES:
        val = final_metrics[m]
        if isinstance(val, float):
            print(f"  {m:8s}: {val:.6f}")
        else:
            print(f"  {m:8s}: {val}")
    print(f"  {'BACC':8s}: {final_metrics['BACC']:.6f}")
    print(f"  {'GMEAN':8s}: {final_metrics['GMEAN']:.6f}")
    print(f"  N_Classes:  {final_metrics['N_Classes_Found']}")
    print(f"  TN={final_metrics['TN']} FP={final_metrics['FP']} FN={final_metrics['FN']} TP={final_metrics['TP']}")

    # --- Save outputs ---
    out = args.output_dir

    # Metrics txt
    save_metrics_txt(final_metrics, os.path.join(out, "metrics.txt"))

    # Metrics CSV
    with open(os.path.join(out, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for m in METRIC_NAMES + ["BACC", "GMEAN", "TN", "FP", "FN", "TP", "N_Classes_Found"]:
            writer.writerow([m, final_metrics.get(m, "N/A")])

    # Per-sample predictions
    arrays = compute_reliability_arrays(ensemble_logits, final_labels, positive_label=args.positive_label)
    with open(os.path.join(out, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        n_probs = ensemble_probs.shape[1]
        writer.writerow(
            ["sample_index", "label", "pred_label", "correct",
             "confidence", "uncertainty", "entropy", "margin",
             "positive_score"]
            + [f"prob_class_{i}" for i in range(n_probs)]
            + [f"fold_{fi}_prob_class_{i}" for fi in folds_to_eval for i in range(n_probs)]
        )
        for idx in range(len(final_labels)):
            fold_extra = []
            for fi in folds_to_eval:
                fold_extra.extend(fold_logits_dict[fi]["probs"][idx].tolist())
            writer.writerow([
                idx, int(final_labels[idx]), int(arrays["preds"][idx]), int(arrays["correct"][idx]),
                float(arrays["confidences"][idx]), float(arrays["uncertainty"][idx]),
                float(arrays["entropy"][idx]), float(arrays["margins"][idx]),
                float(arrays["positive_scores"][idx]),
            ] + ensemble_probs[idx].tolist() + fold_extra)

    # Calibration & reliability
    save_calibration_outputs(ensemble_logits, final_labels, out, prefix="ensemble", bins=args.calibration_bins)
    save_reliability_csv(ensemble_logits, final_labels, out, prefix="ensemble", positive_label=args.positive_label)

    # Per-fold metrics comparison
    with open(os.path.join(out, "fold_comparison.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold"] + METRIC_NAMES + ["BACC", "GMEAN", "Threshold_Strategy", "Decision_Threshold"])
        for fi in folds_to_eval:
            fd = fold_logits_dict[fi]
            fp = fd.get("preds", np.argmax(fd["probs"], axis=1))
            fs = fd["probs"][:, args.positive_label]
            fm, _, ft = apply_threshold_strategy(
                fd["labels"], fp, fs,
                positive_label=args.positive_label,
                threshold_strategy=args.threshold_strategy,
                threshold_value=args.threshold_value,
                logits=fd["logits"],
                calibration_bins=args.calibration_bins,
            )
            writer.writerow(
                [fi] + [fm[m] for m in METRIC_NAMES] + [fm["BACC"], fm["GMEAN"], args.threshold_strategy, ft]
            )
    
    summary = summarize_fold_metrics(
    fold_metrics_all,
    METRIC_NAMES + ["BACC", "GMEAN"]
    )

    summary_csv = os.path.join(out, "summary_95ci.csv")

    with open(summary_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Metric",
            "Mean",
            "Std",
            "CI95_Low",
            "CI95_High"
        ])

        for row in summary:

            writer.writerow([
                row["Metric"],
                row["Mean"],
                row["Std"],
                row["CI95_Low"],
                row["CI95_High"],
            ])

    print(f"\nAll results saved to: {out}")


if __name__ == "__main__":
    main()
