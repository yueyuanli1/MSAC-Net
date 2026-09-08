import argparse
import csv
import datetime
import json
import os
import random
import shutil
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
MONITOR_CHOICES = ["ACC", "AUC", "F1", "REC", "SEN", "SPE", "PRE", "ACC_AUC", "AUC_BACC", "AUC_GMEAN"]
THRESHOLD_STRATEGIES = ["argmax", "youden"]


def get_patient_id(info, fallback):
    for key in ("patient_id", "id", "case_id", "name", "pid", "PatientID"):
        if isinstance(info, dict) and key in info:
            return info[key]
    return fallback


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.99, gamma=1.5, label_smoothing=0.02):
        super().__init__()
        counts = torch.tensor(samples_per_class, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
        weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-8)
        weights = weights / weights.sum() * len(samples_per_class)

        self.register_buffer("class_weights", weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        num_classes = logits.size(1)

        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / max(num_classes - 1, 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        focal_weight = torch.pow(1.0 - probs, self.gamma)
        class_weights = self.class_weights.to(logits.device).unsqueeze(0)
        loss = -true_dist * focal_weight * log_probs * class_weights
        return loss.sum(dim=1).mean()


def resolve_clinical_path(config, config_path):
    clinical_path = config["clinical_dir"]
    if not os.path.isabs(clinical_path):
        candidate = os.path.join(os.path.dirname(config_path), "clinical.json")
        if os.path.exists(candidate):
            return candidate
    return clinical_path


def load_clinical_infos(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    clinical_path = resolve_clinical_path(config, config_path)
    with open(clinical_path, "r", encoding="utf-8") as f:
        infos = json.load(f)
    return infos


def build_model(args, device):
    modalities = parse_modalities(args.modalities)
    model = MSHF(num_classes=args.num_classes, backbone=args.backbone, modalities=modalities).to(device)

    for name, param in model.mg_backbone.named_parameters():
        if should_train_backbone_param(args.backbone, name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in model.us_backbone.named_parameters():
        if should_train_backbone_param(args.backbone, name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def should_train_backbone_param(backbone, name):
    first = name.split(".")[0]
    if backbone == "ResNet50":
        return first == "7"
    if backbone == "DenseNet121":
        return "denseblock4" in name or "norm5" in name
    if backbone == "InceptionV3":
        return "Mixed_7" in name
    if backbone == "VGG16":
        return first in {"24", "26", "28"}
    if backbone == "ViT-B_16":
        return "encoder_layer_11" in name or "encoder.ln" in name
    return False


def parse_modalities(modalities):
    parsed = [modality.strip().lower() for modality in modalities.split(",") if modality.strip()]
    valid_modalities = {"mg", "us", "clinical"}
    invalid = sorted(set(parsed) - valid_modalities)
    if invalid:
        raise ValueError(f"Invalid modalities {invalid}. Choose from {sorted(valid_modalities)}.")
    if not ({"mg", "us"} & set(parsed)):
        raise ValueError("At least one image modality must be enabled: 'mg' or 'us'.")
    return parsed


def build_optimizer(model, args):
    head_params = [p for n, p in model.named_parameters() if "mg_backbone" not in n and "us_backbone" not in n]
    mg_params = [p for p in model.mg_backbone.parameters() if p.requires_grad]
    us_params = [p for p in model.us_backbone.parameters() if p.requires_grad]

    return optim.AdamW(
        [
            {"params": mg_params, "lr": 1e-6},
            {"params": us_params, "lr": 1e-6},
            {"params": head_params, "lr": args.lr},
        ],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4,
    )


def build_criterion(train_info, args, device):
    if args.loss_type == "ce":
        print("Loss setting: CrossEntropyLoss")
        return nn.CrossEntropyLoss()

    train_labels = [info["label"] for info in train_info]
    counts = Counter(train_labels)
    samples_per_class = [counts.get(i, 1) for i in range(args.num_classes)]
    criterion = ClassBalancedFocalLoss(
        samples_per_class=samples_per_class,
        beta=args.cb_beta,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    ).to(device)
    print(
        "Loss setting: ClassBalancedFocalLoss "
        f"samples_per_class={samples_per_class}, "
        f"weights={criterion.class_weights.detach().cpu().tolist()}, "
        f"beta={args.cb_beta}, gamma={args.focal_gamma}, "
        f"label_smoothing={args.label_smoothing}"
    )
    return criterion


def compute_probability_metrics(logits, labels, bins=15):
    logits = torch.as_tensor(logits, dtype=torch.float32).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()

    if logits.numel() == 0 or labels.numel() == 0:
        return {"ECE": 0.0, "MCE": 0.0, "NLL": 0.0, "Brier": 0.0}

    return {
        "ECE": float(expected_calibration_error(logits, labels, bins=bins)),
        "MCE": float(maximum_calibration_error(logits, labels, bins=bins)),
        "NLL": float(F.cross_entropy(logits, labels).item()),
        "Brier": float(brier_score(logits, labels)),
    }


def compute_binary_metrics(labels, preds, scores, positive_label=0, logits=None, calibration_bins=15):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    scores = np.asarray(scores).astype(float)

    labels_positive = (labels == positive_label).astype(int)
    preds_positive = (preds == positive_label).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels_positive, preds_positive, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    pre = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = f1_score(labels_positive, preds_positive, pos_label=1, zero_division=0)
    bacc = 0.5 * (sen + spe)
    gmean = float(np.sqrt(sen * spe))
    try:
        auc = roc_auc_score(labels_positive, scores)
    except ValueError:
        auc = 0.0
    probability_metrics = (
        compute_probability_metrics(logits, labels, bins=calibration_bins)
        if logits is not None
        else {"ECE": 0.0, "MCE": 0.0, "NLL": 0.0, "Brier": 0.0}
    )

    return {
        "ACC": acc,
        "AUC": auc,
        "F1": f1,
        "Rec": sen,
        "SEN": sen,
        "SPE": spe,
        "PRE": pre,
        "PPV": pre,
        "NPV": npv,
        "ECE": probability_metrics["ECE"],
        "MCE": probability_metrics["MCE"],
        "NLL": probability_metrics["NLL"],
        "Brier": probability_metrics["Brier"],
        "BACC": bacc,
        "GMEAN": gmean,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Positive_Label": positive_label,
    }


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
    labels,
    preds,
    scores,
    positive_label=0,
    threshold_strategy="argmax",
    logits=None,
    calibration_bins=15,
):
    if threshold_strategy == "argmax":
        threshold = np.nan
        thresholded_preds = np.asarray(preds).astype(int)
    elif threshold_strategy == "youden":
        threshold = find_best_youden_threshold(labels, scores, positive_label=positive_label)
        thresholded_preds = predict_from_scores(scores, threshold, positive_label=positive_label)
    else:
        raise ValueError(f"Invalid threshold strategy: {threshold_strategy}")

    metrics = compute_binary_metrics(
        labels,
        thresholded_preds,
        scores,
        positive_label=positive_label,
        logits=logits,
        calibration_bins=calibration_bins,
    )
    metrics["Threshold_Strategy"] = threshold_strategy
    metrics["Decision_Threshold"] = threshold
    return metrics, thresholded_preds, threshold


def save_calibration_outputs(logits, labels, save_dir, prefix="best_val", bins=19):
    logits = torch.as_tensor(logits, dtype=torch.float32).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()

    calibrator = ConfidenceBinTemperatureCalibrator(bins=bins)
    calibrator.fit(logits, labels)
    calibrated_logits = calibrator.transform(logits)

    before = calibration_report(logits, labels)
    after = calibration_report(logits, labels, after_logits=calibrated_logits)

    pt_path = os.path.join(save_dir, f"{prefix}_calibration.pt")
    torch.save(
        {
            "logits": logits,
            "labels": labels,
            "calibrated_logits": calibrated_logits,
            "temperatures": calibrator.temperatures,
            "before": before,
            "after": after,
            "bins": bins,
        },
        pt_path,
    )

    csv_path = os.path.join(save_dir, f"{prefix}_calibration_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "before", "after"])
        for metric in before:
            writer.writerow([metric, before[metric], after[metric]])

    txt_path = os.path.join(save_dir, f"{prefix}_calibration_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("Confidence calibration metrics\n")
        f.write(f"Bins: {bins}\n")
        f.write("Method: confidence-bin temperature calibration\n")
        for metric in before:
            f.write(f"{metric}: before={before[metric]:.6f}, after={after[metric]:.6f}\n")
        f.write("Temperatures:\n")
        f.write(",".join(f"{temperature:.8g}" for temperature in calibrator.temperatures))
        f.write("\n")

    print(f"Saved calibration metrics: {csv_path}")
    print(f"Saved calibration tensors: {pt_path}")
    return before, after


def compute_reliability_arrays(logits, labels, positive_label=0):
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)
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


def binary_average_precision(labels, scores, positive_label=0):
    labels_positive = (np.asarray(labels).astype(int) == positive_label).astype(int)
    if len(np.unique(labels_positive)) < 2:
        return 0.0
    return float(average_precision_score(labels_positive, np.asarray(scores, dtype=float)))


def write_sample_reliability_csv(
    labels,
    logits,
    save_dir,
    prefix,
    positive_label=0,
    patient_ids=None,
    folds=None,
    roi_cosine_similarity=None,
):
    arrays = compute_reliability_arrays(logits, labels, positive_label=positive_label)
    probs = arrays["probs"]
    labels = np.asarray(labels, dtype=int)
    if patient_ids is None:
        patient_ids = [f"sample_{idx}" for idx in range(len(labels))]
    if folds is None:
        folds = [""] * len(labels)
    if roi_cosine_similarity is None:
        roi_cosine_similarity = np.full(len(labels), np.nan, dtype=float)
    else:
        roi_cosine_similarity = np.asarray(roi_cosine_similarity, dtype=float)
    roi_consistency_score = np.clip((roi_cosine_similarity + 1.0) / 2.0, 0.0, 1.0)

    csv_path = os.path.join(save_dir, f"{prefix}_sample_confidence_uncertainty.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "sample_index",
            "fold",
            "patient_id",
            "label",
            "pred_label",
            "correct",
            "confidence",
            "uncertainty",
            "entropy",
            "margin",
            "positive_label",
            "positive_score",
            "roi_cosine_similarity",
            "roi_consistency_score",
        ] + [f"prob_class_{idx}" for idx in range(probs.shape[1])]
        writer.writerow(header)
        for idx in range(len(labels)):
            writer.writerow(
                [
                    idx,
                    folds[idx],
                    patient_ids[idx],
                    labels[idx],
                    arrays["preds"][idx],
                    arrays["correct"][idx],
                    arrays["confidences"][idx],
                    arrays["uncertainty"][idx],
                    arrays["entropy"][idx],
                    arrays["margins"][idx],
                    positive_label,
                    arrays["positive_scores"][idx],
                    roi_cosine_similarity[idx],
                    roi_consistency_score[idx],
                ]
                + probs[idx].tolist()
            )
    return csv_path


def compute_threshold_acceptance_table(labels, logits, positive_label=0, thresholds=None):
    arrays = compute_reliability_arrays(logits, labels, positive_label=positive_label)
    labels = np.asarray(labels, dtype=int)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    rows = []
    for threshold in thresholds:
        accepted = arrays["confidences"] >= threshold
        n_accept = int(accepted.sum())
        coverage = n_accept / max(len(labels), 1)
        if n_accept > 0:
            acc = float(arrays["correct"][accepted].mean())
            ap = binary_average_precision(labels[accepted], arrays["positive_scores"][accepted], positive_label=positive_label)
            mean_conf = float(arrays["confidences"][accepted].mean())
            mean_uncert = float(arrays["uncertainty"][accepted].mean())
        else:
            acc = np.nan
            ap = np.nan
            mean_conf = np.nan
            mean_uncert = np.nan
        rows.append(
            {
                "confidence_threshold": float(threshold),
                "accepted": n_accept,
                "total": int(len(labels)),
                "acceptance_ratio": float(coverage),
                "accuracy": acc,
                "average_precision": ap,
                "mean_confidence": mean_conf,
                "mean_uncertainty": mean_uncert,
            }
        )
    return rows


def compute_coverage_performance_table(labels, logits, positive_label=0, coverages=None):
    arrays = compute_reliability_arrays(logits, labels, positive_label=positive_label)
    labels = np.asarray(labels, dtype=int)
    if coverages is None:
        coverages = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]

    order = np.argsort(-arrays["confidences"])
    rows = []
    for coverage in coverages:
        n_accept = max(1, int(np.ceil(len(labels) * coverage)))
        selected = order[:n_accept]
        selected_labels = labels[selected]
        selected_scores = arrays["positive_scores"][selected]
        selected_preds = arrays["preds"][selected]
        metrics = compute_binary_metrics(selected_labels, selected_preds, selected_scores, positive_label=positive_label)
        ap = binary_average_precision(selected_labels, selected_scores, positive_label=positive_label)
        rows.append(
            {
                "target_acceptance_ratio": float(coverage),
                "accepted": int(n_accept),
                "total": int(len(labels)),
                "actual_acceptance_ratio": float(n_accept / max(len(labels), 1)),
                "confidence_threshold": float(arrays["confidences"][selected[-1]]),
                "ACC": metrics["ACC"],
                "AP": ap,
                "AUC": metrics["AUC"],
                "SEN": metrics["SEN"],
                "SPE": metrics["SPE"],
                "F1": metrics["F1"],
                "mean_confidence": float(arrays["confidences"][selected].mean()),
                "mean_uncertainty": float(arrays["uncertainty"][selected].mean()),
            }
        )
    return rows


def save_rows_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_simple_svg_plot(x, ys, labels, path, title, xlabel, ylabel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.4, 4.8))
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker="o", linewidth=2.0, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    if labels:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_confidence_uncertainty_scatter(labels, logits, save_dir, prefix, positive_label=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = compute_reliability_arrays(logits, labels, positive_label=positive_label)
    colors = np.where(arrays["correct"] == 1, "#1f77b4", "#d62728")
    svg_path = os.path.join(save_dir, f"{prefix}_confidence_uncertainty_scatter.svg")
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(arrays["confidences"], arrays["uncertainty"], c=colors, s=28, alpha=0.78, edgecolors="none")
    plt.xlabel("Confidence")
    plt.ylabel("Uncertainty (normalized entropy)")
    plt.title("Sample confidence vs uncertainty")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(svg_path)
    plt.close()
    return svg_path


def save_reliability_analysis(
    labels,
    logits,
    save_dir,
    prefix,
    positive_label=0,
    patient_ids=None,
    folds=None,
    roi_cosine_similarity=None,
):
    os.makedirs(save_dir, exist_ok=True)
    sample_csv = write_sample_reliability_csv(
        labels,
        logits,
        save_dir,
        prefix,
        positive_label=positive_label,
        patient_ids=patient_ids,
        folds=folds,
        roi_cosine_similarity=roi_cosine_similarity,
    )
    scatter_svg = save_confidence_uncertainty_scatter(labels, logits, save_dir, prefix, positive_label=positive_label)

    threshold_rows = compute_threshold_acceptance_table(labels, logits, positive_label=positive_label)
    threshold_csv = os.path.join(save_dir, f"{prefix}_confidence_threshold_acceptance.csv")
    save_rows_csv(threshold_rows, threshold_csv)
    save_simple_svg_plot(
        [row["confidence_threshold"] for row in threshold_rows],
        [[row["acceptance_ratio"] for row in threshold_rows]],
        ["Acceptance ratio"],
        os.path.join(save_dir, f"{prefix}_confidence_threshold_acceptance.svg"),
        "Confidence threshold vs acceptance ratio",
        "Confidence threshold",
        "Acceptance ratio",
    )

    coverage_rows = compute_coverage_performance_table(labels, logits, positive_label=positive_label)
    coverage_csv = os.path.join(save_dir, f"{prefix}_coverage_acc_ap.csv")
    save_rows_csv(coverage_rows, coverage_csv)
    save_simple_svg_plot(
        [row["actual_acceptance_ratio"] for row in coverage_rows],
        [
            [row["ACC"] for row in coverage_rows],
            [row["AP"] for row in coverage_rows],
        ],
        ["ACC", "AP"],
        os.path.join(save_dir, f"{prefix}_coverage_acc_ap.svg"),
        "Acceptance ratio vs ACC/AP",
        "Acceptance ratio",
        "Metric",
    )

    arrays = compute_reliability_arrays(logits, labels, positive_label=positive_label)
    summary_rows = [
        {"metric": "mean_confidence", "value": float(arrays["confidences"].mean())},
        {"metric": "mean_uncertainty", "value": float(arrays["uncertainty"].mean())},
        {"metric": "mean_entropy", "value": float(arrays["entropy"].mean())},
        {"metric": "mean_margin", "value": float(arrays["margins"].mean())},
        {"metric": "accuracy_from_argmax", "value": float(arrays["correct"].mean())},
        {
            "metric": "high_conf_error_rate_0.8",
            "value": float(((arrays["confidences"] >= 0.8) & (arrays["correct"] == 0)).mean()),
        },
        {
            "metric": "high_conf_error_rate_0.9",
            "value": float(((arrays["confidences"] >= 0.9) & (arrays["correct"] == 0)).mean()),
        },
        {
            "metric": "average_precision",
            "value": binary_average_precision(labels, arrays["positive_scores"], positive_label=positive_label),
        },
    ]
    summary_csv = os.path.join(save_dir, f"{prefix}_reliability_summary.csv")
    save_rows_csv(summary_rows, summary_csv)
    print(f"Saved reliability sample CSV: {sample_csv}")
    print(f"Saved confidence/uncertainty scatter: {scatter_svg}")
    return {
        "sample_csv": sample_csv,
        "scatter_svg": scatter_svg,
        "threshold_csv": threshold_csv,
        "coverage_csv": coverage_csv,
        "summary_csv": summary_csv,
    }


def compute_calibration_bin_gaps(logits, labels, bins=19):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels)

    gaps = torch.zeros(bins, dtype=torch.float32)
    over_gaps = torch.zeros(bins, dtype=torch.float32)
    under_gaps = torch.zeros(bins, dtype=torch.float32)
    avg_confidences = torch.zeros(bins, dtype=torch.float32)
    avg_accuracies = torch.zeros(bins, dtype=torch.float32)
    counts = torch.zeros(bins, dtype=torch.float32)
    boundaries = torch.linspace(0, 1, bins + 1)
    for i, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        if in_bin.any():
            avg_conf = confidences[in_bin].mean()
            avg_acc = correct[in_bin].float().mean()
            diff = avg_conf - avg_acc
            gaps[i] = torch.abs(diff)
            over_gaps[i] = torch.clamp(diff, min=0.0)
            under_gaps[i] = torch.clamp(-diff, min=0.0)
            avg_confidences[i] = avg_conf
            avg_accuracies[i] = avg_acc
            counts[i] = in_bin.float().sum()
    return {
        "abs": gaps,
        "over": over_gaps,
        "under": under_gaps,
        "avg_confidence": avg_confidences,
        "avg_accuracy": avg_accuracies,
        "count": counts,
    }


def update_calibration_gap_table(previous, current, momentum=0.0):
    if previous is None or momentum <= 0:
        return {key: value.clone().detach() for key, value in current.items()}
    updated = {}
    for key, value in current.items():
        prev_value = previous.get(key, torch.zeros_like(value))
        if key == "count":
            updated[key] = value.clone().detach()
        else:
            updated[key] = (momentum * prev_value + (1.0 - momentum) * value).clone().detach()
    return updated


def summarize_gap_table(gap_table):
    if gap_table is None:
        return {"abs": 0.0, "over": 0.0, "under": 0.0}
    return {
        "abs": float(gap_table["abs"].mean().item()),
        "over": float(gap_table["over"].mean().item()),
        "under": float(gap_table["under"].mean().item()),
    }


def save_calibration_gap_table(logits, labels, save_dir, prefix, bins=19):
    gap_table = compute_calibration_bin_gaps(logits, labels, bins=bins)
    csv_path = os.path.join(save_dir, f"{prefix}_calibration_bin_gap_table.csv")
    boundaries = torch.linspace(0, 1, bins + 1)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bin",
                "lower",
                "upper",
                "count",
                "avg_confidence",
                "avg_accuracy",
                "abs_gap",
                "overconfidence_gap",
                "underconfidence_gap",
            ]
        )
        for idx in range(bins):
            writer.writerow(
                [
                    idx,
                    float(boundaries[idx].item()),
                    float(boundaries[idx + 1].item()),
                    float(gap_table["count"][idx].item()),
                    float(gap_table["avg_confidence"][idx].item()),
                    float(gap_table["avg_accuracy"][idx].item()),
                    float(gap_table["abs"][idx].item()),
                    float(gap_table["over"][idx].item()),
                    float(gap_table["under"][idx].item()),
                ]
            )
    print(f"Saved calibration bin gap table: {csv_path}")
    return csv_path


def save_run_arguments(args, save_dir):
    args_dict = vars(args)
    txt_path = os.path.join(save_dir, "run_arguments.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Run arguments\n")
        f.write("=============\n")
        for key in sorted(args_dict):
            f.write(f"{key}: {args_dict[key]}\n")

    json_path = os.path.join(save_dir, "run_arguments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved run arguments: {txt_path}")
    print(f"Saved run arguments JSON: {json_path}")
    return txt_path


def calibration_feedback_loss(
    logits,
    labels,
    gap_table,
    bins=19,
    gap_weight=0.0,
    over_gap_weight=1.0,
    under_gap_weight=0.25,
    overconfidence_weight=0.0,
    overconfidence_threshold=0.0,
):
    if gap_table is None or (gap_weight <= 0 and overconfidence_weight <= 0):
        return logits.new_tensor(0.0)

    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)

    losses = []
    if gap_weight > 0:
        boundaries = torch.linspace(0, 1, bins + 1, device=logits.device)
        bin_index = torch.bucketize(confidences.detach(), boundaries[1:-1], right=False).clamp(0, bins - 1)
        over_gaps = gap_table["over"].to(logits.device)[bin_index]
        under_gaps = gap_table["under"].to(logits.device)[bin_index]
        sample_gaps = over_gap_weight * over_gaps + under_gap_weight * under_gaps
        ce_per_sample = F.cross_entropy(logits, labels, reduction="none")
        losses.append((sample_gaps * ce_per_sample).mean() * gap_weight)

    if overconfidence_weight > 0:
        wrong = predictions.detach().ne(labels)
        if overconfidence_threshold > 0:
            overconfidence = (torch.clamp(confidences - overconfidence_threshold, min=0.0) * wrong.float()).mean()
        else:
            overconfidence = (confidences * wrong.float()).mean()
        losses.append(overconfidence * overconfidence_weight)

    return sum(losses, logits.new_tensor(0.0))


def t_critical_975(df):
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        20: 2.086,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 20:
        return table[10]
    if df < 30:
        return table[20]
    return 1.96


def summarize_fold_metrics(fold_rows):
    summary = []
    n = len(fold_rows)
    t_value = t_critical_975(max(n - 1, 1))
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in fold_rows], dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        half_width = float(t_value * std / np.sqrt(n)) if n > 1 else 0.0
        summary.append(
            {
                "Metric": metric,
                "Mean": mean,
                "Std": std,
                "CI95_Low": mean - half_width,
                "CI95_High": mean + half_width,
            }
        )
    return summary


def get_monitor_score(metrics, monitor_metric):
    if monitor_metric == "REC":
        return metrics["Rec"]
    if monitor_metric == "ACC_AUC":
        return metrics["ACC"] + metrics["AUC"]
    if monitor_metric == "AUC_BACC":
        return metrics["AUC"] + metrics["BACC"]
    if monitor_metric == "AUC_GMEAN":
        return metrics["AUC"] + metrics["GMEAN"]
    return metrics[monitor_metric]


def is_valid_best_candidate(metrics, args):
    return metrics["SEN"] >= args.min_save_sen and metrics["SPE"] >= args.min_save_spe


def get_summary_row(summary, metric_name):
    for row in summary:
        if row["Metric"] == metric_name:
            return row
    raise ValueError(f"Missing summary metric: {metric_name}")


def save_pooled_roc(
    labels,
    scores,
    preds,
    save_dir,
    prefix="pooled_oof",
    positive_label=0,
    threshold_strategy="argmax",
):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    preds = np.asarray(preds).astype(int)
    labels_positive = (labels == positive_label).astype(int)
    preds_positive = (preds == positive_label).astype(int)

    if len(np.unique(labels_positive)) < 2:
        print("Skipped pooled ROC: labels contain only one class.")
        return

    fpr, tpr, thresholds = roc_curve(labels_positive, scores)
    roc_auc = roc_auc_score(labels_positive, scores)

    points_csv = os.path.join(save_dir, f"{prefix}_roc_points.csv")
    with open(points_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold"])
        for fp, tp, threshold in zip(fpr, tpr, thresholds):
            writer.writerow([fp, tp, threshold])

    inputs_csv = os.path.join(save_dir, f"{prefix}_roc_inputs.csv")
    with open(inputs_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "label", "y_true", "y_score", "pred_label", "y_pred", "positive_label"])
        for idx, (label, label_pos, score, pred, pred_pos) in enumerate(
            zip(labels, labels_positive, scores, preds, preds_positive)
        ):
            writer.writerow([idx, label, label_pos, score, pred, pred_pos, positive_label])

    inputs_npz = os.path.join(save_dir, f"{prefix}_roc_inputs.npz")
    np.savez(
        inputs_npz,
        y_true=labels_positive,
        y_score=scores,
        y_pred=preds_positive,
        y_true_label=labels,
        y_pred_label=preds,
        positive_label=np.asarray(positive_label, dtype=int),
        threshold_strategy=np.asarray(threshold_strategy),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc=np.asarray(roc_auc, dtype=float),
    )

    svg_path = os.path.join(save_dir, f"{prefix}_roc_curve.svg")
    save_roc_svg(fpr, tpr, roc_auc, svg_path, title="Pooled Out-of-Fold ROC")

    print(f"Saved pooled ROC curve: {svg_path}")
    print(f"Saved pooled ROC inputs: {inputs_npz}")


def save_mean_fold_roc(best_payloads, summary, save_dir, prefix="mean_fold", positive_label=0, num_points=101):
    mean_fpr = np.linspace(0.0, 1.0, num_points)
    interp_tprs = []
    fold_aucs = []
    fold_rows = []

    for payload in best_payloads:
        labels = np.asarray(payload["labels"]).astype(int)
        scores = np.asarray(payload["scores"]).astype(float)
        labels_positive = (labels == positive_label).astype(int)

        if len(np.unique(labels_positive)) < 2:
            print(f"Skipped fold {payload['fold']} in mean ROC: labels contain only one class.")
            continue

        fpr, tpr, thresholds = roc_curve(labels_positive, scores)
        fold_auc = roc_auc_score(labels_positive, scores)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        interp_tprs.append(interp_tpr)
        fold_aucs.append(fold_auc)
        fold_rows.append(
            {
                "fold": payload["fold"],
                "epoch": payload["epoch"],
                "auc": fold_auc,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
            }
        )

    if not interp_tprs:
        print("Skipped mean fold ROC: no valid fold ROC curves.")
        return

    interp_tprs = np.asarray(interp_tprs, dtype=float)
    fold_aucs = np.asarray(fold_aucs, dtype=float)
    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr = np.std(interp_tprs, axis=0, ddof=1) if len(interp_tprs) > 1 else np.zeros_like(mean_tpr)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    tpr_low = np.clip(mean_tpr - std_tpr, 0.0, 1.0)
    tpr_high = np.clip(mean_tpr + std_tpr, 0.0, 1.0)

    auc_summary = get_summary_row(summary, "AUC")
    mean_auc = float(auc_summary["Mean"])
    std_auc = float(auc_summary["Std"])
    ci95_low = float(auc_summary["CI95_Low"])
    ci95_high = float(auc_summary["CI95_High"])

    points_csv = os.path.join(save_dir, f"{prefix}_roc_points.csv")
    with open(points_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_fpr", "mean_tpr", "std_tpr", "tpr_low", "tpr_high"])
        for fp, tp, std, low, high in zip(mean_fpr, mean_tpr, std_tpr, tpr_low, tpr_high):
            writer.writerow([fp, tp, std, low, high])

    fold_auc_csv = os.path.join(save_dir, f"{prefix}_roc_fold_auc.csv")
    with open(fold_auc_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_epoch", "auc"])
        for row in fold_rows:
            writer.writerow([row["fold"], row["epoch"], row["auc"]])

    inputs_npz = os.path.join(save_dir, f"{prefix}_roc_inputs.npz")
    np.savez(
        inputs_npz,
        fpr=mean_fpr,
        tpr=mean_tpr,
        mean_fpr=mean_fpr,
        mean_tpr=mean_tpr,
        std_tpr=std_tpr,
        tpr_low=tpr_low,
        tpr_high=tpr_high,
        fold_auc_values=fold_aucs,
        auc=np.asarray(mean_auc, dtype=float),
        mean_auc=np.asarray(mean_auc, dtype=float),
        std_auc=np.asarray(std_auc, dtype=float),
        ci95_low=np.asarray(ci95_low, dtype=float),
        ci95_high=np.asarray(ci95_high, dtype=float),
        positive_label=np.asarray(positive_label, dtype=int),
    )

    svg_path = os.path.join(save_dir, f"{prefix}_roc_curve.svg")
    save_roc_svg(mean_fpr, mean_tpr, mean_auc, svg_path, title="Mean Five-Fold ROC")

    print(f"Saved mean fold ROC curve: {svg_path}")
    print(f"Saved mean fold ROC inputs: {inputs_npz}")
    print(f"Saved mean fold ROC points: {points_csv}")
    print(f"Saved fold AUC values: {fold_auc_csv}")


def save_roc_svg(fpr, tpr, roc_auc, svg_path, title):
    width, height = 640, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 35, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def point(x, y):
        px = margin_left + x * plot_w
        py = height - margin_bottom - y * plot_h
        return f"{px:.2f},{py:.2f}"

    roc_points = " ".join(point(fp, tp) for fp, tp in zip(fpr, tpr))
    diag_points = f"{point(0, 0)} {point(1, 1)}"
    axis_x1, axis_y1 = margin_left, height - margin_bottom
    axis_x2, axis_y2 = width - margin_right, height - margin_bottom
    axis_y_top = margin_top
    legend_x = width - margin_right - 215
    legend_y = height - margin_bottom - 58

    tick_elements = []
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = margin_left + value * plot_w
        y = height - margin_bottom - value * plot_h
        tick_elements.append(f'<line x1="{x:.2f}" y1="{axis_y1}" x2="{x:.2f}" y2="{axis_y1 + 6}" stroke="#333"/>')
        tick_elements.append(f'<text x="{x:.2f}" y="{axis_y1 + 24}" text-anchor="middle" font-size="12">{value:.2f}</text>')
        tick_elements.append(f'<line x1="{margin_left - 6}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="#333"/>')
        tick_elements.append(f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{value:.2f}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-family="Arial">{title}</text>
<line x1="{axis_x1}" y1="{axis_y1}" x2="{axis_x2}" y2="{axis_y2}" stroke="#222" stroke-width="1.5"/>
<line x1="{axis_x1}" y1="{axis_y1}" x2="{axis_x1}" y2="{axis_y_top}" stroke="#222" stroke-width="1.5"/>
{''.join(tick_elements)}
<polyline points="{diag_points}" fill="none" stroke="#1f2a7a" stroke-dasharray="6 6" stroke-width="2"/>
<polyline points="{roc_points}" fill="none" stroke="#f5b335" stroke-width="3"/>
<rect x="{legend_x}" y="{legend_y}" width="205" height="34" fill="white" stroke="#dddddd" rx="2"/>
<line x1="{legend_x + 14}" y1="{legend_y + 17}" x2="{legend_x + 48}" y2="{legend_y + 17}" stroke="#f5b335" stroke-width="3"/>
<text x="{legend_x + 58}" y="{legend_y + 21}" font-size="13" font-family="Arial">ROC curve (AUC = {roc_auc:.2f})</text>
<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-size="14" font-family="Arial">False Positive Rate</text>
<text x="20" y="{height / 2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 20 {height / 2})">True Positive Rate</text>
</svg>
'''
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_seg,
    train_mode,
    epoch,
    epochs,
    positive_label,
    modalities,
    threshold_strategy="argmax",
    calibration_gap_table=None,
    calibration_bins=19,
    calibration_feedback_weight=0.0,
    over_gap_weight=1.0,
    under_gap_weight=0.25,
    overconfidence_weight=0.0,
    overconfidence_threshold=0.0,
    roi_consistency_weight=0.0,
    collect_consistency=False,
):
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels, all_preds, all_scores, all_logits = [], [], [], []
    all_roi_cosine_similarity = []
    desc = f"Epoch {epoch}/{epochs} [{'Train' if train_mode else 'Val'}]"
    iterator = tqdm(loader, desc=desc)

    for batch in iterator:
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
        labels = labels.to(device)
        clinical = clinical.to(device)

        with torch.set_grad_enabled(train_mode):
            if train_mode:
                optimizer.zero_grad()

            with autocast():
                if train_mode:
                    outputs = model(img_mlo, img_cc, img_us, clinical, mask_mlo, mask_cc, mask_us)
                    outputs, out_mg, out_us, roi_consistency_loss = outputs
                    loss_main = criterion(outputs, labels)
                    loss = loss_main
                    if "mg" in modalities and out_mg is not None:
                        loss = loss + 0.3 * criterion(out_mg, labels)
                    if "us" in modalities and out_us is not None:
                        loss = loss + 0.3 * criterion(out_us, labels)
                    if roi_consistency_weight > 0:
                        loss = loss + roi_consistency_weight * roi_consistency_loss
                    loss = loss + calibration_feedback_loss(
                        outputs,
                        labels,
                        calibration_gap_table,
                        bins=calibration_bins,
                        gap_weight=calibration_feedback_weight,
                        over_gap_weight=over_gap_weight,
                        under_gap_weight=under_gap_weight,
                        overconfidence_weight=overconfidence_weight,
                        overconfidence_threshold=overconfidence_threshold,
                    )
                elif collect_consistency:
                    outputs, roi_cosine_similarity = model(
                        img_mlo,
                        img_cc,
                        img_us,
                        clinical,
                        mask_mlo,
                        mask_cc,
                        mask_us,
                        return_consistency_score=True,
                    )
                    loss = criterion(outputs, labels)
                else:
                    outputs = model(img_mlo, img_cc, img_us, clinical, mask_mlo, mask_cc, mask_us)
                    loss = criterion(outputs, labels)

            if train_mode:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item() * labels.size(0)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_scores.extend(probs[:, positive_label].detach().cpu().numpy())
        all_logits.extend(outputs.detach().cpu().numpy())
        if (not train_mode) and collect_consistency:
            all_roi_cosine_similarity.extend(roi_cosine_similarity.detach().cpu().numpy())
        iterator.set_postfix(loss=loss.item())

    avg_loss = running_loss / max(len(loader.dataset), 1)
    metrics, final_preds, threshold = apply_threshold_strategy(
        all_labels,
        all_preds,
        all_scores,
        positive_label=positive_label,
        threshold_strategy=threshold_strategy,
        logits=all_logits,
        calibration_bins=calibration_bins,
    )
    if collect_consistency:
        return avg_loss, metrics, all_labels, final_preds, all_scores, threshold, all_logits, all_roi_cosine_similarity

    return avg_loss, metrics, all_labels, final_preds, all_scores, threshold, all_logits


def split_inner_train_calibration(train_info, seed):
    """Split each fold's 80% training partition into 70% train and 10% calibration feedback."""
    labels = [info["label"] for info in train_info]
    inner_train, inner_calib = train_test_split(
        train_info,
        test_size=1 / 8,
        random_state=seed,
        stratify=labels,
    )
    return inner_train, inner_calib


def run_fold(fold_idx, train_info, val_info, args, root_save_dir, device):
    set_seed(args.seed + fold_idx)
    modalities = parse_modalities(args.modalities)
    fold_name = f"fold_{fold_idx}"
    fold_dir = os.path.join(root_save_dir, fold_name)
    os.makedirs(fold_dir, exist_ok=True)

    inner_train_info, calib_info = split_inner_train_calibration(train_info, args.seed + fold_idx)

    train_dataset = MyDataset(inner_train_info, args.config_path, use_seg=args.use_seg, is_train=True)
    calib_dataset = MyDataset(calib_info, args.config_path, use_seg=args.use_seg, is_train=False)
    val_dataset = MyDataset(val_info, args.config_path, use_seg=args.use_seg, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Fold {fold_idx} outer train labels: {dict(sorted(Counter([x['label'] for x in train_info]).items()))}")
    print(f"Fold {fold_idx} inner train labels: {dict(sorted(Counter([x['label'] for x in inner_train_info]).items()))}")
    print(f"Fold {fold_idx} calibration-feedback labels: {dict(sorted(Counter([x['label'] for x in calib_info]).items()))}")
    print(f"Fold {fold_idx} held-out val labels: {dict(sorted(Counter([x['label'] for x in val_info]).items()))}")
    print(f"Fold {fold_idx} use_seg={args.use_seg}")
    print(f"Fold {fold_idx} modalities={','.join(modalities)}")
    print(f"Fold {fold_idx} positive_label={args.positive_label} (0=Luminal, 1=non-Luminal)")
    print(f"Fold {fold_idx} best-model monitor metric: {args.monitor_metric}")
    print(f"Fold {fold_idx} best-model constraints: SEN>={args.min_save_sen}, SPE>={args.min_save_spe}")
    print(f"Fold {fold_idx} threshold strategy: {args.threshold_strategy}")

    model = build_model(args, device)
    optimizer = build_optimizer(model, args)
    criterion = build_criterion(inner_train_info, args, device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-7
    )
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=os.path.join("runs", os.path.basename(root_save_dir), fold_name))

    metrics_csv = os.path.join(fold_dir, "epoch_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(
            ["Epoch", "Train_Loss"]
            + [f"Train_{metric}" for metric in METRIC_NAMES]
            + ["Calib_Loss"]
            + [f"Calib_{metric}" for metric in METRIC_NAMES]
            + ["Val_Loss"]
            + [f"Val_{metric}" for metric in METRIC_NAMES]
            + [
                "Val_BACC",
                "Val_GMEAN",
                "Val_ACC_AUC_Score",
                "Monitor_Metric",
                "Monitor_Score",
                "Threshold_Strategy",
                "Val_Decision_Threshold",
                "Meets_Save_Constraints",
                "Calibration_Feedback_Enabled",
                "Calibration_Mean_Bin_Gap",
                "Calibration_Mean_Over_Gap",
                "Calibration_Mean_Under_Gap",
                "Calibration_MA_Momentum",
            ]
        )

    best_score = -1.0
    best_payload = None
    fallback_score = -1.0
    fallback_payload = None
    fallback_model_path = os.path.join(fold_dir, "fallback_model.pth")
    calibration_gap_table = None
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_metrics, _, _, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.use_seg, True, epoch, args.num_epochs,
            args.positive_label, modalities, threshold_strategy="argmax",
            calibration_gap_table=calibration_gap_table,
            calibration_bins=args.calibration_bins,
            calibration_feedback_weight=args.calibration_feedback_weight,
            over_gap_weight=args.over_gap_weight,
            under_gap_weight=args.under_gap_weight,
            overconfidence_weight=args.overconfidence_weight,
            overconfidence_threshold=args.overconfidence_threshold,
            roi_consistency_weight=args.roi_consistency_weight,
        )
        calib_loss, calib_metrics, calib_labels, _, _, _, calib_logits = run_epoch(
            model, calib_loader, criterion, optimizer, scaler, device, args.use_seg, False, epoch, args.num_epochs,
            args.positive_label, modalities, threshold_strategy="argmax"
        )
        (
            val_loss,
            val_metrics,
            val_labels,
            val_preds,
            val_scores,
            val_threshold,
            val_logits,
            val_roi_cosine_similarity,
        ) = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, args.use_seg, False, epoch, args.num_epochs,
            args.positive_label, modalities, threshold_strategy=args.threshold_strategy, collect_consistency=True
        )

        val_acc_auc_score = val_metrics["ACC"] + val_metrics["AUC"]
        monitor_score = get_monitor_score(val_metrics, args.monitor_metric)
        meets_save_constraints = is_valid_best_candidate(val_metrics, args)
        current_gap_table = compute_calibration_bin_gaps(calib_logits, calib_labels, bins=args.calibration_bins)
        calibration_gap_table = update_calibration_gap_table(
            calibration_gap_table,
            current_gap_table,
            momentum=args.calibration_gap_momentum,
        )
        gap_summary = summarize_gap_table(calibration_gap_table)
        current_mean_gap = gap_summary["abs"]
        print(
            f"Fold {fold_idx} Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, ACC={train_metrics['ACC']:.4f}, AUC={train_metrics['AUC']:.4f}; "
            f"Calib Loss={calib_loss:.4f}, ACC={calib_metrics['ACC']:.4f}, AUC={calib_metrics['AUC']:.4f}; "
            f"Val Loss={val_loss:.4f}, ACC={val_metrics['ACC']:.4f}, Rec={val_metrics['Rec']:.4f}, "
            f"SEN={val_metrics['SEN']:.4f}, "
            f"SPE={val_metrics['SPE']:.4f}, PRE={val_metrics['PRE']:.4f}, "
            f"F1={val_metrics['F1']:.4f}, AUC={val_metrics['AUC']:.4f}, "
            f"ECE={val_metrics['ECE']:.4f}, MCE={val_metrics['MCE']:.4f}, "
            f"NLL={val_metrics['NLL']:.4f}, Brier={val_metrics['Brier']:.4f}, "
            f"ACC+AUC={val_acc_auc_score:.4f}, "
            f"BACC={val_metrics['BACC']:.4f}, GMEAN={val_metrics['GMEAN']:.4f}, "
            f"Monitor({args.monitor_metric})={monitor_score:.4f}, "
            f"Threshold({args.threshold_strategy})={val_threshold:.6g}, MeetsConstraints={meets_save_constraints}, "
            f"CalibMeanGap={current_mean_gap:.4f}, "
            f"OverGap={gap_summary['over']:.4f}, UnderGap={gap_summary['under']:.4f}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/calibration", calib_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for metric in METRIC_NAMES:
            writer.add_scalar(f"{metric}/train", train_metrics[metric], epoch)
            writer.add_scalar(f"{metric}/calibration", calib_metrics[metric], epoch)
            writer.add_scalar(f"{metric}/val", val_metrics[metric], epoch)

        with open(metrics_csv, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(
                [epoch, train_loss]
                + [train_metrics[m] for m in METRIC_NAMES]
                + [calib_loss]
                + [calib_metrics[m] for m in METRIC_NAMES]
                + [val_loss]
                + [val_metrics[m] for m in METRIC_NAMES]
                + [
                    val_metrics["BACC"],
                    val_metrics["GMEAN"],
                    val_acc_auc_score,
                    args.monitor_metric,
                    monitor_score,
                    args.threshold_strategy,
                    val_threshold,
                    meets_save_constraints,
                    args.calibration_feedback_weight > 0 or args.overconfidence_weight > 0,
                    current_mean_gap,
                    gap_summary["over"],
                    gap_summary["under"],
                    args.calibration_gap_momentum,
                ]
            )

        scheduler.step(monitor_score)

        current_payload = {
            "fold": fold_idx,
            "epoch": epoch,
            "monitor_metric": args.monitor_metric,
            "score": monitor_score,
            "acc_auc_score": val_acc_auc_score,
            "meets_save_constraints": meets_save_constraints,
            "threshold_strategy": args.threshold_strategy,
            "decision_threshold": val_threshold,
            "metrics": val_metrics,
            "labels": np.asarray(val_labels, dtype=int),
            "preds": np.asarray(val_preds, dtype=int),
            "scores": np.asarray(val_scores, dtype=float),
            "logits": np.asarray(val_logits, dtype=float),
            "roi_cosine_similarity": np.asarray(val_roi_cosine_similarity, dtype=float),
            "patient_ids": np.asarray([get_patient_id(info, f"fold{fold_idx}_val{idx}") for idx, info in enumerate(val_info)]),
        }

        if monitor_score > fallback_score:
            fallback_score = monitor_score
            fallback_payload = current_payload
            torch.save(model.state_dict(), fallback_model_path)

        if meets_save_constraints and monitor_score > best_score:
            best_score = monitor_score
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
            best_payload = current_payload

            with open(os.path.join(fold_dir, "best_metrics.txt"), "w") as f:
                f.write(f"Best Fold {fold_idx} Metrics (Epoch {epoch}):\n")
                f.write(f"Positive label: {args.positive_label} (0=Luminal, 1=non-Luminal)\n")
                f.write(f"Monitor metric: {args.monitor_metric}\n")
                f.write(f"Monitor score: {monitor_score:.4f}\n")
                f.write(f"Threshold strategy: {args.threshold_strategy}\n")
                f.write(f"Decision threshold: {val_threshold:.8g}\n")
                f.write(f"Save constraints: SEN>={args.min_save_sen}, SPE>={args.min_save_spe}\n")
                f.write(f"Meets save constraints: {meets_save_constraints}\n")
                f.write(f"Acc+AUC Score: {val_acc_auc_score:.4f}\n")
                for metric in METRIC_NAMES:
                    f.write(f"{metric}: {val_metrics[metric]:.4f}\n")
                f.write(f"BACC: {val_metrics['BACC']:.4f}\n")
                f.write(f"GMEAN: {val_metrics['GMEAN']:.4f}\n")
                f.write(f"TN: {val_metrics['TN']}\n")
                f.write(f"FP: {val_metrics['FP']}\n")
                f.write(f"FN: {val_metrics['FN']}\n")
                f.write(f"TP: {val_metrics['TP']}\n")

            save_calibration_outputs(val_logits, val_labels, fold_dir, "best_val", bins=args.calibration_bins)
            save_calibration_outputs(calib_logits, calib_labels, fold_dir, "best_calibration_split", bins=args.calibration_bins)
            save_calibration_gap_table(val_logits, val_labels, fold_dir, "best_val", bins=args.calibration_bins)
            save_calibration_gap_table(calib_logits, calib_labels, fold_dir, "best_calibration_split", bins=args.calibration_bins)
            save_reliability_analysis(
                val_labels,
                val_logits,
                fold_dir,
                "best_val",
                positive_label=args.positive_label,
                patient_ids=current_payload["patient_ids"],
                folds=np.full(len(val_labels), fold_idx, dtype=int),
                roi_cosine_similarity=current_payload["roi_cosine_similarity"],
            )

    writer.close()
    if best_payload is None:
        if fallback_payload is None:
            raise RuntimeError(f"Fold {fold_idx} did not produce a best model")

        best_payload = fallback_payload
        shutil.copyfile(fallback_model_path, os.path.join(fold_dir, "best_model.pth"))
        metrics = best_payload["metrics"]
        with open(os.path.join(fold_dir, "best_metrics.txt"), "w") as f:
            f.write(f"Best Fold {fold_idx} Metrics (Epoch {best_payload['epoch']}):\n")
            f.write("WARNING: No epoch met the save constraints; saved the best fallback model.\n")
            f.write(f"Positive label: {args.positive_label} (0=Luminal, 1=non-Luminal)\n")
            f.write(f"Monitor metric: {args.monitor_metric}\n")
            f.write(f"Monitor score: {best_payload['score']:.4f}\n")
            f.write(f"Threshold strategy: {best_payload['threshold_strategy']}\n")
            f.write(f"Decision threshold: {best_payload['decision_threshold']:.8g}\n")
            f.write(f"Save constraints: SEN>={args.min_save_sen}, SPE>={args.min_save_spe}\n")
            f.write(f"Meets save constraints: {best_payload['meets_save_constraints']}\n")
            f.write(f"Acc+AUC Score: {best_payload['acc_auc_score']:.4f}\n")
            for metric in METRIC_NAMES:
                f.write(f"{metric}: {metrics[metric]:.4f}\n")
            f.write(f"BACC: {metrics['BACC']:.4f}\n")
            f.write(f"GMEAN: {metrics['GMEAN']:.4f}\n")
            f.write(f"TN: {metrics['TN']}\n")
            f.write(f"FP: {metrics['FP']}\n")
            f.write(f"FN: {metrics['FN']}\n")
            f.write(f"TP: {metrics['TP']}\n")
        print(
            f"WARNING: Fold {fold_idx} had no epoch meeting SEN>={args.min_save_sen} and "
            f"SPE>={args.min_save_spe}; saved the best fallback model."
        )
        save_calibration_outputs(
            best_payload["logits"],
            best_payload["labels"],
            fold_dir,
            "best_val",
            bins=args.calibration_bins,
        )
        save_calibration_gap_table(
            best_payload["logits"],
            best_payload["labels"],
            fold_dir,
            "best_val",
            bins=args.calibration_bins,
        )
        save_reliability_analysis(
            best_payload["labels"],
            best_payload["logits"],
            fold_dir,
            "best_val",
            positive_label=args.positive_label,
            patient_ids=best_payload.get("patient_ids"),
            folds=np.full(len(best_payload["labels"]), fold_idx, dtype=int),
            roi_cosine_similarity=best_payload.get("roi_cosine_similarity"),
        )

    return best_payload


def write_fold_outputs(best_payloads, save_dir, positive_label, calibration_bins=19):
    fold_metrics_path = os.path.join(save_dir, "fold_metrics.csv")
    with open(fold_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Fold", "Best_Epoch", "Positive_Label", "Monitor_Metric", "Monitor_Score", "Acc_AUC_Score"]
            + ["Threshold_Strategy", "Decision_Threshold"]
            + METRIC_NAMES
            + ["BACC", "GMEAN", "Meets_Save_Constraints", "TN", "FP", "FN", "TP"]
        )
        for payload in best_payloads:
            metrics = payload["metrics"]
            writer.writerow(
                [
                    payload["fold"],
                    payload["epoch"],
                    metrics["Positive_Label"],
                    payload["monitor_metric"],
                    payload["score"],
                    payload["acc_auc_score"],
                    payload["threshold_strategy"],
                    payload["decision_threshold"],
                ]
                + [metrics[m] for m in METRIC_NAMES]
                + [
                    metrics["BACC"],
                    metrics["GMEAN"],
                    payload["meets_save_constraints"],
                    metrics["TN"],
                    metrics["FP"],
                    metrics["FN"],
                    metrics["TP"],
                ]
            )

    summary = summarize_fold_metrics([payload["metrics"] for payload in best_payloads])
    summary_path = os.path.join(save_dir, "summary_95ci.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Mean", "Std", "CI95_Low", "CI95_High"])
        writer.writeheader()
        writer.writerows(summary)

    labels = np.concatenate([payload["labels"] for payload in best_payloads])
    preds = np.concatenate([payload["preds"] for payload in best_payloads])
    scores = np.concatenate([payload["scores"] for payload in best_payloads])
    logits = np.concatenate([payload["logits"] for payload in best_payloads])
    roi_cosine_similarity = np.concatenate(
        [
            payload.get("roi_cosine_similarity", np.full(len(payload["labels"]), np.nan, dtype=float))
            for payload in best_payloads
        ]
    )
    patient_ids = np.concatenate(
        [
            payload.get("patient_ids", np.asarray([f"fold{payload['fold']}_sample{idx}" for idx in range(len(payload["labels"]))]))
            for payload in best_payloads
        ]
    )
    folds = np.concatenate([np.full(len(payload["labels"]), payload["fold"], dtype=int) for payload in best_payloads])

    threshold_strategies = {payload["threshold_strategy"] for payload in best_payloads}
    threshold_strategy = threshold_strategies.pop() if len(threshold_strategies) == 1 else "mixed"
    pooled_metrics = compute_binary_metrics(
        labels,
        preds,
        scores,
        positive_label=positive_label,
        logits=logits,
        calibration_bins=calibration_bins,
    )
    pooled_metrics["Threshold_Strategy"] = threshold_strategy
    pooled_metrics["Decision_Threshold"] = "mixed"

    pooled_metrics_csv = os.path.join(save_dir, "pooled_oof_metrics.csv")
    with open(pooled_metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for metric in METRIC_NAMES:
            writer.writerow([metric, pooled_metrics[metric]])
        for metric in ["BACC", "GMEAN", "TN", "FP", "FN", "TP"]:
            writer.writerow([metric, pooled_metrics[metric]])

    pooled_metrics_txt = os.path.join(save_dir, "pooled_oof_metrics.txt")
    with open(pooled_metrics_txt, "w") as f:
        f.write("Pooled out-of-fold metrics\n")
        f.write(f"Positive label: {positive_label} (0=Luminal, 1=non-Luminal)\n")
        for metric in METRIC_NAMES:
            f.write(f"{metric}: {pooled_metrics[metric]:.6f}\n")
        f.write(f"BACC: {pooled_metrics['BACC']:.6f}\n")
        f.write(f"GMEAN: {pooled_metrics['GMEAN']:.6f}\n")
        f.write(f"TN: {pooled_metrics['TN']}\n")
        f.write(f"FP: {pooled_metrics['FP']}\n")
        f.write(f"FN: {pooled_metrics['FN']}\n")
        f.write(f"TP: {pooled_metrics['TP']}\n")

    save_pooled_roc(
        labels,
        scores,
        preds,
        save_dir,
        prefix="pooled_oof",
        positive_label=positive_label,
        threshold_strategy=threshold_strategy,
    )
    save_mean_fold_roc(best_payloads, summary, save_dir, prefix="mean_fold", positive_label=positive_label)
    save_calibration_outputs(logits, labels, save_dir, "pooled_oof", bins=calibration_bins)
    save_calibration_gap_table(logits, labels, save_dir, "pooled_oof", bins=calibration_bins)
    save_reliability_analysis(
        labels,
        logits,
        save_dir,
        "pooled_oof",
        positive_label=positive_label,
        patient_ids=patient_ids,
        folds=folds,
        roi_cosine_similarity=roi_cosine_similarity,
    )

    print(f"Saved fold metrics: {fold_metrics_path}")
    print(f"Saved 95% CI summary: {summary_path}")
    print(f"Saved pooled OOF metrics: {pooled_metrics_csv}")


def train_kfold(args):
    set_seed(args.seed)
    modalities = parse_modalities(args.modalities)
    infos = load_clinical_infos(args.config_path)
    labels = np.asarray([info["label"] for info in infos])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Total label distribution: {dict(sorted(Counter(labels).items()))}")
    print(f"Enabled modalities: {','.join(modalities)}")
    print(f"Positive label for SEN/SPE/PRE/F1/AUC: {args.positive_label} (0=Luminal, 1=non-Luminal)")
    print(f"Best-model monitor metric: {args.monitor_metric}")
    print(f"Best-model save constraints: SEN>={args.min_save_sen}, SPE>={args.min_save_spe}")
    print(f"Validation threshold strategy: {args.threshold_strategy}")
    print(f"Output root: {args.output_root}")
    print(
        "Reliability feedback: "
        f"momentum={args.calibration_gap_momentum}, "
        f"over_gap_weight={args.over_gap_weight}, "
        f"under_gap_weight={args.under_gap_weight}, "
        f"overconfidence_threshold={args.overconfidence_threshold}"
    )

    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{start_time}_{args.model}_{args.backbone}_{args.experiment_name}_KFold{args.num_folds}"
    save_dir = os.path.join(args.output_root, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    save_run_arguments(args, save_dir)

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    best_payloads = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        train_info = [infos[i] for i in train_idx]
        val_info = [infos[i] for i in val_idx]
        best_payload = run_fold(fold_idx, train_info, val_info, args, save_dir, device)
        best_payloads.append(best_payload)

    write_fold_outputs(
        best_payloads,
        save_dir,
        positive_label=args.positive_label,
        calibration_bins=args.calibration_bins,
    )
    print(f"K-fold training complete: {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--model", type=str, default="MSHF", choices=["MSHF", "MSHF_ViT"])
    parser.add_argument("--backbone", type=str, default="ResNet50", choices=["ResNet50", "DenseNet121", "InceptionV3", "VGG16", "ViT-B_16"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--positive_label", type=int, default=0, choices=[0, 1], help="Positive class for SEN/SPE/PRE/F1/AUC. 0=Luminal, 1=non-Luminal.")
    parser.add_argument("--monitor_metric", type=str.upper, default="AUC_BACC", choices=MONITOR_CHOICES, help="Validation metric used to save best_model.pth. AUC_BACC = AUC + balanced accuracy.")
    parser.add_argument("--min_save_sen", type=float, default=0.05, help="Minimum validation sensitivity required for an epoch to be saved as best.")
    parser.add_argument("--min_save_spe", type=float, default=0.05, help="Minimum validation specificity required for an epoch to be saved as best.")
    parser.add_argument(
        "--threshold_strategy",
        type=str.lower,
        default="youden",
        choices=THRESHOLD_STRATEGIES,
        help="Decision rule for binary validation metrics. 'youden' uses the ROC threshold with max TPR-FPR.",
    )
    parser.add_argument("--use_seg", action="store_true")
    parser.add_argument("--loss_type", type=str, default="ce", choices=["cb_focal", "ce"])
    parser.add_argument("--modalities", type=str, default="mg,us,clinical", help="Comma-separated enabled modalities: mg,us,clinical. Examples: mg; us; mg,us; mg,clinical; us,clinical.")
    parser.add_argument("--experiment_name", type=str, default="ABLATION_NoSegGuidance_CrossEntropyLoss_CALIBRATION_FEEDBACK_70_10_20")
    parser.add_argument("--cb_beta", type=float, default=0.99)
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--calibration_bins", type=int, default=19, help="Number of confidence bins for validation calibration reporting.")
    parser.add_argument("--calibration_feedback_weight", type=float, default=0.05, help="Weight for calibration-gap feedback loss from the previous calibration-feedback split epoch.")
    parser.add_argument("--overconfidence_weight", type=float, default=0.05, help="Weight for penalizing high-confidence wrong predictions during training.")
    parser.add_argument("--over_gap_weight", type=float, default=1.0, help="Weight for overconfidence bin gap inside calibration feedback.")
    parser.add_argument("--under_gap_weight", type=float, default=0.25, help="Weight for underconfidence bin gap inside calibration feedback.")
    parser.add_argument("--calibration_gap_momentum", type=float, default=0.8, help="Moving-average momentum for calibration gap feedback. 0 disables smoothing.")
    parser.add_argument("--overconfidence_threshold", type=float, default=0.8, help="Only penalize wrong predictions above this confidence when >0.")
    parser.add_argument("--roi_consistency_weight", type=float, default=0.05, help="Weight for segmentation-prior cross-modal ROI consistency loss.")
    parser.add_argument("--output_root", type=str, default="calibration", help="Root directory for reliability calibration experiments.")
    return parser.parse_args()


if __name__ == "__main__":
    train_kfold(parse_args())
