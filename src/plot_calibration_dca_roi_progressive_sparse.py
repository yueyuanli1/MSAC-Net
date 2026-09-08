import argparse
import csv
import datetime
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for import_path in (ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

DEFAULT_CHECKPOINT = (
    "checkpoints/20260507_194702_MSHF_ResNet50_ABLATION_NoSegGuidance_"
    "CrossEntropyLoss_CALIBRATION_FEEDBACK_KFold5/fold_3/best_model.pth"
)
DEFAULT_CHECKPOINT_ROOT = (
    "checkpoints/20260507_194702_MSHF_ResNet50_ABLATION_NoSegGuidance_"
    "CrossEntropyLoss_CALIBRATION_FEEDBACK_KFold5"
)


def parse_modalities(value):
    modalities = [item.strip().lower() for item in value.split(",") if item.strip()]
    valid = {"mg", "us", "clinical"}
    invalid = sorted(set(modalities) - valid)
    if invalid:
        raise ValueError(f"Invalid modalities: {invalid}. Valid choices are {sorted(valid)}")
    if not ({"mg", "us"} & set(modalities)):
        raise ValueError("At least one image modality must be enabled: mg or us.")
    return modalities


def resolve_clinical_path(config, config_path):
    clinical_path = Path(config["clinical_dir"])
    if clinical_path.is_absolute():
        return clinical_path

    candidates = [
        ROOT / clinical_path,
        Path(config_path).resolve().parent / clinical_path.name,
        Path.cwd() / clinical_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_infos(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    clinical_path = resolve_clinical_path(config, config_path)
    with open(clinical_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_fold_from_path(checkpoint_path):
    match = re.search(r"fold_(\d+)", str(checkpoint_path).replace("\\", "/"))
    return int(match.group(1)) if match else None


def get_fold_infos(infos, seed, num_folds, fold, allow_fallback_split=False):
    labels = np.asarray([info["label"] for info in infos], dtype=int)
    try:
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
            if fold_idx == fold:
                return [infos[i] for i in val_idx]
    except Exception as exc:
        if not allow_fallback_split:
            raise RuntimeError(
                "Could not import/use sklearn StratifiedKFold, so the original training fold split cannot be "
                "guaranteed. Install/fix sklearn in this environment or rerun with --allow_fallback_split "
                "only if an approximate deterministic split is acceptable."
            ) from exc
        print(f"Warning: sklearn StratifiedKFold unavailable ({exc}); using a deterministic fallback split.")
        rng = np.random.RandomState(seed)
        fold_indices = [[] for _ in range(num_folds)]
        for label in sorted(np.unique(labels)):
            class_indices = np.where(labels == label)[0]
            rng.shuffle(class_indices)
            for fold_idx, split in enumerate(np.array_split(class_indices, num_folds)):
                fold_indices[fold_idx].extend(split.tolist())
        if 1 <= fold <= num_folds:
            return [infos[i] for i in sorted(fold_indices[fold - 1])]
    raise ValueError(f"Fold {fold} is out of range for num_folds={num_folds}.")


def build_model(args, device, checkpoint_path):
    from models.MSHF_roi_consistency_progressive_sparse import MSHF

    model = MSHF(
        backbone=args.backbone,
        num_classes=args.num_classes,
        modalities=parse_modalities(args.modalities),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing[:8]}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected[:8]}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device, use_seg, positive_label):
    labels, logits, scores, patient_ids = [], [], [], []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting predictions")):
        if use_seg:
            (img_cc, mask_cc), (img_mlo, mask_mlo), (img_us, mask_us), batch_labels, clinical = batch
            mask_cc = mask_cc.to(device)
            mask_mlo = mask_mlo.to(device)
            mask_us = mask_us.to(device)
        else:
            img_cc, img_mlo, img_us, batch_labels, clinical = batch
            mask_cc = mask_mlo = mask_us = None

        img_cc = img_cc.to(device)
        img_mlo = img_mlo.to(device)
        img_us = img_us.to(device)
        clinical = clinical.to(device)

        batch_logits = model(img_mlo, img_cc, img_us, clinical, mask_mlo, mask_cc, mask_us)
        batch_probs = F.softmax(batch_logits, dim=1)

        labels.extend(batch_labels.cpu().numpy().astype(int).tolist())
        logits.extend(batch_logits.cpu().numpy().tolist())
        scores.extend(batch_probs[:, positive_label].cpu().numpy().tolist())

        start = batch_idx * loader.batch_size
        end = start + len(batch_labels)
        patient_ids.extend(loader.dataset.ids[start:end])

    return {
        "patient_ids": np.asarray(patient_ids),
        "labels": np.asarray(labels, dtype=int),
        "logits": np.asarray(logits, dtype=float),
        "scores": np.asarray(scores, dtype=float),
    }


def write_prediction_csv(payload, output_path, positive_label):
    labels_positive = (payload["labels"] == positive_label).astype(int)
    probs = F.softmax(torch.as_tensor(payload["logits"], dtype=torch.float32), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    folds = payload.get("folds", np.full(len(payload["labels"]), np.nan))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sample_index", "fold", "patient_id", "label", "y_true", "pred_label", "y_score", "prob_class0", "prob_class1"]
        )
        for idx, (fold, patient_id, label, y_true, pred, score, prob) in enumerate(
            zip(folds, payload["patient_ids"], payload["labels"], labels_positive, preds, payload["scores"], probs)
        ):
            writer.writerow([idx, fold, patient_id, label, y_true, pred, score, prob[0], prob[1]])


def compute_binary_calibration(y_true, y_score, bins):
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(y_score, bin_edges[1:-1], right=True)

    prob_pred = []
    prob_true = []
    rows = []
    for bin_idx in range(bins):
        in_bin = bin_ids == bin_idx
        if not np.any(in_bin):
            rows.append([bin_idx, bin_edges[bin_idx], bin_edges[bin_idx + 1], 0, np.nan, np.nan])
            continue
        mean_pred = float(np.mean(y_score[in_bin]))
        observed = float(np.mean(y_true[in_bin]))
        prob_pred.append(mean_pred)
        prob_true.append(observed)
        rows.append(
            [
                bin_idx,
                bin_edges[bin_idx],
                bin_edges[bin_idx + 1],
                int(in_bin.sum()),
                mean_pred,
                observed,
            ]
        )
    return np.asarray(prob_pred), np.asarray(prob_true), rows


def write_validation_samples_csv(val_infos, output_path):
    keys = sorted({key for info in val_infos for key in info.keys()})
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(val_infos)


def write_run_metadata(args, output_dir, folds, val_infos, timestamp, checkpoints):
    metadata = {
        "timestamp": timestamp,
        "checkpoint": args.checkpoint,
        "checkpoint_root": args.checkpoint_root,
        "checkpoints": checkpoints,
        "config_path": args.config_path,
        "folds": folds,
        "seed": args.seed,
        "num_folds": args.num_folds,
        "validation_sample_count": len(val_infos),
        "validation_patient_ids": [info.get("id") for info in val_infos],
        "positive_label": args.positive_label,
        "bins": args.bins,
        "modalities": args.modalities,
        "use_seg": args.use_seg,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def parse_folds(value, num_folds):
    if value.lower() == "all":
        return list(range(1, num_folds + 1))
    folds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        folds.append(int(part))
    if not folds:
        raise ValueError("No folds were provided.")
    return folds


def resolve_checkpoint_for_fold(args, fold):
    if args.checkpoint_root:
        checkpoint_root = Path(args.checkpoint_root)
        if not checkpoint_root.is_absolute():
            checkpoint_root = ROOT / checkpoint_root
        return checkpoint_root.resolve() / f"fold_{fold}" / "best_model.pth"
    return Path(args.checkpoint).resolve()


def merge_payloads(payloads):
    return {
        "patient_ids": np.concatenate([payload["patient_ids"] for payload in payloads]),
        "labels": np.concatenate([payload["labels"] for payload in payloads]),
        "logits": np.concatenate([payload["logits"] for payload in payloads]),
        "scores": np.concatenate([payload["scores"] for payload in payloads]),
        "folds": np.concatenate([payload["folds"] for payload in payloads]),
    }


def save_calibration_plot(labels, logits, output_dir, bins, positive_label, apply_posthoc):
    y_true = (labels == positive_label).astype(int)
    probs = F.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()
    raw_score = probs[:, positive_label]

    curves = [("Raw model", raw_score, "#1f77b4")]
    if apply_posthoc:
        from utils.calibration import ConfidenceBinTemperatureCalibrator

        calibrator = ConfidenceBinTemperatureCalibrator(bins=bins)
        tensor_logits = torch.as_tensor(logits, dtype=torch.float32)
        tensor_labels = torch.as_tensor(labels, dtype=torch.long)
        calibrator.fit(tensor_logits, tensor_labels)
        calibrated_logits = calibrator.transform(tensor_logits)
        calibrated_probs = F.softmax(calibrated_logits, dim=1).numpy()
        curves.append(("Post-hoc calibrated", calibrated_probs[:, positive_label], "#d62728"))

    plt.figure(figsize=(6.2, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.5, label="Perfect calibration")

    all_rows = []
    for name, score, color in curves:
        prob_pred, prob_true, rows = compute_binary_calibration(y_true, score, bins)
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2.0, color=color, label=name)
        for row in rows:
            all_rows.append([name] + row)

    plt.xlabel(f"Predicted probability of class {positive_label}")
    plt.ylabel(f"Observed fraction of class {positive_label}")
    plt.title("Calibration Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    png_path = output_dir / "calibration_curve.png"
    svg_path = output_dir / "calibration_curve.svg"
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()

    csv_path = output_dir / "calibration_curve_points.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["curve", "bin", "bin_low", "bin_high", "count", "mean_predicted_probability", "observed_fraction"])
        writer.writerows(all_rows)

    return png_path, svg_path, csv_path


def decision_curve(y_true, y_score, thresholds):
    n = len(y_true)
    rows = []
    prevalence = float(np.mean(y_true))
    for threshold in thresholds:
        predicted_positive = y_score >= threshold
        tp = int(np.sum(predicted_positive & (y_true == 1)))
        fp = int(np.sum(predicted_positive & (y_true == 0)))
        net_benefit = tp / n - fp / n * threshold / (1.0 - threshold)
        treat_all = prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
        rows.append([threshold, net_benefit, treat_all, 0.0, tp, fp])
    return rows


def save_dca_plot(labels, logits, output_dir, positive_label, threshold_min, threshold_max, threshold_step):
    y_true = (labels == positive_label).astype(int)
    probs = F.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()
    y_score = probs[:, positive_label]
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]
    rows = decision_curve(y_true, y_score, thresholds)

    arr = np.asarray(rows, dtype=float)
    plt.figure(figsize=(6.4, 5.2))
    plt.plot(arr[:, 0], arr[:, 1], color="#1f77b4", linewidth=2.2, label="Model")
    plt.plot(arr[:, 0], arr[:, 2], color="#444444", linestyle="--", linewidth=1.6, label="Treat all")
    plt.axhline(0.0, color="#888888", linestyle=":", linewidth=1.6, label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    png_path = output_dir / "dca_curve.png"
    svg_path = output_dir / "dca_curve.svg"
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()

    csv_path = output_dir / "dca_curve_points.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "model_net_benefit", "treat_all_net_benefit", "treat_none_net_benefit", "tp", "fp"])
        writer.writerows(rows)

    return png_path, svg_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot calibration curve and DCA curve for a trained MOTAI3 fold model.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to best_model.pth")
    parser.add_argument(
        "--checkpoint_root",
        default=None,
        help="K-fold checkpoint root containing fold_*/best_model.pth. When set, all requested folds use their own model.",
    )
    parser.add_argument("--config_path", default="configs/config.yaml")
    parser.add_argument("--output_dir", default=None, help="Defaults to <checkpoint_dir>/calibration_dca/<timestamp>")
    parser.add_argument("--no_timestamp", action="store_true", help="Do not append a timestamp subdirectory to output_dir.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index. Defaults to inferring from checkpoint path.")
    parser.add_argument("--folds", default=None, help="Comma-separated folds, or 'all'. Used with --checkpoint_root.")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--allow_fallback_split", action="store_true", help="Allow a non-sklearn fallback split if sklearn is unavailable.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backbone", default="ResNet50", choices=["ResNet50", "DenseNet121", "InceptionV3", "VGG16", "ViT-B_16"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--modalities", default="mg,us,clinical")
    parser.add_argument("--positive_label", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_seg", action="store_true")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--posthoc_calibrated_curve", action="store_true")
    parser.add_argument("--threshold_min", type=float, default=0.01)
    parser.add_argument("--threshold_max", type=float, default=0.99)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--device", default=None, help="Examples: cuda, cuda:0, cpu. Defaults to cuda if available.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint = str((ROOT / args.checkpoint).resolve()) if not os.path.isabs(args.checkpoint) else args.checkpoint
    if args.checkpoint_root and not os.path.isabs(args.checkpoint_root):
        args.checkpoint_root = str((ROOT / args.checkpoint_root).resolve())
    args.config_path = str((ROOT / args.config_path).resolve()) if not os.path.isabs(args.config_path) else args.config_path

    if args.checkpoint_root:
        folds = parse_folds(args.folds or "all", args.num_folds)
    else:
        fold = args.fold or infer_fold_from_path(args.checkpoint)
        if fold is None:
            raise ValueError("Could not infer fold from checkpoint path. Please pass --fold, for example --fold 3.")
        folds = [fold]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    elif args.checkpoint_root:
        base_output_dir = Path(args.checkpoint_root).resolve() / "pooled_oof_calibration_dca"
    else:
        base_output_dir = Path(args.checkpoint).resolve().parent / "calibration_dca"
    output_dir = base_output_dir if args.no_timestamp else base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    infos = load_infos(args.config_path)
    print(f"Using device: {device}")
    print(f"Folds: {folds}")
    print(f"Positive label for curves: {args.positive_label}")
    print(f"Output dir: {output_dir}")

    from dataloader.load_data import MyDataset

    payloads = []
    all_val_infos = []
    checkpoint_records = {}
    for fold in folds:
        checkpoint_path = resolve_checkpoint_for_fold(args, fold)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {checkpoint_path}")
        val_infos = get_fold_infos(
            infos,
            seed=args.seed,
            num_folds=args.num_folds,
            fold=fold,
            allow_fallback_split=args.allow_fallback_split,
        )
        for info in val_infos:
            copied = dict(info)
            copied["fold"] = fold
            all_val_infos.append(copied)

        print(f"Fold {fold}: checkpoint={checkpoint_path}")
        print(f"Fold {fold}: validation samples={len(val_infos)}")
        dataset = MyDataset(val_infos, args.config_path, use_seg=args.use_seg, is_train=False)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        model = build_model(args, device, checkpoint_path)
        fold_payload = collect_predictions(model, loader, device, args.use_seg, args.positive_label)
        fold_payload["folds"] = np.full(len(fold_payload["labels"]), fold, dtype=int)
        payloads.append(fold_payload)
        checkpoint_records[str(fold)] = str(checkpoint_path)

    payload = merge_payloads(payloads)
    write_validation_samples_csv(all_val_infos, output_dir / "validation_samples.csv")
    write_run_metadata(args, output_dir, folds, all_val_infos, timestamp, checkpoint_records)

    predictions_csv = output_dir / "predictions.csv"
    write_prediction_csv(payload, predictions_csv, args.positive_label)
    np.savez(
        output_dir / "predictions.npz",
        patient_ids=payload["patient_ids"],
        labels=payload["labels"],
        logits=payload["logits"],
        scores=payload["scores"],
        folds=payload["folds"],
        positive_label=np.asarray(args.positive_label, dtype=int),
        fold=np.asarray(folds, dtype=int),
    )

    calibration_outputs = save_calibration_plot(
        payload["labels"],
        payload["logits"],
        output_dir,
        bins=args.bins,
        positive_label=args.positive_label,
        apply_posthoc=args.posthoc_calibrated_curve,
    )
    dca_outputs = save_dca_plot(
        payload["labels"],
        payload["logits"],
        output_dir,
        positive_label=args.positive_label,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )

    print(f"Saved predictions: {predictions_csv}")
    print(f"Saved calibration outputs: {', '.join(str(path) for path in calibration_outputs)}")
    print(f"Saved DCA outputs: {', '.join(str(path) for path in dca_outputs)}")


if __name__ == "__main__":
    main()
