import copy
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def max_class_probability(logits):
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


def brier_score(logits, labels, num_classes=None):
    if num_classes is None:
        num_classes = logits.size(-1)
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).float()
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))


def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels.long()).float().mean()


def expected_calibration_error(logits, labels, bins=15):
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels.long())

    ece = torch.zeros(1, device=logits.device)
    boundaries = torch.linspace(0, 1, bins + 1, device=logits.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return ece


def maximum_calibration_error(logits, labels, bins=15):
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels.long())

    mce = torch.zeros(1, device=logits.device)
    boundaries = torch.linspace(0, 1, bins + 1, device=logits.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        if in_bin.float().mean().item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            mce = torch.maximum(mce, torch.abs(conf_in_bin - acc_in_bin))
    return mce


class AdaptiveECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super().__init__()
        self.n_bins = n_bins

    def _histedges_equal_n(self, values):
        n = len(values)
        return np.interp(
            np.linspace(0, n, self.n_bins + 1),
            np.arange(n),
            np.sort(values),
        )

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels.long())

        edges = self._histedges_equal_n(confidences.detach().cpu().numpy())
        ece = torch.zeros(1, device=logits.device)
        for lower, upper in zip(edges[:-1], edges[1:]):
            in_bin = confidences.gt(float(lower)) & confidences.le(float(upper))
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                acc_in_bin = accuracies[in_bin].float().mean()
                conf_in_bin = confidences[in_bin].mean()
                ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
        return ece


def calibration_report(before_logits, labels, after_logits=None, prefix=""):
    logits = before_logits if after_logits is None else after_logits
    nll = nn.CrossEntropyLoss()(logits, labels.long())
    result = {
        f"{prefix}acc": accuracy(logits, labels).item(),
        f"{prefix}ece": expected_calibration_error(logits, labels).item(),
        f"{prefix}ada_ece": AdaptiveECELoss()(logits, labels).item(),
        f"{prefix}mce": maximum_calibration_error(logits, labels).item(),
        f"{prefix}nll": nll.item(),
        f"{prefix}brier": brier_score(logits, labels).item(),
    }
    return result


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def scale(self, logits):
        temperature = self.temperature.clamp_min(1e-8)
        return logits / temperature

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        device = logits.device
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.scale(logits), labels.long())
            loss.backward()
            return loss

        optimizer.step(closure)
        return self


class ConfidenceBinTemperatureCalibrator(nn.Module):
    def __init__(self, bins=19, max_temperature=5.0):
        super().__init__()
        self.bins = bins
        self.max_temperature = max_temperature
        self.temperatures = [1.0] * bins

    def get_bin_index(self, logits):
        confidences = max_class_probability(logits).detach()
        boundaries = torch.linspace(0, 1, self.bins + 1, device=logits.device)
        bin_index = torch.bucketize(confidences, boundaries[1:-1], right=False)
        return bin_index.clamp(0, self.bins - 1)

    def fit(self, logits, labels):
        with torch.no_grad():
            correct = logits.argmax(dim=-1).eq(labels.long())
            bin_index = self.get_bin_index(logits)
            for i in range(self.bins):
                mask = bin_index.eq(i)
                if mask.sum().item() == 0:
                    self.temperatures[i] = 1.0
                    continue
                target_acc = correct[mask].float().mean()
                self.temperatures[i] = self._binary_search_temperature(logits[mask], target_acc)
        return self

    def _binary_search_temperature(
        self,
        logits,
        target_confidence,
        min_temperature=1e-8,
        epsilon=1e-8,
        max_iter=100,
    ):
        low = min_temperature
        high = self.max_temperature
        temperature = 1.0
        for _ in range(max_iter):
            temperature = (low + high) / 2
            confidence = max_class_probability(logits / temperature).mean()
            if confidence > target_confidence:
                low = temperature
            else:
                high = temperature
            if torch.abs(confidence - target_confidence).item() < epsilon:
                break
        return float(temperature)

    def transform(self, logits):
        calibrated = copy.deepcopy(logits)
        bin_index = self.get_bin_index(logits)
        for i, temperature in enumerate(self.temperatures):
            mask = bin_index.eq(i)
            calibrated[mask] = calibrated[mask] / temperature
        return calibrated


def iterative_confidence_calibration(
    calibration_logits,
    calibration_labels,
    target_logits,
    target_labels=None,
    bins=19,
    times=6,
):
    history = defaultdict(list)
    best_logits = target_logits
    best_mce = float("inf")

    train_logits = calibration_logits
    test_logits = target_logits
    for step in range(times):
        calibrator = ConfidenceBinTemperatureCalibrator(bins=bins)
        if step != 0:
            calibrator.fit(train_logits, calibration_labels)

        train_logits = calibrator.transform(train_logits)
        test_logits = calibrator.transform(test_logits)

        train_report = calibration_report(train_logits, calibration_labels)
        for key, value in train_report.items():
            history[f"train_{key}"].append(value)

        if target_labels is not None:
            test_report = calibration_report(test_logits, target_labels)
            for key, value in test_report.items():
                history[f"target_{key}"].append(value)

        if train_report["mce"] < best_mce:
            best_mce = train_report["mce"]
            best_logits = test_logits

    return best_logits, dict(history)
