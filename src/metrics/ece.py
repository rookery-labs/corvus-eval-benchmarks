import numpy as np
from typing import Tuple, Dict

def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        confidences: (N,) array of maximum prediction probabilities in [0, 1]
        predictions: (N,) array of predicted class indices
        ground_truth: (N,) array of true class labels
        n_bins: number of confidence bins (default 10)
        
    Returns:
        (ece, mce, bin_stats)
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    ground_truth = np.asarray(ground_truth, dtype=np.int64)

    assert len(confidences) == len(predictions) == len(ground_truth), "Array lengths must match"

    n_samples = len(confidences)
    if n_samples == 0:
        return 0.0, 0.0, {}

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    accuracies = predictions == ground_truth

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    ece = 0.0
    mce = 0.0

    for m in range(n_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]

        if m == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        count = np.sum(in_bin)
        bin_counts[m] = count

        if count > 0:
            acc = np.mean(accuracies[in_bin])
            conf = np.mean(confidences[in_bin])
            bin_accuracies[m] = acc
            bin_confidences[m] = conf

            gap = np.abs(acc - conf)
            ece += (count / n_samples) * gap
            mce = max(mce, gap)

    bin_stats = {
        "counts": bin_counts,
        "accuracies": bin_accuracies,
        "confidences": bin_confidences,
        "boundaries": bin_boundaries,
    }

    return float(ece), float(mce), bin_stats


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scales logits and applies softmax."""
    temp = max(float(temperature), 1e-6)
    scaled = logits / temp
    scaled -= np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def optimize_temperature(logits: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Optimizes temperature T using grid search and Brent optimization
    to minimize cross-entropy / negative log likelihood.
    """
    from scipy.optimize import minimize_scalar

    def nll(temp):
        probs = apply_temperature_scaling(logits, temp)
        n = len(ground_truth)
        log_likelihood = np.log(np.clip(probs[np.arange(n), ground_truth], 1e-12, 1.0))
        return -np.mean(log_likelihood)

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)
