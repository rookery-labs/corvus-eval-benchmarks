import numpy as np

from src.metrics.ece import apply_temperature_scaling, compute_ece, optimize_temperature


def test_perfectly_calibrated_predictions():
    # 100 samples with 0.8 confidence and 80% accuracy
    np.random.seed(42)
    n = 1000
    confidences = np.full(n, 0.85)
    # Exactly 85% correct predictions
    correct_mask = np.random.rand(n) < 0.85
    ground_truth = np.zeros(n, dtype=int)
    predictions = np.where(correct_mask, 0, 1)

    ece, mce, _stats = compute_ece(confidences, predictions, ground_truth, n_bins=10)
    # ECE should be very close to 0 (|0.85 - 0.85| approx 0)
    assert ece < 0.05
    assert mce < 0.05


def test_overconfident_uncalibrated_model():
    n = 500
    # Overconfident: predicts with 0.99 confidence but only 50% accuracy
    confidences = np.full(n, 0.99)
    ground_truth = np.zeros(n, dtype=int)
    predictions = np.random.choice([0, 1], size=n, p=[0.5, 0.5])

    ece, mce, _ = compute_ece(confidences, predictions, ground_truth, n_bins=10)
    # ECE must capture the large calibration gap (~0.49)
    assert ece > 0.40
    assert mce > 0.40


def test_temperature_optimization_calibration_improvement():
    np.random.seed(1337)
    n = 400
    k = 4
    # Generate overconfident raw logits
    logits = np.random.randn(n, k) * 6.0
    ground_truth = np.random.randint(0, k, size=n)

    # Before calibration
    uncalibrated_probs = apply_temperature_scaling(logits, 1.0)
    conf_before = np.max(uncalibrated_probs, axis=1)
    pred_before = np.argmax(uncalibrated_probs, axis=1)
    ece_before, _, _ = compute_ece(conf_before, pred_before, ground_truth, n_bins=10)

    # Optimize temperature
    opt_t = optimize_temperature(logits, ground_truth)
    assert opt_t > 1.0  # Temperature should increase to soften overconfident logits

    # After calibration
    calibrated_probs = apply_temperature_scaling(logits, opt_t)
    conf_after = np.max(calibrated_probs, axis=1)
    pred_after = np.argmax(calibrated_probs, axis=1)
    ece_after, _, _ = compute_ece(conf_after, pred_after, ground_truth, n_bins=10)

    assert ece_after < ece_before
