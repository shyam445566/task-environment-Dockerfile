"""
Verifier for the sensor signal denoising / anomaly detection task.
Ground truth lives only here in tests/, never baked into the
environment image, so the agent cannot read it during its run.
"""
import json
import numpy as np
import pandas as pd

DENOISED_PATH = "/app/output/denoised_signal.csv"
FREQ_PATH = "/app/output/dominant_frequency.json"
ANOMALIES_PATH = "/app/output/anomalies.csv"
GROUND_TRUTH_PATH = "/tests/ground_truth.json"

with open(GROUND_TRUTH_PATH) as f:
    GT = json.load(f)


def test_denoised_signal_close_to_clean():
    df = pd.read_csv(DENOISED_PATH)
    assert list(df.columns[:2]) == ["time", "value"]
    denoised = df["value"].values
    clean = np.array(GT["clean_signal"])
    assert len(denoised) == len(clean), "denoised_signal.csv has wrong number of rows"
    rmse = np.sqrt(np.mean((denoised - clean) ** 2))
    assert rmse < 0.6, f"denoised signal RMSE too high: {rmse:.3f}"


def test_dominant_frequency():
    with open(FREQ_PATH) as f:
        result = json.load(f)
    assert "dominant_frequency_hz" in result
    freq = float(result["dominant_frequency_hz"])
    true_freq = GT["dominant_frequency_hz"]
    assert abs(freq - true_freq) < 0.3, f"dominant frequency {freq} not close to true {true_freq}"


def test_anomalies_detected():
    df = pd.read_csv(ANOMALIES_PATH)
    assert "time" in df.columns
    detected = df["time"].tolist()
    true_times = GT["anomaly_times"]

    tolerance = 0.05
    matched = 0
    used = set()
    for tt in true_times:
        for i, dt in enumerate(detected):
            if i in used:
                continue
            if abs(dt - tt) <= tolerance:
                matched += 1
                used.add(i)
                break

    recall = matched / len(true_times)
    false_positives = len(detected) - matched
    assert recall >= 0.8, f"recall too low: {recall:.2f} ({matched}/{len(true_times)})"
    assert false_positives <= 2, f"too many false positive anomalies: {false_positives}"
