import numpy as np
import pandas as pd
import json
from scipy.signal import butter, filtfilt

INPUT_PATH = "/app/data/sensor_signal.csv"
OUT_DENOISED = "/app/output/denoised_signal.csv"
OUT_FREQ = "/app/output/dominant_frequency.json"
OUT_ANOMALIES = "/app/output/anomalies.csv"

FS = 50.0  # sampling rate (Hz) -- given in instruction.md

df = pd.read_csv(INPUT_PATH)
t = df["time"].values
raw = df["value"].values

# 1. Low-pass Butterworth filter, cutoff 5 Hz, order 4
cutoff = 5.0
nyq = FS / 2.0
b, a = butter(4, cutoff / nyq, btype="low")
denoised = filtfilt(b, a, raw)

pd.DataFrame({"time": t, "value": denoised}).to_csv(OUT_DENOISED, index=False)

# 2. Dominant frequency via FFT of denoised signal (exclude DC bin)
n = len(denoised)
fft_vals = np.fft.rfft(denoised)
fft_freqs = np.fft.rfftfreq(n, d=1.0 / FS)
mags = np.abs(fft_vals)
mags[0] = 0
dominant_freq = float(fft_freqs[np.argmax(mags)])

with open(OUT_FREQ, "w") as f:
    json.dump({"dominant_frequency_hz": dominant_freq}, f)

# 3. Anomaly detection via local residual thresholding
residual = raw - denoised
window = 21
half = window // 2
anomaly_times = []
for i in range(n):
    lo = max(0, i - half)
    hi = min(n, i + half + 1)
    local = np.delete(residual[lo:hi], min(i, half) if hi - lo == window else 0)
    thresh = np.mean(local) + 4 * np.std(local)
    if abs(residual[i]) > max(thresh, 1.0):
        anomaly_times.append(t[i])

pd.DataFrame({"time": anomaly_times}).to_csv(OUT_ANOMALIES, index=False)
