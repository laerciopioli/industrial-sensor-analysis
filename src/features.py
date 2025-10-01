# src/features.py
import numpy as np
import pandas as pd
import json
from scipy.fft import rfft, rfftfreq
import pywt
from scipy import stats

def split_sensors_from_row(row, meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    shapes = meta["sensor_shapes"]
    use_idxs = meta["use_indices"]
    # keep only used sensors shapes
    shapes_used = [shapes[i] for i in use_idxs]
    splits = np.cumsum(shapes_used)[:-1]
    arr = np.array(row, dtype=float)  # força todos os valores a serem floats
    sensors = [np.array(s, dtype=float) for s in np.split(arr, splits)]  # garante float em cada sensor
    return sensors


def features_from_window(window, sr=10000.0):
    # window: 1D np.array
    feats = {}
    feats["mean"] = np.mean(window)
    feats["std"] = np.std(window)
    feats["rms"] = np.sqrt(np.mean(window**2))
    feats["max"] = np.max(window)
    feats["min"] = np.min(window)
    feats["skew"] = stats.skew(window)
    feats["kurtosis"] = stats.kurtosis(window)
    # FFT features
    fft_vals = np.abs(rfft(window))
    freqs = rfftfreq(len(window), 1/sr)
    if fft_vals.sum() > 0:
        feats["spec_centroid"] = (freqs * fft_vals).sum() / fft_vals.sum()
    else:
        feats["spec_centroid"] = 0.0
    dom_idx = np.argmax(fft_vals[1:]) + 1
    feats["dom_freq"] = freqs[dom_idx]
    # band energies
    bands = [(0,50),(50,500),(500,2000),(2000,5000)]
    for i,(lo,hi) in enumerate(bands):
        mask = (freqs>=lo)&(freqs<hi)
        feats[f"band_{i}_energy"] = float(fft_vals[mask].sum())
    # wavelet features (db4)
    try:
        coeffs = pywt.wavedec(window, "db4", level=3)
        for i,c in enumerate(coeffs):
            feats[f"wcoef_{i}_mean"]=np.mean(c)
            feats[f"wcoef_{i}_std"]=np.std(c)
    except Exception:
        pass
    return feats

def extract_features_dataframe(df, meta_path, sr=10000.0, max_rows=None):
    """
    df: DataFrame with concatenated numeric columns + Classe
    meta_path: path to meta_sensors.json
    returns: (X_features_df, y)
    """
    rows = df.shape[0] if max_rows is None else min(df.shape[0], max_rows)
    out = []
    y = []
    for i in range(rows):
        row = df.iloc[i, :-1].values  # remove classe col
        sensors = split_sensors_from_row(row, meta_path)
        feats_row = {}
        for s_idx, s in enumerate(sensors):
            feats = features_from_window(s, sr=sr)
            # prefix keys
            feats_row.update({f"s{s_idx+1}_{k}": v for k,v in feats.items()})
        out.append(feats_row)
        y.append(df.iloc[i, -1])
    Xf = pd.DataFrame(out)
    return Xf, pd.Series(y)
