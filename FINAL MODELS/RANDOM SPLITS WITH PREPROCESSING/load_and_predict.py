import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.signal import butter, filtfilt
from math import atan2, asin


SENSORS = ["L5", "T4", "C7", "T12"]
ACC_FEATURES = ["Acceleration X(g)", "Acceleration Y(g)", "Acceleration Z(g)"]
GYRO_FEATURES = ["Angular velocity X(°/s)", "Angular velocity Y(°/s)", "Angular velocity Z(°/s)"]
QUAT_FEATURES = ["Quaternions 0()", "Quaternions 1()", "Quaternions 2()", "Quaternions 3()"]

SENSOR_SLICES = [
    slice(0, 6),
    slice(6, 12),
    slice(12, 18),
    slice(18, 24),
]


def detect_mag_cols(df, sensor):
    cols = df.columns
    out = []

    for axis in ["x", "y", "z"]:
        matches = [
            c
            for c in cols
            if c.lower().startswith(sensor.lower() + "_magnetic field")
            and axis in c.lower()
        ]
        if not matches:
            raise KeyError(f"Missing magnetometer axis={axis} for sensor={sensor}\n{list(cols)}")

        out.append(matches[0])

    return out


def quat_reorder_to_wxyz(qx, qy, qz, qw):
    return np.array([qw, qx, qy, qz], dtype=np.float32)


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float32)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def rotate_vector_by_quaternion(q, v):
    vq = np.array([0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_multiply(quat_multiply(q, vq), quat_conjugate(q))[1:]


def quaternion_to_euler(q):
    w, x, y, z = q
    yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = asin(np.clip(2 * (w * y - z * x), -1, 1))
    roll = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return yaw, pitch, roll


def hampel_filter(x, window_size=5, n_sigmas=3):
    x = x.copy()
    y = x.copy()
    n = len(x)
    k = window_size

    for i in range(k, n - k):
        window = x[i - k : i + k + 1]
        median = np.nanmedian(window)
        mad = np.nanmedian(np.abs(window - median))

        if mad < 1e-6:
            continue

        if abs(x[i] - median) > n_sigmas * 1.4826 * mad:
            y[i] = median

    return y


def butter_lowpass_filter(x, cutoff=3.0, fs=50, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    try:
        return filtfilt(b, a, x, axis=0)
    except Exception:
        return x


def remove_bias(x):
    return x - np.nanmean(x, axis=0)


def preprocess_single_file(raw_path):
    df = pd.read_csv(raw_path)

    keep = []
    for col in df.columns:
        for s in SENSORS:
            if col.startswith(f"{s}_Acceleration"):
                keep.append(col)
            elif col.startswith(f"{s}_Angular velocity"):
                keep.append(col)
            elif col.startswith(f"{s}_Quaternions"):
                keep.append(col)
            elif col.startswith(f"{s}_Magnetic field"):
                keep.append(col)

    df = df[keep].copy().apply(pd.to_numeric, errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.interpolate(limit_direction="both", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    blocks = []

    for sensor in SENSORS:
        acc = df[[f"{sensor}_{f}" for f in ACC_FEATURES]].values
        gyr = df[[f"{sensor}_{f}" for f in GYRO_FEATURES]].values
        quat_raw = df[[f"{sensor}_{f}" for f in QUAT_FEATURES]].values
        mag_cols = detect_mag_cols(df, sensor)
        mag = df[mag_cols].values

        quats = []
        for (qx, qy, qz, qw) in quat_raw:
            q = quat_reorder_to_wxyz(qx, qy, qz, qw)
            q = q / np.linalg.norm(q)
            quats.append(q)
        quats = np.array(quats)

        acc = remove_bias(acc)
        gyr = remove_bias(gyr)
        mag = remove_bias(mag)

        for i in range(acc.shape[1]):
            acc[:, i] = hampel_filter(acc[:, i])
        for i in range(gyr.shape[1]):
            gyr[:, i] = hampel_filter(gyr[:, i])
        for i in range(mag.shape[1]):
            mag[:, i] = hampel_filter(mag[:, i])

        acc = butter_lowpass_filter(acc)
        gyr = butter_lowpass_filter(gyr)
        mag = butter_lowpass_filter(mag)

        a_vecs, g_vecs, m_vecs, e_vecs = [], [], [], []
        for t in range(len(df)):
            q = quats[t]
            a_vecs.append(rotate_vector_by_quaternion(q, acc[t]))
            g_vecs.append(rotate_vector_by_quaternion(q, gyr[t]))
            m_vecs.append(rotate_vector_by_quaternion(q, mag[t]))
            e_vecs.append(quaternion_to_euler(q))

        a_vecs = np.array(a_vecs)
        g_vecs = np.array(g_vecs)
        m_vecs = np.array(m_vecs)
        e_vecs = np.array(e_vecs)

        a_mag = np.linalg.norm(a_vecs, axis=1, keepdims=True)
        g_mag = np.linalg.norm(g_vecs, axis=1, keepdims=True)
        m_mag = np.linalg.norm(m_vecs, axis=1, keepdims=True)

        block = np.concatenate([a_vecs, g_vecs, m_vecs, a_mag, g_mag, m_mag, e_vecs], axis=1)
        blocks.append(block)

    return np.concatenate(blocks, axis=1)


def select_acc_euler(x):
    selected = []
    for i in range(4):
        start = i * 15
        acc = x[:, start + 0 : start + 3]
        euler = x[:, start + 12 : start + 15]
        selected.append(acc)
        selected.append(euler)
    return np.concatenate(selected, axis=1)


def make_windows(x, win_len, stride):
    t, f = x.shape
    if t < win_len:
        w = np.zeros((win_len, f))
        w[:t] = x
        return w[None, :, :]

    windows = []
    for i in range(0, t - win_len + 1, stride):
        windows.append(x[i : i + win_len])
    return np.array(windows)


def compute_sensor_norm(x_windows):
    means = []
    stds = []

    for s in SENSOR_SLICES:
        flat = x_windows[:, :, s].reshape(-1, 6)
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True) + 1e-8
        means.append(mean)
        stds.append(std)

    return means, stds


def apply_sensor_norm(x_windows, means, stds):
    x_norm = x_windows.copy()
    for s, mean, std in zip(SENSOR_SLICES, means, stds):
        x_norm[:, :, s] = (x_norm[:, :, s] - mean) / std
    return x_norm


def build_model(input_shape, num_classes):
    inp = keras.Input(shape=input_shape)
    x = inp

    l5 = layers.Lambda(lambda x: x[:, :, 0:6])(x)
    t4 = layers.Lambda(lambda x: x[:, :, 6:12])(x)
    c7 = layers.Lambda(lambda x: x[:, :, 12:18])(x)
    t12 = layers.Lambda(lambda x: x[:, :, 18:24])(x)

    def sensor_block(x):
        x = layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        attention = layers.Dense(x.shape[-1], activation="tanh")(x)
        attention = layers.Softmax(axis=-1)(attention)
        x = layers.Multiply()([x, attention])
        return x

    l5 = sensor_block(l5)
    t4 = sensor_block(t4)
    c7 = sensor_block(c7)
    t12 = sensor_block(t12)

    sensors = layers.Lambda(lambda x: tf.stack(x, axis=2))([l5, t4, c7, t12])
    attention = layers.Dense(1, activation="tanh")(sensors)
    attention = layers.Softmax(axis=2)(attention)
    sensors = layers.Multiply()([sensors, attention])
    fused = layers.Lambda(lambda x: tf.reduce_sum(x, axis=2))(sensors)

    x = layers.GlobalAveragePooling1D()(fused)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    return model


def iter_csvs(root):
    root = Path(root)
    for path in root.rglob("*.csv"):
        yield path


def load_norms(norm_npz):
    data = np.load(norm_npz, allow_pickle=True)
    means = [data[f"mean_{i}"] for i in range(4)]
    stds = [data[f"std_{i}"] for i in range(4)]
    return means, stds


def save_norms(norm_npz, means, stds):
    np.savez(norm_npz, **{f"mean_{i}": m for i, m in enumerate(means)}, **{f"std_{i}": s for i, s in enumerate(stds)})


def compute_norms_from_dir(norm_dir, win_len, stride):
    all_windows = []
    for csv_path in iter_csvs(norm_dir):
        x_full = preprocess_single_file(csv_path)
        x_full = select_acc_euler(x_full)
        x_win = make_windows(x_full, win_len, stride)
        all_windows.append(x_win)

    if not all_windows:
        raise FileNotFoundError(f"No CSV files found under: {norm_dir}")

    x_windows = np.concatenate(all_windows, axis=0)
    return compute_sensor_norm(x_windows)


def main():
    parser = argparse.ArgumentParser(description="Load weights and predict from a CSV.")
    parser.add_argument("--weights", required=True, help="Path to model_weights_best.weights.h5")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--postures_root", default="wide_data", help="Root folder with posture subfolders")
    parser.add_argument("--norm_npz", default="norm_stats.npz", help="Path to saved normalization .npz")
    parser.add_argument("--norm_dir", default=None, help="Folder to compute normalization if norm_npz missing")
    parser.add_argument("--win_len", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)

    args = parser.parse_args()

    postures_root = Path(args.postures_root)
    postures = sorted([p.name for p in postures_root.iterdir() if p.is_dir()])
    if not postures:
        raise FileNotFoundError(f"No posture folders found under: {postures_root}")

    norm_path = Path(args.norm_npz)
    if norm_path.exists():
        means, stds = load_norms(norm_path)
    else:
        if not args.norm_dir:
            raise FileNotFoundError(
                f"Normalization stats not found at {norm_path}. "
                "Provide --norm_dir to compute them or --norm_npz to point to an existing file."
            )
        means, stds = compute_norms_from_dir(args.norm_dir, args.win_len, args.stride)
        save_norms(norm_path, means, stds)
        print("Saved normalization stats to:", norm_path)

    model = build_model((args.win_len, 24), num_classes=len(postures))
    model.load_weights(args.weights)

    x_full = preprocess_single_file(args.csv)
    x_full = select_acc_euler(x_full)
    x_windows = make_windows(x_full, args.win_len, args.stride)
    x_windows = apply_sensor_norm(x_windows, means, stds)

    probs = model.predict(x_windows, verbose=0)
    summed = np.sum(np.log(probs + 1e-8), axis=0)
    pred_label = int(np.argmax(summed))

    print("Predicted label id:", pred_label)
    print("Predicted posture:", postures[pred_label])


if __name__ == "__main__":
    main()
