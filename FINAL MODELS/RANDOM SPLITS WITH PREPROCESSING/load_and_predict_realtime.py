import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
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


RECORD_ROOT = Path(r"D:\WitMotion(V2024.12.27.0)\Record")

# Fill these with your actual device IDs once printed.
SENSOR_ID_MAP = {
    "ed:35:33:d3:6c:f8": "C7",
    "ed:40:fe:65:30:6c": "T4",
    "f6:90:cc:01:6d:25": "T12",
    "e3:ca:2d:fd:e0:8c": "L5",
}

EXPECTED_SENSORS = ["C7", "T4", "T12", "L5"]
FEATURE_KEYWORDS = ["Acceleration", "Angular velocity", "Quaternions", "Magnetic field"]
DEVICE_COL_CANDIDATES = [
    "Device",
    "Device Name",
    "DeviceName",
    "Sensor",
    "SensorID",
    "Device ID",
    "DeviceID",
    "SN",
    "Mac",
    "MAC",
    "MacAddress",
    "Address",
    "Name",
]
TIME_COL_CANDIDATES = [
    "Time",
    "Timestamp",
    "Time(s)",
    "Time (s)",
    "DateTime",
    "Date",
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


def preprocess_dataframe(df):
    df = df.copy()

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


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_device_column(df):
    col = find_column(df, DEVICE_COL_CANDIDATES)
    if col is not None:
        return col

    # Fallback: try fuzzy matches for device/mac/id columns.
    for candidate in df.columns:
        name = str(candidate).lower()
        if "device" in name or "mac" in name or "address" in name:
            return candidate

    return None


def is_likely_timestamp(series):
    if series.empty:
        return False
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
    return parsed.notna().mean() > 0.8


def is_likely_device_id(series):
    if series.empty:
        return False
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    mac_like = sample.str.contains(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}", regex=True)
    return mac_like.mean() > 0.5


def infer_device_column(df):
    for col in DEVICE_COL_CANDIDATES:
        if col in df.columns and is_likely_device_id(df[col]):
            return col

    for col in df.columns:
        if is_likely_device_id(df[col]):
            return col

    object_cols = [c for c in df.columns if df[c].dtype == object]
    if not object_cols:
        return None

    unique_counts = {c: df[c].nunique(dropna=True) for c in object_cols}
    return min(unique_counts, key=unique_counts.get)


def infer_time_column(df):
    for col in TIME_COL_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        if is_likely_timestamp(df[col]):
            return col
    return None


def list_unique_devices(df, device_col):
    if device_col is None:
        return []
    return sorted(df[device_col].dropna().astype(str).unique().tolist())


def filter_feature_columns(df):
    feature_cols = []
    for col in df.columns:
        if any(key in col for key in FEATURE_KEYWORDS):
            feature_cols.append(col)
    return feature_cols


def long_to_wide_latest(raw_df, device_col, time_col, allow_single_sensor_mirror=False):
    if raw_df.empty:
        return pd.DataFrame()

    unique_devices = list_unique_devices(raw_df, device_col)
    if unique_devices:
        print("Unique device IDs found:", unique_devices)

    if not SENSOR_ID_MAP:
        print("SENSOR_ID_MAP is empty. Fill it with the device IDs printed above.")
        return pd.DataFrame()

    missing_map = [d for d in unique_devices if d not in SENSOR_ID_MAP]
    if missing_map:
        print("Unmapped device IDs:", missing_map)
        return pd.DataFrame()

    df = raw_df.copy()
    df[device_col] = df[device_col].astype(str)
    df["__sensor"] = df[device_col].map(SENSOR_ID_MAP)
    df = df[df["__sensor"].isin(EXPECTED_SENSORS)]

    feature_cols = filter_feature_columns(df)
    if not feature_cols:
        print("No feature columns found in live CSV.")
        return pd.DataFrame()

    if time_col and time_col in df.columns:
        df = df.sort_values(time_col)

    latest_rows = {}
    wide_rows = []

    for _, row in df.iterrows():
        sensor = row["__sensor"]
        latest_rows[sensor] = row

        if allow_single_sensor_mirror and len(latest_rows) == 1 and len(unique_devices) == 1:
            # Single-device test mode: mirror one sensor to all expected sensors.
            only_row = next(iter(latest_rows.values()))
            for missing_sensor in EXPECTED_SENSORS:
                latest_rows.setdefault(missing_sensor, only_row)

        if all(s in latest_rows for s in EXPECTED_SENSORS):
            wide_row = {}
            for sensor_name in EXPECTED_SENSORS:
                for col in feature_cols:
                    wide_row[f"{sensor_name}_{col}"] = latest_rows[sensor_name][col]

            if time_col and time_col in row:
                wide_row["__time"] = row[time_col]

            wide_rows.append(wide_row)

    wide_df = pd.DataFrame(wide_rows)
    if wide_df.empty:
        return wide_df

    expected_cols = [f"{sensor}_{col}" for sensor in EXPECTED_SENSORS for col in feature_cols]
    missing_cols = [c for c in expected_cols if c not in wide_df.columns]
    if missing_cols:
        print("Missing columns in wide_df:", missing_cols)
        return pd.DataFrame()

    print("wide_df shape:", wide_df.shape)
    return wide_df


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


def find_latest_data_csv(record_root, retry_sleep=0.2):
    print("Waiting for latest WitMotion CSV...")
    while True:
        csv_path = get_latest_data_csv_once(record_root)
        if csv_path is None:
            time.sleep(retry_sleep)
            continue

        print("Using CSV:", csv_path)
        return csv_path


def get_latest_data_csv_once(record_root):
    if not record_root.exists():
        return None

    date_dirs = [p for p in record_root.iterdir() if p.is_dir()]
    if not date_dirs:
        return None

    latest_date = max(date_dirs, key=lambda p: p.stat().st_mtime)
    session_dirs = [p for p in latest_date.iterdir() if p.is_dir()]
    if not session_dirs:
        return None

    latest_session = max(session_dirs, key=lambda p: p.stat().st_mtime)
    csv_path = latest_session / "data_0.csv"
    if not csv_path.exists():
        return None

    return csv_path


def read_new_rows(csv_path, last_processed_row):
    if last_processed_row == 0:
        df = pd.read_csv(csv_path)
        return df, len(df)

    skip = range(1, last_processed_row + 1)
    df = pd.read_csv(csv_path, skiprows=skip)
    return df, last_processed_row + len(df)


def main():
    parser = argparse.ArgumentParser(description="Load weights and predict from a CSV.")
    parser.add_argument("--weights", required=True, help="Path to model_weights_best.weights.h5")
    parser.add_argument("--postures_root", default="wide_data", help="Root folder with posture subfolders")
    parser.add_argument("--norm_npz", default="norm_stats.npz", help="Path to saved normalization .npz")
    parser.add_argument("--norm_dir", default=None, help="Folder to compute normalization if norm_npz missing")
    parser.add_argument("--win_len", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--record_root", default=str(RECORD_ROOT), help="WitMotion Record root folder")
    parser.add_argument("--csv", default=None, help="Optional fixed CSV path (overrides live discovery)")
    parser.add_argument("--buffer_mult", type=int, default=2, help="Keep at most win_len * buffer_mult raw rows")
    parser.add_argument("--min_new_rows", type=int, default=1, help="Minimum new rows before processing")
    parser.add_argument("--poll_sleep", type=float, default=0.05, help="Sleep seconds between polls when idle")

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

    record_root = Path(args.record_root)
    csv_path = Path(args.csv) if args.csv else find_latest_data_csv(record_root)

    last_processed_row = 0
    raw_buffer = None
    max_buffer_rows = max(args.win_len * args.buffer_mult, args.win_len)
    last_size = 0
    last_mtime = None
    idle_cycles = 0
    device_col = None
    time_col = None

    while True:
        try:
            latest_csv = get_latest_data_csv_once(record_root)
            if latest_csv is not None and latest_csv != csv_path:
                csv_path = latest_csv
                print("Switched to new CSV:", csv_path)
                last_processed_row = 0
                raw_buffer = None

            if not csv_path.exists():
                csv_path = find_latest_data_csv(record_root)
                last_processed_row = 0
                raw_buffer = None
                last_size = 0
                last_mtime = None
                idle_cycles = 0

            stat = csv_path.stat()
            if last_mtime is None:
                last_mtime = stat.st_mtime
                last_size = stat.st_size

            if stat.st_size < last_size:
                # File was truncated or rewritten.
                last_processed_row = 0
                raw_buffer = None
                idle_cycles = 0

            new_rows, last_processed_row = read_new_rows(csv_path, last_processed_row)
            if new_rows.empty:
                if stat.st_mtime != last_mtime:
                    idle_cycles += 1
                    if idle_cycles >= 3:
                        # File changed but no new rows parsed; reset and reread.
                        last_processed_row = 0
                        raw_buffer = None
                        idle_cycles = 0
                time.sleep(args.poll_sleep)
                last_size = stat.st_size
                last_mtime = stat.st_mtime
                continue

            print("New rows received:", len(new_rows))
            idle_cycles = 0
            last_size = stat.st_size
            last_mtime = stat.st_mtime

            if raw_buffer is None:
                raw_buffer = new_rows
            else:
                raw_buffer = pd.concat([raw_buffer, new_rows], ignore_index=True)

            if len(raw_buffer) > max_buffer_rows:
                raw_buffer = raw_buffer.iloc[-max_buffer_rows:].reset_index(drop=True)

            if device_col is None:
                device_col = infer_device_column(raw_buffer)
            if time_col is None:
                time_col = infer_time_column(raw_buffer)
            if device_col is not None and time_col is None and is_likely_timestamp(raw_buffer[device_col]):
                time_col = device_col
                device_col = None

            if device_col is None:
                device_col = find_device_column(raw_buffer)
            if device_col is None:
                print("Device ID column not found. Update DEVICE_COL_CANDIDATES if needed.")
                print("Available columns:", list(raw_buffer.columns))
                time.sleep(args.poll_sleep)
                continue

            if time_col is None:
                print("Time column not found; continuing without time-based sorting.")

            wide_df = long_to_wide_latest(raw_buffer, device_col, time_col)
            if wide_df.empty:
                time.sleep(args.poll_sleep)
                continue

            if len(wide_df) < args.win_len:
                time.sleep(args.poll_sleep)
                continue

            # Only preprocess the most recent rows to keep latency low.
            wide_recent = wide_df.tail(args.win_len + 20)
            x_full = preprocess_dataframe(wide_recent)
            x_full = select_acc_euler(x_full)
            latest_window = x_full[-args.win_len :]
            x_windows = latest_window[None, :, :]
            x_windows = apply_sensor_norm(x_windows, means, stds)

            probs = model.predict(x_windows, verbose=0)
            pred_label = int(np.argmax(probs[0]))

            print("Predicted label id:", pred_label)
            print("Predicted posture:", postures[pred_label])

        except (PermissionError, EmptyDataError, ParserError, FileNotFoundError, OSError):
            time.sleep(args.poll_sleep)


if __name__ == "__main__":
    main()
