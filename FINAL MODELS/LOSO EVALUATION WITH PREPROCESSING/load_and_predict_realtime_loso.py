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

SENSOR_ID_MAP = {
    "ed:35:33:d3:6c:f8": "C7",
    "ed:40:fe:65:30:6c": "T4",
    "f6:90:cc:01:6d:25": "T12",
    "e3:ca:2d:fd:e0:8c": "L5",
}

EXPECTED_SENSORS = ["C7", "T4", "T12", "L5"]


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


def normalize_quat(q0, q1, q2, q3, mode="reorder_wxyz"):
    if mode == "reorder_wxyz":
        # Notebook training used quat_reorder_to_wxyz(qx,qy,qz,qw).
        q = np.array([q3, q0, q1, q2], dtype=np.float32)
    else:
        # no_reorder: direct q0,q1,q2,q3
        q = np.array([q0, q1, q2, q3], dtype=np.float32)

    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    return q / norm


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


def preprocess_dataframe(df, quat_mode="reorder_wxyz"):
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
        for (q0, q1, q2, q3) in quat_raw:
            q = normalize_quat(q0, q1, q2, q3, mode=quat_mode)
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


def load_norms(norm_npz):
    data = np.load(norm_npz, allow_pickle=True)
    means = [data[f"mean_{i}"] for i in range(4)]
    stds = [data[f"std_{i}"] for i in range(4)]
    return means, stds


class SensorDropout(layers.Layer):
    def __init__(self, drop_prob=0.25):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training:
            return x

        batch = tf.shape(x)[0]
        n_sensors = 4
        feat_per_sensor = tf.shape(x)[-1] // n_sensors

        mask = tf.cast(
            tf.random.uniform((batch, n_sensors, 1)) > self.drop_prob,
            x.dtype,
        )

        mask = tf.repeat(mask, repeats=feat_per_sensor, axis=2)
        mask = tf.reshape(mask, (batch, 1, n_sensors * feat_per_sensor))

        return x * mask


def build_model(input_shape, num_classes):
    inp = keras.Input(shape=input_shape)
    x = SensorDropout(0.25)(inp)

    l5 = layers.Lambda(lambda x: x[:, :, 0:6])(x)
    t4 = layers.Lambda(lambda x: x[:, :, 6:12])(x)
    c7 = layers.Lambda(lambda x: x[:, :, 12:18])(x)
    t12 = layers.Lambda(lambda x: x[:, :, 18:24])(x)

    def sensor_block(x, name_prefix):
        x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        attention = layers.Dense(32, activation="tanh", name=f"{name_prefix}_feat_dense")(x)
        attention = layers.Softmax(axis=-1, name=f"{name_prefix}_feat_softmax")(attention)
        x = layers.Multiply(name=f"{name_prefix}_feat_mul")([x, attention])

        return x, attention

    l5, l5_feat_att = sensor_block(l5, "L5")
    t4, t4_feat_att = sensor_block(t4, "T4")
    c7, c7_feat_att = sensor_block(c7, "C7")
    t12, t12_feat_att = sensor_block(t12, "T12")

    sensors = layers.Lambda(lambda x: tf.stack(x, axis=2))([l5, t4, c7, t12])
    sensor_attention = layers.Dense(1, activation="tanh", name="sensor_dense")(sensors)
    sensor_attention = layers.Softmax(axis=2, name="sensor_softmax")(sensor_attention)

    sensors = layers.Multiply()([sensors, sensor_attention])
    fused = layers.Lambda(lambda x: tf.reduce_sum(x, axis=2))(sensors)

    x = layers.GlobalAveragePooling1D()(fused)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.feature_attentions = [l5_feat_att, t4_feat_att, c7_feat_att, t12_feat_att]
    model.sensor_attention = sensor_attention

    return model


def feature_order_per_sensor():
    return [
        "Acceleration X(g)",
        "Acceleration Y(g)",
        "Acceleration Z(g)",
        "Euler yaw",
        "Euler pitch",
        "Euler roll",
    ]


def main():
    parser = argparse.ArgumentParser(description="LOSO load helper.")
    parser.add_argument("--norm_npz", default="loso_norm_stats.npz")
    parser.add_argument("--win_len", type=int, default=200)
    args = parser.parse_args()

    _ = load_norms(Path(args.norm_npz))


if __name__ == "__main__":
    main()
