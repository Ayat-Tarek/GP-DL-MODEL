import argparse

import numpy as np
import pandas as pd

from load_and_predict_realtime_loso import (
    ACC_FEATURES,
    GYRO_FEATURES,
    QUAT_FEATURES,
    SENSORS,
    detect_mag_cols,
    hampel_filter,
    butter_lowpass_filter,
    remove_bias,
    rotate_vector_by_quaternion,
    quaternion_to_euler,
    preprocess_dataframe,
    select_acc_euler,
)


def quat_reorder_to_wxyz(qx, qy, qz, qw):
    return np.array([qw, qx, qy, qz], dtype=np.float32)


def preprocess_notebook_style(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LOSO preprocessing against realtime pipeline.")
    parser.add_argument("--csv", required=True, help="Path to a LOSO wide CSV file")
    args = parser.parse_args()

    x_notebook = preprocess_notebook_style(args.csv)
    x_realtime = preprocess_dataframe(pd.read_csv(args.csv), quat_mode="reorder_wxyz")

    diff_full = np.max(np.abs(x_notebook - x_realtime))
    print("Full preprocess shape:", x_notebook.shape)
    print("Max abs diff (full):", diff_full)

    x_notebook_sel = select_acc_euler(x_notebook)
    x_realtime_sel = select_acc_euler(x_realtime)
    diff_sel = np.max(np.abs(x_notebook_sel - x_realtime_sel))
    print("ACC+Euler shape:", x_notebook_sel.shape)
    print("Max abs diff (acc+euler):", diff_sel)


if __name__ == "__main__":
    main()
