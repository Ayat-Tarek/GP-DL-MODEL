import argparse
import asyncio
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bleak import BleakClient, BleakScanner

from load_and_predict_realtime import (
    EXPECTED_SENSORS,
    SENSOR_ID_MAP,
    build_model,
    load_norms,
    long_to_wide_latest,
    apply_sensor_norm,
    preprocess_dataframe,
    select_acc_euler,
    QUAT_FEATURES,
)

RAW_COLUMNS = [
    "Time",
    "Device name",
    "Acceleration X(g)",
    "Acceleration Y(g)",
    "Acceleration Z(g)",
    "Angular velocity X(°/s)",
    "Angular velocity Y(°/s)",
    "Angular velocity Z(°/s)",
    "Angle X(°)",
    "Angle Y(°)",
    "Angle Z(°)",
    "Magnetic field X(uT)",
    "Magnetic field Y(uT)",
    "Magnetic field Z(uT)",
    "Quaternions 0()",
    "Quaternions 1()",
    "Quaternions 2()",
    "Quaternions 3()",
]

PRINT_EVERY = 1
FRAME_COUNTER = 0
QUAT_DEBUG_LAST = 0.0
QUAT_DEBUG_INTERVAL = 3.0


def split_frames(payload: bytes) -> List[bytes]:
    frames = []
    i = 0
    while i <= len(payload) - 20:
        if payload[i] != 0x55:
            i += 1
            continue
        frames.append(payload[i : i + 20])
        i += 20
    return frames


def decode_frame_0x61(frame: bytes) -> Optional[dict]:
    if len(frame) < 20 or frame[0] != 0x55 or frame[1] != 0x61:
        return None

    vals = np.frombuffer(frame[2:20], dtype="<i2")
    if vals.size < 9:
        return None

    ax, ay, az, gx, gy, gz, roll, pitch, yaw = vals[:9]
    acc_scale = 16.0 / 32768.0
    gyro_scale = 2000.0 / 32768.0
    angle_scale = 180.0 / 32768.0

    return {
        "acc": (ax * acc_scale, ay * acc_scale, az * acc_scale),
        "gyro": (gx * gyro_scale, gy * gyro_scale, gz * gyro_scale),
        "angle": (roll * angle_scale, pitch * angle_scale, yaw * angle_scale),
    }


def decode_frame(frame: bytes) -> Optional[Tuple[str, dict]]:
    if len(frame) < 20 or frame[0] != 0x55:
        return None

    frame_type = frame[1]
    vals = np.frombuffer(frame[2:20], dtype="<i2")
    if vals.size < 3:
        return None

    if frame_type == 0x61:
        decoded = decode_frame_0x61(frame)
        if not decoded:
            return None
        return "combined", decoded

    if frame_type == 0x51:
        ax, ay, az = vals[:3]
        acc_scale = 16.0 / 32768.0
        return "acc", {"acc": (ax * acc_scale, ay * acc_scale, az * acc_scale)}

    if frame_type == 0x52:
        gx, gy, gz = vals[:3]
        gyro_scale = 2000.0 / 32768.0
        return "gyro", {"gyro": (gx * gyro_scale, gy * gyro_scale, gz * gyro_scale)}

    if frame_type == 0x53:
        roll, pitch, yaw = vals[:3]
        angle_scale = 180.0 / 32768.0
        return "angle", {"angle": (roll * angle_scale, pitch * angle_scale, yaw * angle_scale)}

    if frame_type == 0x54:
        mx, my, mz = vals[:3]
        mag_scale = 4912.0 / 32768.0
        return "mag", {"mag": (mx * mag_scale, my * mag_scale, mz * mag_scale)}

    if frame_type == 0x59:
        if vals.size < 4:
            return None
        q0, q1, q2, q3 = vals[:4]
        quat_scale = 1.0 / 32768.0
        q0 *= quat_scale
        q1 *= quat_scale
        q2 *= quat_scale
        q3 *= quat_scale
        quat_csv = (q1, q2, q3, q0)
        return "quat", {"quat_raw": (q0, q1, q2, q3), "quat": quat_csv}

    return None


def build_row(device_address: str, decoded: dict) -> Dict[str, object]:
    timestamp = datetime.now().isoformat(timespec="milliseconds")
    ax, ay, az = decoded["acc"]
    gx, gy, gz = decoded["gyro"]
    roll, pitch, yaw = decoded["angle"]

    mag = decoded.get("mag", (0.0, 0.0, 0.0))
    quat = decoded.get("quat", (0.0, 0.0, 0.0, 1.0))

    return {
        "Time": timestamp,
        "Device name": device_address,
        "Acceleration X(g)": ax,
        "Acceleration Y(g)": ay,
        "Acceleration Z(g)": az,
        "Angular velocity X(°/s)": gx,
        "Angular velocity Y(°/s)": gy,
        "Angular velocity Z(°/s)": gz,
        "Angle X(°)": roll,
        "Angle Y(°)": pitch,
        "Angle Z(°)": yaw,
        "Magnetic field X(uT)": mag[0],
        "Magnetic field Y(uT)": mag[1],
        "Magnetic field Z(uT)": mag[2],
        "Quaternions 0()": quat[0],
        "Quaternions 1()": quat[1],
        "Quaternions 2()": quat[2],
        "Quaternions 3()": quat[3],
    }


def normalize_row_device(row: Dict[str, object], device_address: str) -> Dict[str, object]:
    row["Device name"] = device_address.lower()
    return row


def ensure_missing_columns(wide_df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "Quaternions 0()": 0.0,
        "Quaternions 1()": 0.0,
        "Quaternions 2()": 0.0,
        "Quaternions 3()": 1.0,
        "Magnetic field X(uT)": 0.0,
        "Magnetic field Y(uT)": 0.0,
        "Magnetic field Z(uT)": 0.0,
    }

    for sensor in EXPECTED_SENSORS:
        for base_name, default_val in defaults.items():
            col = f"{sensor}_{base_name}"
            if col not in wide_df.columns:
                wide_df[col] = default_val

        for quat in QUAT_FEATURES:
            col = f"{sensor}_{quat}"
            if col not in wide_df.columns:
                wide_df[col] = 0.0

        w_col = f"{sensor}_Quaternions 3()"
        if w_col in wide_df.columns:
            wide_df[w_col] = wide_df[w_col].fillna(1.0)

    return wide_df


class BleRealtimePredictor:
    def __init__(
        self,
        model,
        postures: List[str],
        means,
        stds,
        win_len: int,
        buffer_mult: int,
        print_every: int,
    ) -> None:
        self.model = model
        self.postures = postures
        self.means = means
        self.stds = stds
        self.win_len = win_len
        self.max_rows = max(win_len * buffer_mult, win_len)
        self.print_every = max(1, print_every)
        self.raw_rows: Deque[Dict[str, object]] = deque(maxlen=self.max_rows)

    def add_row(self, row: Dict[str, object]) -> None:
        self.raw_rows.append(row)

    def predict_if_ready(self) -> None:
        if len(self.raw_rows) < self.win_len:
            return

        df = pd.DataFrame(list(self.raw_rows), columns=RAW_COLUMNS)
        wide_df = long_to_wide_latest(
            df,
            device_col="Device name",
            time_col="Time",
            allow_single_sensor_mirror=True,
        )
        if wide_df.empty or len(wide_df) < self.win_len:
            return

        wide_df = ensure_missing_columns(wide_df)
        wide_recent = wide_df.tail(self.win_len + 20)
        x_full = preprocess_dataframe(wide_recent)
        x_full = select_acc_euler(x_full)
        latest_window = x_full[-self.win_len :]
        x_windows = latest_window[None, :, :]
        x_windows = apply_sensor_norm(x_windows, self.means, self.stds)

        probs = self.model.predict(x_windows, verbose=0)
        pred_idx = int(np.argmax(probs[0]))
        conf = float(probs[0][pred_idx]) * 100.0

        print(f"Predicted posture: {self.postures[pred_idx]}")
        print(f"Confidence: {conf:.1f}%")


async def scan_devices(filter_address: Optional[str] = None) -> None:
    print("Scanning for BLE devices... (5s)")
    devices = await BleakScanner.discover(timeout=5.0)
    if not devices:
        print("No BLE devices found.")
        return

    for dev in devices:
        addr = dev.address
        if filter_address and addr.lower() != filter_address.lower():
            continue
        name = dev.name or "(unknown)"
        rssi = getattr(dev, "rssi", None)
        rssi_text = f" RSSI={rssi}" if rssi is not None else ""
        print(f"{name} - {addr}{rssi_text}")


async def connect_and_predict(address: str, predictor: BleRealtimePredictor) -> None:
    print(f"Connecting to {address}...")
    async with BleakClient(address) as client:
        if not client.is_connected:
            raise RuntimeError(f"Failed to connect to {address}")

        print("Connected. Discovering services...")
        if hasattr(client, "get_services"):
            services = await client.get_services()
        else:
            services = client.services

        notify_chars = [
            char
            for service in services
            for char in service.characteristics
            if "notify" in char.properties
        ]

        if not notify_chars:
            print("No notify characteristics found.")
            return

        device_state: Dict[str, Dict[str, Tuple[float, ...]]] = {}

        def handle_notify(sender, data):
            global FRAME_COUNTER
            global QUAT_DEBUG_LAST
            frames = split_frames(bytes(data))
            if not frames:
                return
            for frame in frames:
                FRAME_COUNTER += 1
                if predictor.print_every > 1 and FRAME_COUNTER % predictor.print_every != 0:
                    continue
                decoded_entry = decode_frame(frame)
                if not decoded_entry:
                    continue
                _, decoded = decoded_entry

                state = device_state.setdefault(address.lower(), {})
                if "acc" in decoded:
                    state["acc"] = decoded["acc"]
                if "gyro" in decoded:
                    state["gyro"] = decoded["gyro"]
                if "angle" in decoded:
                    state["angle"] = decoded["angle"]
                if "mag" in decoded:
                    state["mag"] = decoded["mag"]
                if "quat" in decoded:
                    state["quat"] = decoded["quat"]
                if "quat_raw" in decoded:
                    state["quat_raw"] = decoded["quat_raw"]
                    now = time.time()
                    if now - QUAT_DEBUG_LAST >= QUAT_DEBUG_INTERVAL:
                        q0, q1, q2, q3 = decoded["quat_raw"]
                        qx, qy, qz, qw = decoded["quat"]
                        print(
                            "Quat raw (wxyz): "
                            f"q0={q0:.4f}, q1={q1:.4f}, q2={q2:.4f}, q3={q3:.4f}"
                        )
                        print(
                            "Quat stored (xyzw): "
                            f"qx={qx:.4f}, qy={qy:.4f}, qz={qz:.4f}, qw={qw:.4f}"
                        )
                        QUAT_DEBUG_LAST = now

                if not all(k in state for k in ("acc", "gyro", "angle")):
                    continue

                row = build_row(address, state)
                row = normalize_row_device(row, address)
                predictor.add_row(row)
                predictor.predict_if_ready()

        print("Subscribing to notify characteristics...")
        for char in notify_chars:
            await client.start_notify(char.uuid, handle_notify)
            print(f"  Notifying: {char.uuid}")

        print("Listening for notifications. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1.0)


async def main() -> None:
    parser = argparse.ArgumentParser(description="WT901BLECL BLE predictor (live).")
    parser.add_argument("--scan", action="store_true", help="Scan for BLE devices and exit")
    parser.add_argument("--address", default=None, help="BLE device MAC address to connect")
    parser.add_argument("--weights", required=True, help="Path to model_weights_best.weights.h5")
    parser.add_argument("--postures_root", default="wide_data", help="Root folder with posture subfolders")
    parser.add_argument("--norm_npz", default="norm_stats.npz", help="Path to saved normalization .npz")
    parser.add_argument("--win_len", type=int, default=200)
    parser.add_argument("--buffer_mult", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--quat_order", choices=["xyzw", "wxyz"], default="xyzw")

    args = parser.parse_args()

    if args.scan or not args.address:
        await scan_devices(filter_address=args.address)
        if args.scan:
            return
        if not args.address:
            print("Provide --address to connect.")
            return

    postures_root = args.postures_root
    import os

    postures = sorted([p for p in os.listdir(postures_root) if os.path.isdir(os.path.join(postures_root, p))])
    if not postures:
        raise FileNotFoundError(f"No posture folders found under: {postures_root}")

    means, stds = load_norms(args.norm_npz)

    model = build_model((args.win_len, 24), num_classes=len(postures))
    model.load_weights(args.weights)

    predictor = BleRealtimePredictor(
        model=model,
        postures=postures,
        means=means,
        stds=stds,
        win_len=args.win_len,
        buffer_mult=args.buffer_mult,
        print_every=args.print_every,
    )

    await connect_and_predict(args.address, predictor)


if __name__ == "__main__":
    asyncio.run(main())
