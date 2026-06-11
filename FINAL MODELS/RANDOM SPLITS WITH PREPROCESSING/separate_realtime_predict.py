import argparse
import asyncio
import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, List

import numpy as np
import pandas as pd

from load_and_predict_realtime import (
    EXPECTED_SENSORS,
    SENSOR_ID_MAP,
    QUAT_FEATURES,
    apply_sensor_norm,
    build_model,
    load_norms,
    long_to_wide_latest,
    preprocess_dataframe,
    select_acc_euler,
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


class PredictionBroadcaster:
    def __init__(self) -> None:
        self.clients: List[asyncio.StreamWriter] = []

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.clients.append(writer)
        try:
            while await reader.readline():
                pass
        finally:
            if writer in self.clients:
                self.clients.remove(writer)
            writer.close()
            await writer.wait_closed()

    async def broadcast(self, payload: dict) -> None:
        if not self.clients:
            return
        data = (json.dumps(payload) + "\n").encode("utf-8")
        dead = []
        for writer in self.clients:
            try:
                writer.write(data)
                await writer.drain()
            except Exception:
                dead.append(writer)
        for writer in dead:
            if writer in self.clients:
                self.clients.remove(writer)


class RealtimeProcessor:
    def __init__(
        self,
        model,
        postures: List[str],
        means,
        stds,
        win_len: int,
        buffer_mult: int,
        quat_order: str,
        require_all_sensors: bool,
        recent_seconds: float,
        sync_max_ms: float,
        require_real_quat: bool,
        synced_window_seconds: float,
        debug_interval: float,
        broadcaster: PredictionBroadcaster,
    ) -> None:
        self.model = model
        self.postures = postures
        self.means = means
        self.stds = stds
        self.win_len = win_len
        self.max_rows = max(win_len * buffer_mult, win_len)
        self.raw_rows: Deque[Dict[str, object]] = deque(maxlen=self.max_rows)
        self.device_buffers: Dict[str, Deque[Dict[str, object]]] = defaultdict(
            lambda: deque(maxlen=self.max_rows)
        )
        self.synced_rows: Deque[Dict[str, object]] = deque(maxlen=self.max_rows)
        self.synced_meta: Deque[Dict[str, float]] = deque(maxlen=self.max_rows)
        self.broadcaster = broadcaster
        self.last_quat_debug = 0.0
        self.quat_order = quat_order
        self.require_all_sensors = require_all_sensors
        self.recent_seconds = recent_seconds
        self.sync_max_ms = sync_max_ms
        self.require_real_quat = require_real_quat
        self.synced_window_seconds = synced_window_seconds
        self.debug_interval = debug_interval
        self.last_debug = 0.0
        self.last_seen: Dict[str, float] = {}
        self.last_spread_print = 0.0
        self.spread_print_every = 25
        self.synced_count = 0
        self.last_predict_time = 0.0
        self.last_predict_synced = 0
        self.last_fresh_print = 0.0
        self.sensors_with_real_quat: set[str] = set()

    async def handle_row(self, row: Dict[str, object]) -> None:
        now = time.monotonic()
        if "Device name" in row:
            row["Device name"] = str(row["Device name"]).lower()
            self.last_seen[row["Device name"]] = now

        quat_real = bool(row.get("Quat real", False))

        q0_key = "Quaternions 0()"
        q1_key = "Quaternions 1()"
        q2_key = "Quaternions 2()"
        q3_key = "Quaternions 3()"
        if all(key in row for key in (q0_key, q1_key, q2_key, q3_key)):
            try:
                q0 = float(row[q0_key])
                q1 = float(row[q1_key])
                q2 = float(row[q2_key])
                q3 = float(row[q3_key])
            except (TypeError, ValueError):
                q0 = q1 = q2 = q3 = None

            if q0 is not None:
                is_placeholder = q0 == 0.0 and q1 == 0.0 and q2 == 0.0 and q3 == 1.0

                if self.require_real_quat and not quat_real:
                    return

                if self.quat_order == "raw_wxyz" and not (is_placeholder and not quat_real):
                    # Raw register order is w,x,y,z -> store as x,y,z,w.
                    row[q0_key] = q1
                    row[q1_key] = q2
                    row[q2_key] = q3
                    row[q3_key] = q0

                now = time.monotonic()
                if now - self.last_quat_debug >= 3.0:
                    print(
                        "Raw quat wxyz: "
                        f"q0={q0:.6f}, q1={q1:.6f}, q2={q2:.6f}, q3={q3:.6f}"
                    )
                    print(
                        "Stored quat xyzw: "
                        f"qx={row[q0_key]:.6f}, qy={row[q1_key]:.6f}, "
                        f"qz={row[q2_key]:.6f}, qw={row[q3_key]:.6f}"
                    )
                    self.last_quat_debug = now

                if quat_real and "Device name" in row:
                    mapped_sensor = SENSOR_ID_MAP.get(row["Device name"], row["Device name"])
                    self.sensors_with_real_quat.add(mapped_sensor)

        self.raw_rows.append(row)

        device_name = row.get("Device name")
        if isinstance(device_name, str):
            mapped_sensor = SENSOR_ID_MAP.get(device_name, device_name)
            if not (self.require_real_quat and not quat_real):
                self.device_buffers[mapped_sensor].append(
                    {"_t": now, "_device": device_name, **row}
                )

        if now - self.last_debug >= self.debug_interval:
            unique_devices = sorted(
                {
                    r.get("Device name")
                    for r in self.raw_rows
                    if isinstance(r.get("Device name"), str)
                }
            )
            mapped_sensors = sorted(
                {
                    SENSOR_ID_MAP.get(d, d)
                    for d in unique_devices
                    if d is not None
                }
            )
            device_counts = {
                sensor: len(buf) for sensor, buf in self.device_buffers.items()
            }
            synced_count = len(self.synced_rows)
            recent_meta = list(self.synced_meta)[-self.win_len :]
            used_counts = {
                sensor: sum(1 for meta in recent_meta if sensor in meta)
                for sensor in EXPECTED_SENSORS
            }
            print("Unique device IDs received:", unique_devices)
            print("Mapped anatomical sensors:", mapped_sensors)
            print("Rows per device:", device_counts)
            print("Synced rows:", synced_count)
            print("Samples used in last window:", used_counts)
            if device_counts:
                min_rows = min(device_counts.values())
                max_rows = max(device_counts.values())
                if min_rows == 0 or max_rows > (min_rows * 2):
                    print("Warning: sensor row counts are imbalanced.")
            self.last_debug = now

        if self.require_all_sensors:
            mapped_last_seen: Dict[str, float] = {}
            for device_id, ts in self.last_seen.items():
                mapped = SENSOR_ID_MAP.get(device_id, device_id)
                mapped_last_seen[mapped] = ts

            if not all(s in mapped_last_seen for s in EXPECTED_SENSORS):
                return

            if not all((now - mapped_last_seen[s]) <= self.recent_seconds for s in EXPECTED_SENSORS):
                return

        if any(
            (now - self.last_seen.get(device_id, 0.0)) > self.recent_seconds
            for device_id in SENSOR_ID_MAP
        ):
            return

        if self.require_real_quat and not all(s in self.sensors_with_real_quat for s in EXPECTED_SENSORS):
            return

        synced = self._try_sync_row()
        if synced is not None:
            wide_row, meta, spread_ms = synced
            sync_time = time.monotonic()
            wide_row["__sync_time"] = sync_time
            meta["__sync_time"] = sync_time
            self.synced_rows.append(wide_row)
            self.synced_meta.append(meta)
            self.synced_count += 1
            if self.synced_count % self.spread_print_every == 0 or now - self.last_spread_print >= 1.0:
                print(f"Sync spread: {spread_ms:.1f} ms")
                self.last_spread_print = now

        fresh_rows, fresh_meta = self._get_fresh_synced(now)
        fresh_count = len(fresh_rows)
        if fresh_count < self.win_len:
            if now - self.last_fresh_print >= 1.0:
                print(f"Fresh synced rows: {fresh_count} / {self.win_len}")
                self.last_fresh_print = now
            return

        recent_meta = fresh_meta[-self.win_len :]
        if any(sum(1 for meta in recent_meta if sensor in meta) < self.win_len for sensor in EXPECTED_SENSORS):
            return

        if not self._should_predict(now):
            return

        wide_df = pd.DataFrame(fresh_rows[-self.win_len :])
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
        posture = self.postures[pred_idx]

        timestamp = datetime.now().isoformat(timespec="milliseconds")
        payload = {
            "type": "prediction",
            "Time": timestamp,
            "Predicted posture": posture,
            "Confidence": conf,
            "Source": "Random Splits",
        }
        await self.broadcaster.broadcast(payload)
        print(f"Predicted posture: {posture}")
        print(f"Confidence: {conf:.1f}%")
        self.last_predict_time = now
        self.last_predict_synced = self.synced_count

    def _try_sync_row(self) -> tuple[Dict[str, object], Dict[str, float], float] | None:
        if not all(s in self.device_buffers and self.device_buffers[s] for s in EXPECTED_SENSORS):
            return None

        while True:
            latest_samples = {s: self.device_buffers[s][0] for s in EXPECTED_SENSORS}
            times = {s: latest_samples[s]["_t"] for s in EXPECTED_SENSORS}
            min_sensor = min(times, key=times.get)
            max_sensor = max(times, key=times.get)
            spread_ms = (times[max_sensor] - times[min_sensor]) * 1000.0

            if spread_ms <= self.sync_max_ms:
                wide_row = self._build_wide_row(latest_samples)
                for sensor in EXPECTED_SENSORS:
                    self.device_buffers[sensor].popleft()
                return wide_row, times, spread_ms

            # Drop oldest sample and try again.
            self.device_buffers[min_sensor].popleft()
            if not all(self.device_buffers[s] for s in EXPECTED_SENSORS):
                return None

    def _build_wide_row(self, samples: Dict[str, Dict[str, object]]) -> Dict[str, object]:
        wide_row: Dict[str, object] = {}
        for sensor, sample in samples.items():
            for key, value in sample.items():
                if key in ("Time", "Device name", "_t", "_device"):
                    continue
                if key not in RAW_COLUMNS:
                    continue
                wide_row[f"{sensor}_{key}"] = value
        return wide_row

    def _get_fresh_synced(self, now: float) -> tuple[List[Dict[str, object]], List[Dict[str, float]]]:
        fresh_rows = []
        fresh_meta = []
        cutoff = now - self.synced_window_seconds
        for row, meta in zip(self.synced_rows, self.synced_meta):
            if meta.get("__sync_time", 0.0) >= cutoff:
                fresh_rows.append(row)
                fresh_meta.append(meta)
        return fresh_rows, fresh_meta

    def _should_predict(self, now: float) -> bool:
        if self.synced_count - self.last_predict_synced >= 25:
            return True
        if now - self.last_predict_time >= 0.5:
            return True
        return False


async def handle_ingest(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    processor: RealtimeProcessor,
) -> None:
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                row = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            await processor.handle_row(row)
    finally:
        writer.close()
        await writer.wait_closed()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Process BLE IMU stream and predict postures.")
    parser.add_argument("--in_host", default="127.0.0.1")
    parser.add_argument("--in_port", type=int, default=9001)
    parser.add_argument("--out_host", default="127.0.0.1")
    parser.add_argument("--out_port", type=int, default=9002)
    parser.add_argument("--weights", required=True, help="Path to model_weights_best.weights.h5")
    parser.add_argument("--postures_root", default="wide_data")
    parser.add_argument("--norm_npz", default="norm_stats.npz")
    parser.add_argument("--win_len", type=int, default=200)
    parser.add_argument("--buffer_mult", type=int, default=2)
    parser.add_argument(
        "--require_all_sensors",
        action="store_true",
        help="Only predict when all four sensors have recent data.",
    )
    parser.add_argument(
        "--recent_seconds",
        type=float,
        default=2.0,
        help="Max age in seconds for each sensor when --require_all_sensors is set.",
    )
    parser.add_argument(
        "--sync_max_ms",
        type=float,
        default=80.0,
        help="Max time difference in ms between sensor samples for a synced row.",
    )
    parser.add_argument(
        "--require_real_quat",
        action="store_true",
        help="Only build synced rows after real quaternion values are received.",
    )
    parser.add_argument(
        "--synced_window_seconds",
        type=float,
        default=5.0,
        help="Seconds of synced rows to keep for prediction freshness.",
    )
    parser.add_argument(
        "--debug_interval",
        type=float,
        default=3.0,
        help="Seconds between debug summaries of device IDs and buffer counts.",
    )
    parser.add_argument(
        "--quat_order",
        choices=["raw_wxyz", "csv_xyzw"],
        default="csv_xyzw",
        help="Input quaternion order: raw_wxyz from sensor or csv_xyzw (no reorder)",
    )

    args = parser.parse_args()

    postures = sorted(
        [p for p in os.listdir(args.postures_root) if os.path.isdir(os.path.join(args.postures_root, p))]
    )
    if not postures:
        raise FileNotFoundError(f"No posture folders found under: {args.postures_root}")

    means, stds = load_norms(args.norm_npz)
    model = build_model((args.win_len, 24), num_classes=len(postures))
    model.load_weights(args.weights)

    broadcaster = PredictionBroadcaster()
    processor = RealtimeProcessor(
        model=model,
        postures=postures,
        means=means,
        stds=stds,
        win_len=args.win_len,
        buffer_mult=args.buffer_mult,
        quat_order=args.quat_order,
        require_all_sensors=args.require_all_sensors,
        recent_seconds=args.recent_seconds,
        sync_max_ms=args.sync_max_ms,
        require_real_quat=args.require_real_quat,
        synced_window_seconds=args.synced_window_seconds,
        debug_interval=args.debug_interval,
        broadcaster=broadcaster,
    )

    ingest_server = await asyncio.start_server(
        lambda r, w: handle_ingest(r, w, processor),
        args.in_host,
        args.in_port,
    )
    pred_server = await asyncio.start_server(
        broadcaster.handle_client,
        args.out_host,
        args.out_port,
    )

    print(f"Listening for raw IMU on {args.in_host}:{args.in_port}")
    print(f"Publishing predictions on {args.out_host}:{args.out_port}")

    async with ingest_server, pred_server:
        await asyncio.gather(ingest_server.serve_forever(), pred_server.serve_forever())


if __name__ == "__main__":
    import os

    asyncio.run(main())
