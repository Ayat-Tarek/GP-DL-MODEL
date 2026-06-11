import argparse
import json
import queue
import socketserver
import threading
import tkinter as tk
from datetime import datetime
from typing import Dict, Optional

try:
    from load_and_predict_realtime import EXPECTED_SENSORS, SENSOR_ID_MAP
except Exception:
    EXPECTED_SENSORS = ["C7", "T4", "T12", "L5"]
    SENSOR_ID_MAP = {}


DEFAULT_KEYS = {
    "Acceleration X(g)": "ax",
    "Acceleration Y(g)": "ay",
    "Acceleration Z(g)": "az",
    "Angular velocity X(°/s)": "gx",
    "Angular velocity Y(°/s)": "gy",
    "Angular velocity Z(°/s)": "gz",
    "Angle X(°)": "roll",
    "Angle Y(°)": "pitch",
    "Angle Z(°)": "yaw",
    "Magnetic field X(uT)": "mx",
    "Magnetic field Y(uT)": "my",
    "Magnetic field Z(uT)": "mz",
    "Quaternions 0()": "q0",
    "Quaternions 1()": "q1",
    "Quaternions 2()": "q2",
    "Quaternions 3()": "q3",
}


def normalize_device_name(name: str) -> str:
    return str(name).strip().lower()


def resolve_sensor_label(device_name: str) -> str:
    addr = normalize_device_name(device_name)
    if SENSOR_ID_MAP and addr in SENSOR_ID_MAP:
        return SENSOR_ID_MAP[addr]
    return addr


class ImuState:
    def __init__(self) -> None:
        self.latest: Dict[str, Dict[str, float]] = {}
        self.last_seen: Dict[str, str] = {}

    def update(self, device_name: str, row: dict) -> None:
        label = resolve_sensor_label(device_name)
        values = {}
        for key, short in DEFAULT_KEYS.items():
            if key in row:
                try:
                    values[short] = float(row[key])
                except (TypeError, ValueError):
                    values[short] = float("nan")
        self.latest[label] = values
        self.last_seen[label] = datetime.now().strftime("%H:%M:%S")


class UiApp:
    def __init__(self, root: tk.Tk, state: ImuState, update_queue: queue.Queue) -> None:
        self.root = root
        self.state = state
        self.update_queue = update_queue
        self.frames: Dict[str, tk.Label] = {}
        self.value_labels: Dict[str, tk.Label] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        self.root.title("Live IMU - 4 Sensors")
        self.root.geometry("900x600")
        self.root.configure(bg="#101316")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        sensors = EXPECTED_SENSORS if EXPECTED_SENSORS else ["S1", "S2", "S3", "S4"]
        grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for sensor, (r, c) in zip(sensors, grid_positions):
            frame = tk.Frame(self.root, bg="#1b2228", bd=2, relief="ridge")
            frame.grid(row=r, column=c, sticky="nsew", padx=10, pady=10)
            header = tk.Label(
                frame,
                text=f"{sensor}",
                font=("Segoe UI", 18, "bold"),
                fg="#f2f4f7",
                bg="#1b2228",
            )
            header.pack(anchor="w", padx=12, pady=(10, 4))

            value_label = tk.Label(
                frame,
                text="Waiting for data...",
                font=("Consolas", 12),
                fg="#b7c0c8",
                bg="#1b2228",
                justify="left",
            )
            value_label.pack(anchor="w", padx=12, pady=(0, 12))

            self.value_labels[sensor] = value_label

    def start(self) -> None:
        self._poll_updates()
        self.root.mainloop()

    def _poll_updates(self) -> None:
        while True:
            try:
                device_name, row = self.update_queue.get_nowait()
            except queue.Empty:
                break
            self.state.update(device_name, row)

        for sensor in self.value_labels:
            values = self.state.latest.get(sensor)
            last_seen = self.state.last_seen.get(sensor, "--:--:--")
            if not values:
                self.value_labels[sensor].configure(text="Waiting for data...")
                continue

            text = (
                f"Last: {last_seen}\n"
                f"Acc  : {values.get('ax', 0.0):>6.3f}, {values.get('ay', 0.0):>6.3f}, {values.get('az', 0.0):>6.3f}\n"
                f"Gyro : {values.get('gx', 0.0):>6.2f}, {values.get('gy', 0.0):>6.2f}, {values.get('gz', 0.0):>6.2f}\n"
                f"Angle: {values.get('roll', 0.0):>6.2f}, {values.get('pitch', 0.0):>6.2f}, {values.get('yaw', 0.0):>6.2f}\n"
                f"Mag  : {values.get('mx', 0.0):>6.1f}, {values.get('my', 0.0):>6.1f}, {values.get('mz', 0.0):>6.1f}\n"
                f"Quat : {values.get('q0', 0.0):>6.4f}, {values.get('q1', 0.0):>6.4f}, {values.get('q2', 0.0):>6.4f}, {values.get('q3', 0.0):>6.4f}"
            )
            self.value_labels[sensor].configure(text=text)

        self.root.after(100, self._poll_updates)


class JsonLineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                break
            try:
                row = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            device_name = row.get("Device name") or row.get("Device") or "unknown"
            self.server.update_queue.put((device_name, row))


class JsonLineServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, update_queue):
        super().__init__(server_address, handler_class)
        self.update_queue = update_queue


def start_server(host: str, port: int, update_queue: queue.Queue) -> socketserver.ThreadingTCPServer:
    server = JsonLineServer((host, port), JsonLineHandler, update_queue)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Live IMU quadrant UI (TCP JSON line ingest).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    args = parser.parse_args()

    update_queue = queue.Queue(maxsize=2000)
    state = ImuState()

    start_server(args.host, args.port, update_queue)
    print(f"UI listening on {args.host}:{args.port} for JSON lines")

    root = tk.Tk()
    app = UiApp(root, state, update_queue)
    app.start()


if __name__ == "__main__":
    main()
