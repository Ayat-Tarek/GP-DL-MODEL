import argparse
import asyncio
import json
from datetime import datetime
from typing import List, Optional

import numpy as np
from bleak import BleakClient, BleakScanner


PRINT_RAW = False
PRINT_EVERY = 1
FRAME_COUNTER = 0
OUT_QUEUE = None
DEBUG_STATS = False
MAG_COUNT = 0
QUAT_COUNT = 0
LAST_MAG = None
LAST_QUAT = None
REQUEST_MAG_QUAT = False
REQUEST_INTERVAL = 0.2
MAG_CMD = bytes.fromhex("FF AA 27 3A 00")
QUAT_CMD = bytes.fromhex("FF AA 27 51 00")


def format_sender(sender) -> str:
    if hasattr(sender, "handle"):
        return f"handle=0x{sender.handle:04X}"
    if hasattr(sender, "uuid"):
        return f"uuid={sender.uuid}"
    if isinstance(sender, int):
        return f"handle=0x{sender:04X}"
    return f"sender={sender}"


def split_frames(payload: bytes) -> List[bytes]:
    frames = []
    i = 0
    while i <= len(payload) - 20:
        if payload[i] != 0x55:
            i += 1
            continue
        frame = payload[i : i + 20]
        frames.append(frame)
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


def decode_frame(frame: bytes) -> Optional[dict]:
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
        decoded["type"] = "0x61"
        return decoded

    if frame_type == 0x71:
        reg_l = frame[2]
        reg_h = frame[3]
        if reg_l == 0x3A and reg_h == 0x00:
            vals = np.frombuffer(frame[4:10], dtype="<i2")
            if vals.size < 3:
                return None
            hx, hy, hz = vals[:3]
            return {"type": "0x71_mag", "mag": (float(hx), float(hy), float(hz))}

        if reg_l == 0x51 and reg_h == 0x00:
            vals = np.frombuffer(frame[4:12], dtype="<i2")
            if vals.size < 4:
                return None
            q0, q1, q2, q3 = vals[:4]
            quat_scale = 1.0 / 32768.0
            return {
                "type": "0x71_quat",
                "quat": (q0 * quat_scale, q1 * quat_scale, q2 * quat_scale, q3 * quat_scale),
            }

        return None

    if frame_type == 0x54:
        mx, my, mz = vals[:3]
        mag_scale = 4912.0 / 32768.0
        return {"type": "0x54", "mag": (mx * mag_scale, my * mag_scale, mz * mag_scale)}

    if frame_type == 0x59:
        if vals.size < 4:
            return None
        q0, q1, q2, q3 = vals[:4]
        quat_scale = 1.0 / 32768.0
        return {
            "type": "0x59",
            "quat": (q0 * quat_scale, q1 * quat_scale, q2 * quat_scale, q3 * quat_scale),
        }

    return None


def build_row(device_address: str, decoded: dict) -> dict:
    timestamp = datetime.now().isoformat(timespec="milliseconds")
    ax, ay, az = decoded["acc"]
    gx, gy, gz = decoded["gyro"]
    roll, pitch, yaw = decoded["angle"]

    mag = decoded.get("mag", (0.0, 0.0, 0.0))
    quat = decoded.get("quat", (0.0, 0.0, 0.0, 1.0))
    quat_real = bool(decoded.get("quat_real", False))

    return {
        "Time": timestamp,
        "Device name": device_address.lower(),
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
        "Quat real": quat_real,
    }


def enqueue_row(row: dict) -> None:
    if OUT_QUEUE is None:
        return
    payload = json.dumps(row)
    try:
        OUT_QUEUE.put_nowait(payload)
    except asyncio.QueueFull:
        pass


def notification_handler(sender, data: bytearray, device_address: str, state: dict) -> None:
    global FRAME_COUNTER, MAG_COUNT, QUAT_COUNT, LAST_MAG, LAST_QUAT

    frames = split_frames(bytes(data))
    if not frames:
        return

    sender_text = format_sender(sender)
    for frame in frames:
        FRAME_COUNTER += 1
        if PRINT_EVERY > 1 and FRAME_COUNTER % PRINT_EVERY != 0:
            continue

        timestamp = datetime.now().isoformat(timespec="milliseconds")
        if PRINT_RAW:
            print(f"[{timestamp}] {device_address} {sender_text} raw={frame.hex(' ')}")

        decoded = decode_frame(frame)
        if not decoded:
            continue

        # if decoded.get("type") == "0x61":
            # ax, ay, az = decoded["acc"]
            # gx, gy, gz = decoded["gyro"]
            # roll, pitch, yaw = decoded["angle"]
            # print(
            #     f"[{timestamp}] {device_address} "
            #     f"Acc=({ax:.3f},{ay:.3f},{az:.3f})g "
            #     f"Gyro=({gx:.2f},{gy:.2f},{gz:.2f})deg/s "
            #     f"Angle=({roll:.2f},{pitch:.2f},{yaw:.2f})deg"
            # )
        if decoded.get("type") == "0x54":
            mx, my, mz = decoded["mag"]
            MAG_COUNT += 1
            LAST_MAG = (mx, my, mz)
            print(
                f"[{timestamp}] {device_address} "
                f"Mag=({mx:.1f},{my:.1f},{mz:.1f})uT"
            )
        elif decoded.get("type") == "0x59":
            q0, q1, q2, q3 = decoded["quat"]
            QUAT_COUNT += 1
            LAST_QUAT = (q0, q1, q2, q3)
            print(
                f"[{timestamp}] {device_address} "
                f"Quat=({q0:.4f},{q1:.4f},{q2:.4f},{q3:.4f})"
            )
        elif decoded.get("type") == "0x71_mag":
            hx, hy, hz = decoded["mag"]
            MAG_COUNT += 1
            LAST_MAG = (hx, hy, hz)
            print(f"MAG: hx={hx:.0f}, hy={hy:.0f}, hz={hz:.0f}")
        elif decoded.get("type") == "0x71_quat":
            q0, q1, q2, q3 = decoded["quat"]
            QUAT_COUNT += 1
            LAST_QUAT = (q0, q1, q2, q3)
            print(f"QUAT: q0={q0:.4f}, q1={q1:.4f}, q2={q2:.4f}, q3={q3:.4f}")

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
            state["quat_real"] = True

        if all(k in state for k in ("acc", "gyro", "angle")):
            if "quat_real" not in state:
                state["quat_real"] = False
            row = build_row(device_address, state)
            enqueue_row(row)

        if DEBUG_STATS and FRAME_COUNTER % 200 == 0:
            print(
                f"[debug] mag_frames={MAG_COUNT} quat_frames={QUAT_COUNT} "
                f"last_mag={LAST_MAG} last_quat={LAST_QUAT}"
            )


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


async def connect_and_list_services(address: str, out_host: str, out_port: int) -> None:
    print(f"Connecting to {address}...")
    async with BleakClient(address) as client:
        if not client.is_connected:
            raise RuntimeError(f"Failed to connect to {address}")

        print("Connected. Discovering services...")
        if hasattr(client, "get_services"):
            services = await client.get_services()
        else:
            services = client.services
        for service in services:
            print(f"Service {service.uuid}: {service.description}")
            for char in service.characteristics:
                props = ",".join(char.properties)
                print(f"  Char {char.uuid} ({props})")

        notify_chars = [
            char
            for service in services
            for char in service.characteristics
            if "notify" in char.properties
        ]

        write_chars = [
            char
            for service in services
            for char in service.characteristics
            if "write" in char.properties or "write-without-response" in char.properties
        ]
        write_char = next(
            (c for c in write_chars if "0000ffe9" in c.uuid.lower()),
            None,
        )

        if not notify_chars:
            print("No notify characteristics found.")
            return

        device_state = {}

        print("Subscribing to notify characteristics...")
        for char in notify_chars:
            await client.start_notify(
                char.uuid,
                lambda sender, data, addr=address: notification_handler(sender, data, addr, device_state),
            )
            print(f"  Notifying: {char.uuid}")

        async def request_loop():
            if not REQUEST_MAG_QUAT:
                return
            if write_char is None:
                print("Write characteristic ffe9 not found; cannot request mag/quat.")
                return
            while True:
                try:
                    await client.write_gatt_char(write_char.uuid, MAG_CMD, response=False)
                    await client.write_gatt_char(write_char.uuid, QUAT_CMD, response=False)
                except Exception:
                    pass
                await asyncio.sleep(REQUEST_INTERVAL)

        asyncio.create_task(request_loop())

        print("Listening for notifications. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1.0)


async def main() -> None:
    parser = argparse.ArgumentParser(description="WT901BLECL BLE receiver (raw packets).")
    parser.add_argument("--scan", action="store_true", help="Scan for BLE devices and exit")
    parser.add_argument("--address", default=None, help="Single BLE device MAC address to connect")
    parser.add_argument(
        "--addresses",
        nargs="*",
        default=None,
        help="Multiple BLE device MAC addresses to connect",
    )
    parser.add_argument("--raw", action="store_true", help="Print raw notification frames")
    parser.add_argument("--print_every", type=int, default=1, help="Print every Nth frame")
    parser.add_argument("--out_host", default="127.0.0.1", help="Prediction ingest host")
    parser.add_argument("--out_port", type=int, default=9001, help="Prediction ingest port")
    parser.add_argument("--debug_stats", action="store_true", help="Print mag/quat counters")
    parser.add_argument("--request_mag_quat", action="store_true", help="Request mag/quat via ffe9")
    parser.add_argument("--request_interval", type=float, default=0.2, help="Seconds between requests")

    args = parser.parse_args()

    global PRINT_RAW
    global PRINT_EVERY
    PRINT_RAW = args.raw
    PRINT_EVERY = max(1, args.print_every)
    global DEBUG_STATS
    DEBUG_STATS = args.debug_stats
    global REQUEST_MAG_QUAT
    global REQUEST_INTERVAL
    REQUEST_MAG_QUAT = args.request_mag_quat
    REQUEST_INTERVAL = args.request_interval

    global OUT_QUEUE
    OUT_QUEUE = asyncio.Queue(maxsize=2000)

    async def send_loop():
        while True:
            try:
                reader, writer = await asyncio.open_connection(args.out_host, args.out_port)
                print(f"Connected to processor at {args.out_host}:{args.out_port}")
                while True:
                    payload = await OUT_QUEUE.get()
                    writer.write(payload.encode("utf-8") + b"\n")
                    await writer.drain()
            except Exception:
                await asyncio.sleep(1.0)

    asyncio.create_task(send_loop())

    address_list = []
    if args.addresses:
        address_list = [addr for addr in args.addresses if addr]
    elif args.address:
        address_list = [args.address]

    if args.scan or not address_list:
        await scan_devices(filter_address=args.address)
        if args.scan:
            return

        if not address_list:
            print("Provide --address to connect.")
            return

    await asyncio.gather(
        *[
            connect_and_list_services(address, args.out_host, args.out_port)
            for address in address_list
        ]
    )


if __name__ == "__main__":
    asyncio.run(main())
