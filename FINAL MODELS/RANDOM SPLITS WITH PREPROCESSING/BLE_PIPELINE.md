# BLE realtime pipeline

This folder has a two-process realtime pipeline:

- ble_realtime_receiver.py: Connects to the WT901BLECL over BLE, decodes frames, and streams raw IMU rows as JSON lines over TCP.
- separate_realtime_predict.py: Receives those rows, preprocesses, runs the model, and publishes prediction JSON lines over TCP.

## Data flow

WT901BLECL (BLE notifications) -> ble_realtime_receiver.py -> TCP JSON lines (raw IMU) -> separate_realtime_predict.py -> TCP JSON lines (predictions)

## Ports

Default ports:

- Raw IMU ingest: 127.0.0.1:9001
- Predictions out: 127.0.0.1:9002

You can change ports using command-line flags.

## Script 1: BLE receiver

Start it after pairing with the sensor and knowing the BLE address.

Example:

python ble_realtime_receiver.py --address ED:35:33:D3:6C:F8

Options:

- --out_host: Host to send raw IMU JSON lines (default 127.0.0.1)
- --out_port: Port to send raw IMU JSON lines (default 9001)
- --print_every: Print every Nth frame to reduce spam
- --raw: Print raw hex frames

## Script 2: Processor + prediction

Start it first so the receiver can connect and stream.

Example:

python separate_realtime_predict.py --weights model_weights_best.weights.h5 --norm_npz norm_stats.npz

Options:

- --in_host / --in_port: Where to listen for raw IMU JSON lines (default 127.0.0.1:9001)
- --out_host / --out_port: Where to publish predictions (default 127.0.0.1:9002)
- --weights: Model weights file
- --norm_npz: Normalization stats
- --win_len: Window length (default 200)
- --buffer_mult: Raw buffer size multiplier (default 2)

## Raw IMU JSON schema

Each line is a JSON object with these keys:

- Time: ISO timestamp string
- Device name: BLE address (lowercase)
- Acceleration X(g), Acceleration Y(g), Acceleration Z(g)
- Angular velocity X(°/s), Angular velocity Y(°/s), Angular velocity Z(°/s)
- Angle X(°), Angle Y(°), Angle Z(°)
- Magnetic field X(uT), Magnetic field Y(uT), Magnetic field Z(uT)
- Quaternions 0(), Quaternions 1(), Quaternions 2(), Quaternions 3()

Example line:

{"Time":"2026-05-08T12:34:56.789","Device name":"ed:35:33:d3:6c:f8","Acceleration X(g)":0.01,"Acceleration Y(g)":-0.03,"Acceleration Z(g)":1.02,"Angular velocity X(°/s)":2.4,"Angular velocity Y(°/s)":0.1,"Angular velocity Z(°/s)":-0.5,"Angle X(°)":-1.2,"Angle Y(°)":0.4,"Angle Z(°)":90.3,"Magnetic field X(uT)":12.3,"Magnetic field Y(uT)":-4.5,"Magnetic field Z(uT)":33.1,"Quaternions 0()":0.01,"Quaternions 1()":0.02,"Quaternions 2()":-0.03,"Quaternions 3()":0.99}

## Predictions JSON schema

Each line is a JSON object with these keys:

- type: "prediction"
- Time: ISO timestamp string
- Predicted posture: label string
- Confidence: float percent

Example line:

{"type":"prediction","Time":"2026-05-08T12:35:01.234","Predicted posture":"posture_name","Confidence":78.4}

## Notes for spine reconstruction

Your reconstruction script can subscribe to the raw IMU stream on 9001 and ignore the prediction stream. The raw schema matches the existing preprocessing pipeline and includes acc, gyro, angle, mag, and quaternion values.
