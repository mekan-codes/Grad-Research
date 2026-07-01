# TinyPPGCollector Final Audit

Audit date: 2026-06-29

## Build Status

Last verified command:

```powershell
.\gradlew.bat :app:assembleDebug :wear:assembleDebug
```

Result:

```text
BUILD SUCCESSFUL
```

Phone app build: passes.

Wear app build: passes.

## APK Paths

Phone APK:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Wear APK:

```text
wear/build/outputs/apk/debug/wear-debug.apk
```

## Architecture

The existing architecture is preserved:

- `app`: Android phone controller, Polar recorder, Wear OS command client, session storage, CSV export.
- `wear`: Wear OS watch app/service, Samsung PPG recorder, IMU recorder, batched Data Layer sender.

Phone remains the master coordinator:

1. Creates the session folder and raw CSV files.
2. Starts Polar streams.
3. Sends `START_SESSION` to watch with `phone_start_elapsed_realtime_ns`.
4. Sends `STOP_SESSION` to watch.
5. Stops Polar streams.
6. Waits for processed `WATCH_DONE` or timeout.
7. Writes final metadata, validates files, and creates `prepared_session.csv`.

## Real Streams

Watch PPG:

- Real integration through Samsung Health Sensor SDK reflection.
- Uses `PPG_CONTINUOUS`.
- Records green, red, infrared values when available.
- Requires Samsung Health Sensor SDK AAR in `wear/libs`, supported Galaxy Watch firmware, and sensor permissions.

Watch IMU:

- Real Android `SensorManager` accelerometer recording.
- Target sampling period is 40,000 us, approximately 25 Hz.

Polar ECG:

- Real Polar BLE SDK ECG stream.
- Uses Polar SDK timestamps when available in `sensor_timestamp_ns`.

Polar HR/RR:

- Real Polar BLE SDK HR stream.
- Writes one row per RR interval when multiple RR intervals are included in one HR sample.

Polar ACC:

- Real optional Polar BLE SDK accelerometer stream when the device advertises ACC support.
- Session continues if unavailable.

## Scaffold Or Not Implemented

Galaxy Watch ECG:

- Not implemented by design.
- The app uses Polar H10 ECG as the continuous ECG reference.

Samsung SDK dependency:

- The wear module is ready for the Samsung Health Sensor SDK AAR.
- The SDK AAR itself is not committed by this project and must be placed in `wear/libs` according to Samsung SDK terms/instructions.

Synthetic/fake core streams:

- No core required stream is intentionally fake.
- If a real sensor or SDK is unavailable, the app records clear errors/warnings instead of generating fake data.

## Saved Folder Location

Every completed session is saved on the phone under:

```text
Documents/TinyPPGCollector/session_yyyyMMdd_HHmmss_subjectID/
```

## Output Files

Raw files:

- `metadata.json`
- `watch_ppg.csv`
- `watch_imu.csv`
- `polar_ecg.csv`
- `polar_hr.csv`
- `polar_acc.csv`
- `events.csv`

Derived file:

- `prepared_session.csv`

Raw CSV files are not deleted or modified when `prepared_session.csv` is generated.

## CSV Schemas

`watch_ppg.csv`

```text
session_id,timestamp_unix_ms,elapsed_realtime_ns,sensor_timestamp_ns,sample_index,ppg_green,ppg_red,ppg_ir,status
```

`watch_imu.csv`

```text
session_id,timestamp_unix_ms,elapsed_realtime_ns,sensor_timestamp_ns,sample_index,acc_x,acc_y,acc_z
```

`polar_ecg.csv`

```text
session_id,timestamp_unix_ms,elapsed_realtime_ns,sensor_timestamp_ns,sample_index,ecg_uv
```

`polar_hr.csv`

```text
session_id,timestamp_unix_ms,elapsed_realtime_ns,sample_index,hr_bpm,rr_ms
```

`polar_acc.csv`

```text
session_id,timestamp_unix_ms,elapsed_realtime_ns,sensor_timestamp_ns,sample_index,acc_x,acc_y,acc_z
```

`events.csv`

```text
session_id,timestamp_unix_ms,elapsed_realtime_ns,event_type,event_label,notes
```

`prepared_session.csv`

```text
time,ppg,hr,acc_x,acc_y,acc_z,ecg
```

## Prepared Session Export

`prepared_session.csv` is generated after a completed session.

Rules:

- Base timeline is `watch_ppg.csv`.
- `time` is seconds from the first watch PPG timestamp.
- `ppg` is watch green PPG.
- `acc_x`, `acc_y`, `acc_z` are nearest watch accelerometer samples by `timestamp_unix_ms`.
- `hr` is nearest Polar HR sample by `timestamp_unix_ms`.
- `ecg` is nearest Polar ECG sample by `timestamp_unix_ms`.
- Alignment is approximate nearest timestamp alignment for quick TinyPPG/ML ingestion.
- Raw files remain authoritative for scientific analysis.

`metadata.json` records:

- `prepared_session_file`
- `prepared_session_generated`
- `prepared_session_row_count`
- `prepared_session_note`

## Metadata Coverage

`metadata.json` includes:

- Session ID
- Subject ID
- Activity label
- Notes
- Start and stop Unix timestamps
- Phone model
- Watch model when received from watch status
- Polar device ID
- App version
- Watch target/sample-rate fields
- Polar ECG rate if known
- SDK availability flags
- Stream started/stopped flags
- Sample counts for watch PPG, watch IMU, Polar ECG, Polar HR, Polar ACC
- Errors/warnings
- Prepared export status and note

## Start/Stop Reliability

Start:

- Phone creates the session folder and all raw CSV headers.
- Phone writes `START`.
- Phone starts Polar.
- Phone sends `START_SESSION` to watch with `phone_start_elapsed_realtime_ns`.
- Watch starts PPG and IMU after `START_SESSION`.
- Watch sends `WATCH_STARTED` after both PPG and IMU report active.
- Watch sends periodic `WATCH_STATUS`.

Stop:

- Phone writes `STOP_REQUESTED`.
- Phone sends `STOP_SESSION`.
- Watch stops sensors, flushes PPG/IMU buffers, drains queued Data Layer messages, then sends `WATCH_DONE`.
- Phone stops Polar streams.
- Phone waits for processed `WATCH_DONE` or timeout.
- Phone writes `STOP_COMPLETE`.
- Phone closes raw writers, writes `prepared_session.csv`, writes final metadata, and validates files.

## File Validation

After Stop, the app validates:

- `metadata.json` exists
- `events.csv` exists and has the expected header
- `watch_ppg.csv` exists and has the expected header
- `watch_imu.csv` exists and has the expected header
- `polar_ecg.csv` exists and has the expected header
- `polar_hr.csv` exists and has the expected header
- `polar_acc.csv` exists and has the expected header
- `prepared_session.csv` exists and has the expected header

The phone UI shows the saved folder path and validation row counts.

## Known Limitations

- `prepared_session.csv` uses approximate nearest timestamp alignment only; it does not resample, interpolate, or correct clock drift.
- The base timeline for `prepared_session.csv` is watch green PPG, so the file has zero data rows if watch PPG fails.
- Samsung PPG requires the Samsung Health Sensor SDK AAR and supported Galaxy Watch firmware/hardware.
- Wear Data Layer delivery depends on watch-phone connectivity. The watch queues batches and retries, but the phone stop flow still uses a timeout.
- Polar SDK is pinned to `6.4.0` to match the current project Kotlin toolchain. Newer SDK versions previously produced incompatible Kotlin metadata with this project setup.
- `polar_acc.csv` is optional because Polar ACC availability depends on the connected device and SDK-advertised features.
- No cloud upload, database, or background server is used. All files remain local on the phone.

## Device Testing Guide

See:

```text
DEVICE_TESTING.md
```
