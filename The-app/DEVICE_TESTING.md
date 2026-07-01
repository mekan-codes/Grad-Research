# TinyPPGCollector Device Testing

This guide verifies the phone controller, Wear OS watch recorder, Polar H10 recorder, and session files on real devices.

## Build Outputs

After building, install these debug APKs:

```powershell
.\gradlew.bat :app:assembleDebug :wear:assembleDebug
```

Phone APK:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Wear APK:

```text
wear/build/outputs/apk/debug/wear-debug.apk
```

## Install Phone APK

1. Enable Developer options and USB debugging on the Android phone.
2. Connect the phone by USB.
3. Confirm the device is visible:

```powershell
adb devices
```

4. Install the phone app:

```powershell
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Install Wear APK

1. On the Galaxy Watch 5, enable Developer options.
2. Enable ADB debugging and Wireless debugging.
3. From the watch wireless debugging screen, note the IP address and port.
4. Connect from the computer:

```powershell
adb connect WATCH_IP:WATCH_PORT
adb devices
```

5. Install the wear app:

```powershell
adb -s WATCH_IP:WATCH_PORT install -r wear/build/outputs/apk/debug/wear-debug.apk
```

If USB or Wi-Fi ADB has multiple devices connected, always pass `-s DEVICE_SERIAL`.

## Pair Watch And Phone

1. Pair the Galaxy Watch 5 to the Android phone using the normal Galaxy Wearable setup.
2. Keep Bluetooth enabled on both devices.
3. Keep the phone app and watch app installed with the same package project.
4. Open TinyPPGCollector on the phone.
5. Open TinyPPG Watch on the watch and grant permissions.
6. Tap `Connect Watch` on the phone.
7. The phone should show `Watch connected` or a watch status message.

## Connect Polar H10

1. Wet the Polar H10 strap electrodes.
2. Wear the strap firmly on the chest.
3. Make sure the sensor pod is snapped into the strap.
4. Keep the strap close to the phone.
5. In the phone app, tap `Connect Polar H10`.
6. The phone should show the connected Polar device ID.

Do not pair the Polar H10 only through Android Bluetooth settings and assume it is connected to the app. The app scans and connects through the Polar BLE SDK.

## Required Permissions

Phone:

- Body sensors
- Activity recognition
- Nearby devices / Bluetooth scan
- Nearby devices / Bluetooth connect
- Notifications
- Location only on older Android versions where BLE scan requires it

Watch:

- Body sensors, or Samsung additional health data permission on newer Samsung/Wear OS builds
- Activity recognition
- Notifications

If a permission dialog is denied, reopen app settings and grant it manually.

## 30-Second Rest Test

1. Open the phone app.
2. Enter a subject ID, for example `test_rest`.
3. Set activity to `rest`.
4. Add note `30 second rest test`.
5. Tap `Connect Watch`.
6. Tap `Connect Polar H10`.
7. Sit still with the watch snug on the wrist and the Polar strap on the chest.
8. Tap `Start Recording`.
9. Wait 30 seconds.
10. Tap `Stop Recording`.
11. Confirm the saved folder and row-count summary are shown.

Expected quick checks:

- `watch_ppg.csv` should have roughly 25 rows per second if Samsung PPG is available.
- `watch_imu.csv` should have roughly 25 rows per second.
- `polar_hr.csv` should have HR rows.
- `polar_ecg.csv` should have ECG rows if the Polar ECG stream started.
- `prepared_session.csv` should have the same approximate row count as `watch_ppg.csv`.

## 2-Minute Motion Artifact Test

1. Enter a subject ID, for example `test_motion`.
2. Set activity to `motion_artifact`.
3. Add note `2 minute wrist motion artifact test`.
4. Connect watch and Polar H10.
5. Tap `Start Recording`.
6. Record 30 seconds still.
7. Move the watch arm for 60 seconds: wrist rotation, walking arm swing, and mild tapping.
8. Record 30 seconds still.
9. Tap `Stop Recording`.
10. Confirm saved folder and row-count summary.

Use this test to verify obvious motion artifacts are visible in `watch_ppg.csv`, `watch_imu.csv`, and `prepared_session.csv`.

## Saved Files

Files are saved on the phone under:

```text
Documents/TinyPPGCollector/session_yyyyMMdd_HHmmss_subjectID/
```

Expected files:

- `metadata.json`
- `watch_ppg.csv`
- `watch_imu.csv`
- `polar_ecg.csv`
- `polar_hr.csv`
- `polar_acc.csv`
- `events.csv`
- `prepared_session.csv`

Raw CSV files are not deleted or modified. `prepared_session.csv` is a derived quick-ingestion file using approximate nearest timestamp alignment.

## Check Row Counts

The phone UI shows row counts after Stop. You can also inspect files from a computer.

Pull the Documents folder through Android file transfer, or use ADB where supported:

```powershell
adb shell ls "/sdcard/Documents/TinyPPGCollector"
adb shell ls "/sdcard/Documents/TinyPPGCollector/session_yyyyMMdd_HHmmss_subjectID"
```

Count rows:

```powershell
adb shell wc -l "/sdcard/Documents/TinyPPGCollector/session_yyyyMMdd_HHmmss_subjectID/watch_ppg.csv"
adb shell wc -l "/sdcard/Documents/TinyPPGCollector/session_yyyyMMdd_HHmmss_subjectID/prepared_session.csv"
```

Subtract 1 from `wc -l` for the header row.

## Common Failures And Fixes

Watch not connected:

- Confirm the watch is paired to the phone.
- Keep Bluetooth on.
- Open the watch app once.
- Reinstall both APKs after changing package/builds.
- Restart Wear OS Bluetooth if Data Layer messages stop.

Samsung Health Sensor SDK unavailable:

- Confirm the Samsung Health Sensor SDK AAR is present in `wear/libs`.
- Confirm the Galaxy Watch model and firmware support `PPG_CONTINUOUS`.
- Grant the required watch sensor permission.

PPG tracker unavailable:

- Tighten the watch strap.
- Use a supported Samsung watch/firmware.
- Reopen the watch app and grant all permissions.
- Restart the watch if the Samsung sensor service is stuck.

Polar not found:

- Wet the strap electrodes.
- Wear the strap so the pod powers on.
- Keep it close to the phone.
- Make sure another app is not holding the connection.
- Toggle Bluetooth and try `Connect Polar H10` again.

ECG stream unavailable:

- Confirm the device is a Polar H10, not a basic HR-only sensor.
- Disconnect it from other apps.
- Reconnect in TinyPPGCollector.

No rows in `prepared_session.csv`:

- Check whether `watch_ppg.csv` has rows.
- The prepared file uses watch green PPG as the base timeline, so it is empty if watch PPG did not record.

Stop waits then reports timeout:

- The phone did not receive `WATCH_DONE` within the timeout.
- Raw phone-side Polar files should still close.
- Watch may continue retrying queued samples while it is connected; keep phone and watch near each other and run another short test.
