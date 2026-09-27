# v1.8 passive light telemetry

High-speed `CaptureResult.SENSOR_EXPOSURE_TIME` and `SENSOR_SENSITIVITY` are observed
without setting exposure, ISO or a different FPS range. The preceding normal-preview
metering pass is not used as live data. Only advancing `SENSOR_TIMESTAMP` values count;
repeated or reordered results are ignored. The observer has a fixed 160-entry ring and
computes medians at most twice per second over the latest ~0.5 s of result arrivals.

The preview and Camera dialog show measured shutter/ISO with an advisory indicator.
Initial defaults: GOOD LIGHT when exposure <=2.5 ms and ISO <1600; LOW LIGHT above
2.5 ms; BLUR RISK above 4 ms; HIGH ISO at >=1600. Slow-shutter and high-ISO warnings can
coexist. Exposure recovery uses a 10% hysteresis margin, ISO recovery 20%, with a 0.5 s
stable transition. `LightMonitor.Config` owns these tunable defaults. They are heuristic
capture-quality thresholds, not calibrated illuminance or a guarantee of sharpness.
A bright scene can still have high ISO for other reasons. No automatic camera change or
recording stop is triggered by a warning.

Missing, incomplete or stale metadata is labelled UNKNOWN (not green). Local data expires
after 1.5 s without fresh results. The controller receives an optional `light` object in
existing v3 unicast sync replies. Reports contain session, sequence and measurement age;
different monotonic clocks are never compared directly. Outdated sequences/other-session
reports are discarded and remote readings expire after 3 s including sender-side age.
Clients without this optional field remain compatible and show light metadata unavailable.
Install v1.8 on camera nodes as well as the controller to see remote shutter/ISO.

JSON sidecars include bounded `light_monitor` aggregates (valid/missing/duplicate results,
slow-shutter/high-ISO counts and exposure/ISO extrema). These count observed metadata, NOT
encoded frames, because high-speed results can be batched. There is no raw per-frame log.

`check_capture_invariants.py` strips only explicitly marked passive-observer additions
from CameraEngine/NetworkCoordinator and verifies the exact original v1.7 blob hashes.
CameraPreview and PreviewGeometry must remain unchanged. Existing geometry/UI tests remain,
plus threshold/hysteresis/missing/stale/session tests and Android JSON/overlay/dialog checks.
The artifact suite now includes the exact Android source archive alongside the APK and tests.

Primary Android API documentation:
- https://developer.android.com/reference/android/hardware/camera2/CaptureResult#SENSOR_EXPOSURE_TIME
- https://developer.android.com/reference/android/hardware/camera2/CaptureResult#SENSOR_SENSITIVITY
- https://developer.android.com/reference/android/hardware/camera2/CameraConstrainedHighSpeedCaptureSession

The v1.8 test package is `dev.murillo.tkd.multicam.v18` (separate from previous debug APKs).
