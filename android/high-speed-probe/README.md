# HighSpeedProbe

Small Android Camera2 diagnostic app used to verify whether a device really exposes and can run
`CONSTRAINED_HIGH_SPEED_VIDEO` sessions.

It:

- enumerates every Camera2 camera;
- prints `getHighSpeedVideoSizes()` and `getHighSpeedVideoFpsRangesFor(size)`;
- creates a real constrained high-speed preview session for five seconds;
- counts capture results using `SENSOR_TIMESTAMP` and reports the measured rate.

This is intentionally a probe, not the final multi-phone recorder. It requests no network or
storage permissions and does not save video.

## Build

```sh
gradle :app:assembleDebug
```
