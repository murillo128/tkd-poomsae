# TKD MultiCam Recorder prototype

Experimental Android app for synchronized multi-phone poomsae capture.

Modes:
- **Controller**: auto-discovers camera nodes on the same Wi-Fi, continuously estimates clock offsets, arms all devices, and schedules a common start time.
- **Camera**: waits for controller commands and records locally.

Capture defaults:
- Camera2 camera ID `0`
- 1920x1080
- 120 fps constrained high-speed session
- continuous-video autofocus
- OIS/EIS disabled
- target shutter 1/500 s, with ISO derived from a short AE metering pass

Each recording is written locally together with a JSON sidecar containing timing and exposure metadata.

This is a prototype: Wi-Fi broadcast discovery can be blocked by AP/client-isolation settings, and cross-device frame alignment should still be refined from recorded timestamps/visual motion during ingest.
