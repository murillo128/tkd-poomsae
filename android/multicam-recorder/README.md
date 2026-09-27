# TKD MultiCam recorder prototype

Android Camera2 capture at the fixed 1080p120 mode exposed by camera 0. One phone is Controller; other phones are Camera nodes. The controller may record locally. All devices must be on the same reachable Wi-Fi network.

## Capture behavior

ARM starts recording a real high-speed pre-roll and keeps the preview alive. START schedules a shared logical beginning; it does not restart the camera or encoder. STOP finalizes the MP4 and publishes it to Movies/TKDPoomsae through MediaStore. The JSON sidecar records the timing markers and pre-roll. Clock synchronization is software-based, not hardware genlock; the pre-roll marker alone is not a claim of frame-exact alignment.

The current backend uses **automatic exposure and continuous autofocus**, not a guaranteed 1/500 shutter. It is the same source used by v1.3, which the user observed recording at approximately 119 fps on the SM-G991B. A new build compiling successfully does not constitute a new physical-camera test.

## Dashboard and preview

The v1.7 dashboard follows the original portrait and landscape visual references: original compact branding, preview, status controls, separate local-record switch, colored command grid and device cards. Status controls open full details. Errors remain visible as badges. Role selection is available from the Role control/options; landscape also shows a segmented selector. Peer entries show real device/status/RTT data, not invented live thumbnails or battery readings.

The TextureView default transform already handles camera mounting orientation and stretches the naturally-oriented image into the view. `PreviewGeometry` first reverses that stretch in **view coordinates**, then compensates display rotation and applies a single source-pixel scale. It never rotates by sensor orientation a second time. FIT (default) shows the entire frame, including side bars for a portrait capture in a wider container. FILL is an explicitly labelled display-only crop. Neither modifies the camera stream, encoder, saved orientation or recording dimensions.

Reference: Android Developers, "Support resizable surfaces in your camera app", sections 6 and Scaling the viewfinder: https://developer.android.com/codelabs/android-camera2-preview

## Validation

`gradle :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest` builds and checks 192 combinations of sensor orientation, display rotation, viewport dimensions and fit/fill mode. Checks compose with the default TextureView stretch and verify equal source-pixel scale, orthogonality, centering, no reflection and expected fit/crop boundaries.

`DashboardUiTest` runs on Android 15 in CI: portrait/reverse portrait/both landscapes, nonzero native control layouts, status dialogs and a synthetic circle/grid sent through the actual SurfaceTexture. It measures the rendered circle's width/height and emits PNGs. The synthetic source is explicitly marked and never replaces the camera in normal use. This validates UI/rendering only, not Samsung Camera2 high-speed capture or cross-phone synchronization.

CI also asserts that CameraEngine.java and NetworkCoordinator.java retain their pre-redesign blob hashes. Device verification should still cover camera 0 preview orientation, a new 10-second recording and multi-phone discovery/start/stop.

## Installing test builds

The v1.7 test package is separate from earlier debug builds to avoid their signing-certificate conflicts. It appears as TKD MultiCam 1.7. Videos already published to Gallery are not moved or deleted by this app.
