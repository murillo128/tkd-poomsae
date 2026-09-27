#!/usr/bin/env bash
set -euo pipefail
mkdir -p ui-proof
collect() {
  adb pull /sdcard/Android/data/dev.murillo.tkd.multicam.v18/files/ui-proof/. ui-proof/ >/dev/null 2>&1 || true
  adb logcat -d > ui-proof/logcat.txt || true
}
trap collect EXIT
# A cold Google-APIs image can show a Pixel Launcher ANR over an otherwise
# healthy test activity. Evidence from the first two attempts includes that
# system dialog stealing focus. This setting is EMULATOR-ONLY; crashes or ANRs
# in our app still fail instrumentation, focus checks and the package log gate.
adb shell settings put global hide_error_dialogs 1
adb shell am force-stop com.google.android.apps.nexuslauncher
adb shell input keyevent KEYCODE_WAKEUP
adb shell wm dismiss-keyguard
# pixel_7 already supplies 1080x2400 at 420dpi. Avoid unnecessary hot display
# reconfiguration while system packages are still finishing their first boot.
adb install -r artifacts/app/app-debug.apk
adb install -r artifacts/test/app-debug-androidTest.apk
adb logcat -c
adb shell am instrument -w -r dev.murillo.tkd.multicam.v18.test/androidx.test.runner.AndroidJUnitRunner | tee ui-proof/instrumentation.txt
grep -q 'OK (' ui-proof/instrumentation.txt
! grep -q 'FAILURES!!!' ui-proof/instrumentation.txt
adb logcat -d > ui-proof/logcat.txt
! grep -E 'ANR in dev\.murillo\.tkd\.multicam|Process: dev\.murillo\.tkd\.multicam.*PID:' ui-proof/logcat.txt
