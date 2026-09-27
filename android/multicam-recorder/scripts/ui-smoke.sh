#!/usr/bin/env bash
set -euo pipefail
mkdir -p ui-proof
collect() {
  adb pull /sdcard/Android/data/dev.murillo.tkd.multicam.v17/files/ui-proof/. ui-proof/ >/dev/null 2>&1 || true
  adb logcat -d > ui-proof/logcat.txt || true
}
trap collect EXIT
adb shell wm size 1080x2400
adb shell wm density 420
adb install -r artifacts/app/app-debug.apk
adb install -r artifacts/test/app-debug-androidTest.apk
adb shell am instrument -w -r dev.murillo.tkd.multicam.v17.test/androidx.test.runner.AndroidJUnitRunner | tee ui-proof/instrumentation.txt
grep -q 'OK (' ui-proof/instrumentation.txt
! grep -q 'FAILURES!!!' ui-proof/instrumentation.txt
