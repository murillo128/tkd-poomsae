package dev.murillo.tkd.multicam;

import android.app.Activity;
import android.hardware.SensorManager;
import android.os.SystemClock;
import android.view.OrientationEventListener;

/** Tracks physical orientation even with Android auto-rotate disabled or the app UI locked. */
final class DeviceOrientation {
    private final Activity activity;
    private final OrientationEventListener listener;
    private int quarter = -1;
    private long measuredNs = -1;
    private boolean unknown = true;

    DeviceOrientation(Activity activity) {
        this.activity = activity;
        listener = new OrientationEventListener(activity, SensorManager.SENSOR_DELAY_NORMAL) {
            @Override public void onOrientationChanged(int degrees) {
                if (degrees == ORIENTATION_UNKNOWN) { unknown = true; return; }
                quarter = RecordingOrientation.stableQuarter(quarter, degrees);
                measuredNs = SystemClock.elapsedRealtimeNanos();
                unknown = false;
            }
        };
        if (listener.canDetectOrientation()) listener.enable();
    }
    RecordingOrientation.Choice read(int sensorDegrees) {
        if (quarter >= 0) {
            return new RecordingOrientation.Choice(RecordingOrientation.fromPhysical(sensorDegrees, quarter, false),
                    unknown ? "physical_last_known" : "physical_sensor",
                    Math.max(0, SystemClock.elapsedRealtimeNanos() - measuredNs), unknown);
        }
        // WindowManager supports older devices too. Display is only a named fallback,
        // never mistaken for a physical orientation while auto-rotate is disabled.
        int displayDegrees = activity.getWindowManager().getDefaultDisplay().getRotation() * 90;
        return new RecordingOrientation.Choice(RecordingOrientation.fromDisplay(sensorDegrees, displayDegrees, false),
                "display_fallback", -1, true);
    }
    void close() { listener.disable(); }
}
