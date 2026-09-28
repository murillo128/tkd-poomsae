package dev.murillo.tkd.multicam;

import android.content.Context;
import android.graphics.Matrix;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.display.DisplayManager;
import android.os.Handler;
import android.os.Looper;
import android.view.Surface;
import android.view.TextureView;

/** Display-only adapter: never changes Camera2 sessions, recording or their buffers. */
public final class CameraPreview extends TextureView {
    private int sensorDegrees = 90;
    private boolean fill;
    private DisplayManager displays;
    private final DisplayManager.DisplayListener displayListener = new DisplayManager.DisplayListener() {
        @Override public void onDisplayAdded(int id) {}
        @Override public void onDisplayRemoved(int id) {}
        @Override public void onDisplayChanged(int id) { post(CameraPreview.this::refreshTransform); }
    };

    public CameraPreview(Context context) {
        super(context);
        setOpaque(false);
        setTag("camera_preview");
        try {
            CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
            Integer value = manager.getCameraCharacteristics(CameraEngine.CAMERA_ID)
                    .get(CameraCharacteristics.SENSOR_ORIENTATION);
            if (value != null) sensorDegrees = value;
        } catch (Exception ignored) {
            // The recording engine reports unavailable cameras on ARM.
        }
    }

    public int getSensorDegrees() { return sensorDegrees; }
    public boolean isFill() { return fill; }
    public void setFill(boolean value) { fill = value; refreshTransform(); }

    public void refreshTransform() {
        if (getWidth() <= 0 || getHeight() <= 0) return;
        int rotation = getDisplay() == null ? Surface.ROTATION_0 : getDisplay().getRotation();
        int degrees = rotation == Surface.ROTATION_90 ? 90
                : rotation == Surface.ROTATION_180 ? 180
                : rotation == Surface.ROTATION_270 ? 270 : 0;
        Matrix matrix = new Matrix();
        matrix.setValues(PreviewGeometry.matrix(getWidth(), getHeight(),
                CameraEngine.WIDTH, CameraEngine.HEIGHT, sensorDegrees, degrees, fill));
        setRotation(0f);
        setScaleX(1f);
        setScaleY(1f);
        setTransform(matrix);
    }

    @Override protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        refreshTransform();
    }
    @Override protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        displays = (DisplayManager) getContext().getSystemService(Context.DISPLAY_SERVICE);
        if (displays != null) displays.registerDisplayListener(displayListener, new Handler(Looper.getMainLooper()));
        post(this::refreshTransform);
    }
    @Override protected void onDetachedFromWindow() {
        if (displays != null) displays.unregisterDisplayListener(displayListener);
        super.onDetachedFromWindow();
    }
    @Override public void onWindowFocusChanged(boolean focused) {
        super.onWindowFocusChanged(focused);
        if (focused) post(this::refreshTransform);
    }
}
