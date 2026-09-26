package dev.murillo.highspeedprobe;

import android.Manifest;
import android.app.Activity;
import android.content.ClipboardManager;
import android.content.ClipData;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Range;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final int CAMERA_PERMISSION_REQUEST = 1001;
    private static final long TEST_DURATION_MS = 5000;

    private CameraManager cameraManager;
    private TextView reportView;
    private LinearLayout testButtons;
    private TextureView textureView;

    private HandlerThread cameraThread;
    private Handler cameraHandler;

    private CameraDevice cameraDevice;
    private CameraConstrainedHighSpeedCaptureSession highSpeedSession;
    private Surface previewSurface;

    private volatile long firstSensorTimestampNs = -1;
    private volatile long lastSensorTimestampNs = -1;
    private volatile long captureCount = 0;
    private volatile boolean testing = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);

        cameraThread = new HandlerThread("HighSpeedProbeCamera");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());

        buildUi();

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            scanCameras();
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
        }
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(12), dp(12), dp(12), dp(12));

        TextView title = new TextView(this);
        title.setText("HighSpeedProbe");
        title.setTextSize(24);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(title, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.HORIZONTAL);

        Button scan = new Button(this);
        scan.setText("SCAN");
        scan.setOnClickListener(v -> scanCameras());
        controls.addView(scan, new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        Button copy = new Button(this);
        copy.setText("COPY REPORT");
        copy.setOnClickListener(v -> copyReport());
        controls.addView(copy, new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        root.addView(controls);

        textureView = new TextureView(this);
        textureView.setVisibility(View.GONE);
        root.addView(textureView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(230)));

        ScrollView scroll = new ScrollView(this);
        LinearLayout body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);

        reportView = new TextView(this);
        reportView.setTextSize(13);
        reportView.setTextIsSelectable(true);
        reportView.setPadding(dp(4), dp(8), dp(4), dp(8));
        body.addView(reportView);

        testButtons = new LinearLayout(this);
        testButtons.setOrientation(LinearLayout.VERTICAL);
        body.addView(testButtons);

        scroll.addView(body);
        root.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));

        setContentView(root);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void copyReport() {
        ClipboardManager cm = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        cm.setPrimaryClip(ClipData.newPlainText("HighSpeedProbe", reportView.getText()));
        Toast.makeText(this, "Report copied", Toast.LENGTH_SHORT).show();
    }

    private void scanCameras() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
            return;
        }

        closeCamera();
        testButtons.removeAllViews();
        reportView.setText("");
        append("Device: " + android.os.Build.MANUFACTURER + " " + android.os.Build.MODEL);
        append("Android: " + android.os.Build.VERSION.RELEASE + " (SDK " + android.os.Build.VERSION.SDK_INT + ")");
        append("");

        try {
            String[] ids = cameraManager.getCameraIdList();
            for (String id : ids) {
                CameraCharacteristics c = cameraManager.getCameraCharacteristics(id);
                Integer facing = c.get(CameraCharacteristics.LENS_FACING);
                Integer level = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
                int[] caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);

                boolean highSpeed = false;
                if (caps != null) {
                    for (int cap : caps) {
                        if (cap == CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_CONSTRAINED_HIGH_SPEED_VIDEO) {
                            highSpeed = true;
                            break;
                        }
                    }
                }

                append("Camera " + id + "  facing=" + facingName(facing)
                        + "  level=" + hardwareLevelName(level));
                append("  CONSTRAINED_HIGH_SPEED_VIDEO: " + (highSpeed ? "YES" : "NO"));

                StreamConfigurationMap map =
                        c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                if (!highSpeed || map == null) {
                    append("");
                    continue;
                }

                Size[] sizes = map.getHighSpeedVideoSizes();
                Arrays.sort(sizes, Comparator
                        .comparingLong((Size s) -> (long) s.getWidth() * s.getHeight())
                        .reversed());

                for (Size size : sizes) {
                    Range<Integer>[] ranges = map.getHighSpeedVideoFpsRangesFor(size);
                    append("  " + size.getWidth() + "x" + size.getHeight()
                            + " -> " + Arrays.toString(ranges));

                    for (Range<Integer> range : ranges) {
                        if (range.getUpper() >= 120) {
                            addTestButton(id, size, range);
                        }
                    }
                }
                append("");
            }
        } catch (Exception e) {
            append("SCAN ERROR: " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    private void addTestButton(String cameraId, Size size, Range<Integer> fpsRange) {
        Button b = new Button(this);
        b.setAllCaps(false);
        b.setText("Test cam " + cameraId + " · "
                + size.getWidth() + "x" + size.getHeight()
                + " @ " + fpsRange + " for 5 s");
        b.setOnClickListener(v -> startHighSpeedTest(cameraId, size, fpsRange));
        testButtons.addView(b, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));
    }

    private void startHighSpeedTest(String cameraId, Size size, Range<Integer> fpsRange) {
        if (testing) {
            Toast.makeText(this, "A test is already running", Toast.LENGTH_SHORT).show();
            return;
        }
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            return;
        }

        testing = true;
        firstSensorTimestampNs = -1;
        lastSensorTimestampNs = -1;
        captureCount = 0;
        textureView.setVisibility(View.VISIBLE);
        append("TEST START: camera " + cameraId + " "
                + size.getWidth() + "x" + size.getHeight() + " @ " + fpsRange);

        Runnable open = () -> cameraHandler.post(() -> openHighSpeedCamera(cameraId, size, fpsRange));

        if (textureView.isAvailable()) {
            open.run();
        } else {
            textureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
                    textureView.setSurfaceTextureListener(null);
                    open.run();
                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {}

                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
                    return true;
                }

                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture surface) {}
            });
        }
    }

    private void openHighSpeedCamera(String cameraId, Size size, Range<Integer> fpsRange) {
        closeCameraInternal();

        SurfaceTexture st = textureView.getSurfaceTexture();
        if (st == null) {
            failTest("TextureView surface unavailable");
            return;
        }
        st.setDefaultBufferSize(size.getWidth(), size.getHeight());
        previewSurface = new Surface(st);

        try {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                failTest("Camera permission missing");
                return;
            }

            cameraManager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice camera) {
                    cameraDevice = camera;
                    createHighSpeedSession(size, fpsRange);
                }

                @Override
                public void onDisconnected(CameraDevice camera) {
                    camera.close();
                    failTest("Camera disconnected");
                }

                @Override
                public void onError(CameraDevice camera, int error) {
                    camera.close();
                    failTest("Camera error " + error);
                }
            }, cameraHandler);
        } catch (Exception e) {
            failTest(e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    private void createHighSpeedSession(Size size, Range<Integer> fpsRange) {
        try {
            cameraDevice.createConstrainedHighSpeedCaptureSession(
                    Collections.singletonList(previewSurface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            if (!(session instanceof CameraConstrainedHighSpeedCaptureSession)) {
                                failTest("Session is not constrained high-speed");
                                session.close();
                                return;
                            }

                            highSpeedSession = (CameraConstrainedHighSpeedCaptureSession) session;
                            try {
                                CaptureRequest.Builder builder =
                                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
                                builder.addTarget(previewSurface);
                                builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange);
                                builder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
                                builder.set(CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);

                                List<CaptureRequest> requests =
                                        highSpeedSession.createHighSpeedRequestList(builder.build());

                                firstSensorTimestampNs = -1;
                                lastSensorTimestampNs = -1;
                                captureCount = 0;

                                highSpeedSession.setRepeatingBurst(
                                        requests, captureCallback, cameraHandler);

                                runOnUiThread(() -> append("  Session configured; sampling for 5 s..."));
                                cameraHandler.postDelayed(
                                        () -> finishTest(size, fpsRange), TEST_DURATION_MS);
                            } catch (Exception e) {
                                failTest(e.getClass().getSimpleName() + ": " + e.getMessage());
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            failTest("High-speed session configuration FAILED");
                        }
                    },
                    cameraHandler);
        } catch (Exception e) {
            failTest(e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    private final CameraCaptureSession.CaptureCallback captureCallback =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(
                        CameraCaptureSession session,
                        CaptureRequest request,
                        TotalCaptureResult result) {
                    Long ts = result.get(CaptureResult.SENSOR_TIMESTAMP);
                    if (ts == null) {
                        return;
                    }
                    if (firstSensorTimestampNs < 0) {
                        firstSensorTimestampNs = ts;
                    }
                    lastSensorTimestampNs = ts;
                    captureCount++;
                }
            };

    private void finishTest(Size size, Range<Integer> fpsRange) {
        try {
            if (highSpeedSession != null) {
                highSpeedSession.stopRepeating();
                highSpeedSession.abortCaptures();
            }
        } catch (Exception ignored) {
        }

        long count = captureCount;
        long first = firstSensorTimestampNs;
        long last = lastSensorTimestampNs;
        double measuredFps = 0.0;
        double sensorSeconds = 0.0;

        if (count > 1 && first >= 0 && last > first) {
            sensorSeconds = (last - first) / 1_000_000_000.0;
            measuredFps = (count - 1) / sensorSeconds;
        }

        final double finalMeasuredFps = measuredFps;
        final double finalSensorSeconds = sensorSeconds;
        runOnUiThread(() -> {
            append(String.format(Locale.US,
                    "  RESULT: %,d capture results over %.3f s sensor time -> %.2f fps",
                    count, finalSensorSeconds, finalMeasuredFps));
            append("  Requested: " + size.getWidth() + "x" + size.getHeight()
                    + " @ " + fpsRange);
            append("");
        });

        closeCameraInternal();
        testing = false;
    }

    private void failTest(String message) {
        runOnUiThread(() -> {
            append("  TEST FAILED: " + message);
            append("");
            Toast.makeText(this, "Test failed: " + message, Toast.LENGTH_LONG).show();
        });
        closeCameraInternal();
        testing = false;
    }

    private void closeCamera() {
        if (cameraHandler != null) {
            cameraHandler.post(this::closeCameraInternal);
        }
    }

    private void closeCameraInternal() {
        try {
            if (highSpeedSession != null) {
                highSpeedSession.close();
            }
        } catch (Exception ignored) {
        }
        highSpeedSession = null;

        try {
            if (cameraDevice != null) {
                cameraDevice.close();
            }
        } catch (Exception ignored) {
        }
        cameraDevice = null;

        try {
            if (previewSurface != null) {
                previewSurface.release();
            }
        } catch (Exception ignored) {
        }
        previewSurface = null;
    }

    private void append(String text) {
        if (android.os.Looper.myLooper() != android.os.Looper.getMainLooper()) {
            runOnUiThread(() -> append(text));
            return;
        }
        reportView.append(text + "\n");
    }

    private String facingName(Integer facing) {
        if (facing == null) return "UNKNOWN";
        if (facing == CameraCharacteristics.LENS_FACING_BACK) return "BACK";
        if (facing == CameraCharacteristics.LENS_FACING_FRONT) return "FRONT";
        if (facing == CameraCharacteristics.LENS_FACING_EXTERNAL) return "EXTERNAL";
        return String.valueOf(facing);
    }

    private String hardwareLevelName(Integer level) {
        if (level == null) return "UNKNOWN";
        if (level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) return "LEGACY";
        if (level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED) return "LIMITED";
        if (level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL) return "FULL";
        if (level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3) return "LEVEL_3";
        if (level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL) return "EXTERNAL";
        return String.valueOf(level);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            scanCameras();
        } else if (requestCode == CAMERA_PERMISSION_REQUEST) {
            append("Camera permission denied.");
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        closeCamera();
        testing = false;
    }

    @Override
    protected void onDestroy() {
        closeCameraInternal();
        if (cameraThread != null) {
            cameraThread.quitSafely();
        }
        super.onDestroy();
    }
}
