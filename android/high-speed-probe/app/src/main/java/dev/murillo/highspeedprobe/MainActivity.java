package dev.murillo.highspeedprobe;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.MediaMetadataRetriever;
import android.media.MediaRecorder;
import android.os.Build;
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

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
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
    private Surface recorderSurface;
    private MediaRecorder mediaRecorder;
    private File outputFile;

    private volatile long firstSensorTimestampNs = -1;
    private volatile long lastSensorTimestampNs = -1;
    private volatile long captureCount = 0;
    private volatile long captureFailedCount = 0;
    private volatile long bufferLostCount = 0;
    private volatile boolean testing = false;
    private volatile boolean closing = false;
    private volatile boolean recorderStarted = false;

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
        title.setText("HighSpeedProbe v2");
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
        append("Device: " + Build.MANUFACTURER + " " + Build.MODEL);
        append("Android: " + Build.VERSION.RELEASE + " (SDK " + Build.VERSION.SDK_INT + ")");
        append("Probe: v2 encoder-surface test");
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
                        // With preview + recording surfaces Android requires a fixed range.
                        if (range.getLower().equals(range.getUpper()) && range.getUpper() >= 120) {
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
        b.setText("RECORD cam " + cameraId + " · "
                + size.getWidth() + "x" + size.getHeight()
                + " @ " + fpsRange.getUpper() + " fps · 5 s");
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
        closing = false;
        recorderStarted = false;
        firstSensorTimestampNs = -1;
        lastSensorTimestampNs = -1;
        captureCount = 0;
        captureFailedCount = 0;
        bufferLostCount = 0;

        textureView.setVisibility(View.VISIBLE);
        append("TEST START: camera " + cameraId + " "
                + size.getWidth() + "x" + size.getHeight() + " @ " + fpsRange);
        append("  Mode: preview + H.264 encoder surface");

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
            prepareRecorder(size, fpsRange.getUpper());
        } catch (Exception e) {
            failTest("Recorder prepare: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            return;
        }

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
                    if (!closing) failTest("Camera disconnected");
                }

                @Override
                public void onError(CameraDevice camera, int error) {
                    camera.close();
                    if (!closing) {
                        failTest("Camera error " + error + " (" + cameraErrorName(error) + ")");
                    }
                }
            }, cameraHandler);
        } catch (Exception e) {
            failTest(e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    private void prepareRecorder(Size size, int fps) throws Exception {
        File dir = getExternalFilesDir("Movies");
        if (dir == null) dir = getFilesDir();
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("Cannot create output directory");
        }

        outputFile = new File(dir, "highspeed-" + size.getWidth() + "x" + size.getHeight()
                + "-" + fps + "fps-" + System.currentTimeMillis() + ".mp4");

        mediaRecorder = new MediaRecorder();
        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        mediaRecorder.setVideoSize(size.getWidth(), size.getHeight());
        mediaRecorder.setVideoFrameRate(fps);
        mediaRecorder.setVideoEncodingBitRate(fps >= 240 ? 40_000_000 : 24_000_000);
        mediaRecorder.setOutputFile(outputFile.getAbsolutePath());
        mediaRecorder.prepare();
        recorderSurface = mediaRecorder.getSurface();
    }

    private void createHighSpeedSession(Size size, Range<Integer> fpsRange) {
        try {
            List<Surface> outputs = new ArrayList<>();
            outputs.add(previewSurface);
            outputs.add(recorderSurface);

            cameraDevice.createConstrainedHighSpeedCaptureSession(
                    outputs,
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
                                builder.addTarget(recorderSurface);
                                builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange);
                                builder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
                                builder.set(CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);

                                List<CaptureRequest> requests =
                                        highSpeedSession.createHighSpeedRequestList(builder.build());

                                firstSensorTimestampNs = -1;
                                lastSensorTimestampNs = -1;
                                captureCount = 0;
                                captureFailedCount = 0;
                                bufferLostCount = 0;

                                highSpeedSession.setRepeatingBurst(
                                        requests, captureCallback, cameraHandler);

                                mediaRecorder.start();
                                recorderStarted = true;

                                runOnUiThread(() -> append(
                                        "  Session configured; recorder started; sampling for 5 s..."));
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
                    if (ts == null) return;
                    if (firstSensorTimestampNs < 0) firstSensorTimestampNs = ts;
                    lastSensorTimestampNs = ts;
                    captureCount++;
                }

                @Override
                public void onCaptureFailed(
                        CameraCaptureSession session,
                        CaptureRequest request,
                        android.hardware.camera2.CaptureFailure failure) {
                    captureFailedCount++;
                }

                @Override
                public void onCaptureBufferLost(
                        CameraCaptureSession session,
                        CaptureRequest request,
                        Surface target,
                        long frameNumber) {
                    bufferLostCount++;
                }
            };

    private void finishTest(Size size, Range<Integer> fpsRange) {
        closing = true;
        try {
            if (highSpeedSession != null) {
                highSpeedSession.stopRepeating();
            }
        } catch (Exception ignored) {}

        try {
            if (recorderStarted && mediaRecorder != null) {
                mediaRecorder.stop();
            }
        } catch (Exception e) {
            runOnUiThread(() -> append("  Recorder stop warning: "
                    + e.getClass().getSimpleName() + ": " + e.getMessage()));
        }
        recorderStarted = false;

        long count = captureCount;
        long failed = captureFailedCount;
        long lost = bufferLostCount;
        long first = firstSensorTimestampNs;
        long last = lastSensorTimestampNs;
        double measuredFps = 0.0;
        double sensorSeconds = 0.0;

        if (count > 1 && first >= 0 && last > first) {
            sensorSeconds = (last - first) / 1_000_000_000.0;
            measuredFps = (count - 1) / sensorSeconds;
        }

        String metadata = inspectRecordedFile(outputFile);
        long fileBytes = outputFile != null && outputFile.exists() ? outputFile.length() : 0;

        final double finalMeasuredFps = measuredFps;
        final double finalSensorSeconds = sensorSeconds;
        final long finalCount = count;
        final long finalFailed = failed;
        final long finalLost = lost;
        final long finalFileBytes = fileBytes;
        final String finalMetadata = metadata;
        final String finalPath = outputFile != null ? outputFile.getAbsolutePath() : "(none)";

        runOnUiThread(() -> {
            append(String.format(Locale.US,
                    "  SENSOR RESULT: %,d completed over %.3f s -> %.2f callbacks/s",
                    finalCount, finalSensorSeconds, finalMeasuredFps));
            append("  Capture failures: " + finalFailed + "  buffer lost: " + finalLost);
            append("  Requested: " + size.getWidth() + "x" + size.getHeight()
                    + " @ " + fpsRange);
            append("  MP4: " + finalPath);
            append(String.format(Locale.US, "  MP4 size: %.2f MiB",
                    finalFileBytes / 1048576.0));
            append("  MP4 metadata: " + finalMetadata);
            append("");
        });

        closeCameraInternal();
        testing = false;
        closing = false;
    }

    private String inspectRecordedFile(File file) {
        if (file == null || !file.exists() || file.length() == 0) return "no file";
        MediaMetadataRetriever mmr = new MediaMetadataRetriever();
        try {
            mmr.setDataSource(file.getAbsolutePath());
            String duration = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            String captureFps = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE);
            String frames = null;
            if (Build.VERSION.SDK_INT >= 28) {
                frames = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT);
            }
            return "durationMs=" + duration + ", captureFps=" + captureFps + ", frames=" + frames;
        } catch (Exception e) {
            return "metadata error: " + e.getClass().getSimpleName() + ": " + e.getMessage();
        } finally {
            try { mmr.release(); } catch (Exception ignored) {}
        }
    }

    private void failTest(String message) {
        runOnUiThread(() -> {
            append("  TEST FAILED: " + message);
            append("");
            Toast.makeText(this, "Test failed: " + message, Toast.LENGTH_LONG).show();
        });
        closing = true;
        try {
            if (recorderStarted && mediaRecorder != null) mediaRecorder.stop();
        } catch (Exception ignored) {}
        recorderStarted = false;
        closeCameraInternal();
        testing = false;
        closing = false;
    }

    private void closeCamera() {
        if (cameraHandler != null) cameraHandler.post(this::closeCameraInternal);
    }

    private void closeCameraInternal() {
        closing = true;

        try {
            if (highSpeedSession != null) highSpeedSession.close();
        } catch (Exception ignored) {}
        highSpeedSession = null;

        try {
            if (cameraDevice != null) cameraDevice.close();
        } catch (Exception ignored) {}
        cameraDevice = null;

        try {
            if (mediaRecorder != null) mediaRecorder.release();
        } catch (Exception ignored) {}
        mediaRecorder = null;

        try {
            if (recorderSurface != null) recorderSurface.release();
        } catch (Exception ignored) {}
        recorderSurface = null;

        try {
            if (previewSurface != null) previewSurface.release();
        } catch (Exception ignored) {}
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

    private String cameraErrorName(int error) {
        switch (error) {
            case CameraDevice.StateCallback.ERROR_CAMERA_IN_USE: return "ERROR_CAMERA_IN_USE";
            case CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE: return "ERROR_MAX_CAMERAS_IN_USE";
            case CameraDevice.StateCallback.ERROR_CAMERA_DISABLED: return "ERROR_CAMERA_DISABLED";
            case CameraDevice.StateCallback.ERROR_CAMERA_DEVICE: return "ERROR_CAMERA_DEVICE";
            case CameraDevice.StateCallback.ERROR_CAMERA_SERVICE: return "ERROR_CAMERA_SERVICE";
            default: return "UNKNOWN";
        }
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
        if (cameraThread != null) cameraThread.quitSafely();
        super.onDestroy();
    }
}
