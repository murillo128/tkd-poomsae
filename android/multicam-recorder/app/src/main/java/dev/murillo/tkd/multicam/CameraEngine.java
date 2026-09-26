package dev.murillo.tkd.multicam;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.ContentValues;
import android.content.ContentResolver;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.MediaRecorder;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.net.Uri;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Range;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;

import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public final class CameraEngine {
    public static final String CAMERA_ID = "0";
    public static final int WIDTH = 1920;
    public static final int HEIGHT = 1080;
    public static final int FPS = 120;
    public static final long TARGET_EXPOSURE_NS = 2_000_000L; // 1/500 s

    public enum State {
        IDLE, ARMING, READY, RECORDING, STOPPING, ERROR
    }

    public interface Listener {
        void onCameraStatus(String status);
        void onCameraReady(String details);
        void onCameraStarted(long localStartCallNs);
        void onCameraStopped(String videoPath, String metadataPath);
        void onCameraError(String error);
    }

    private final Activity activity;
    private final TextureView textureView;
    private final Listener listener;
    private final CameraManager cameraManager;
    private final HandlerThread cameraThread;
    private final Handler cameraHandler;

    private volatile State state = State.IDLE;

    private CameraCharacteristics characteristics;
    private CameraDevice cameraDevice;
    private CameraCaptureSession previewSession;
    private CameraConstrainedHighSpeedCaptureSession highSpeedSession;
    private Surface previewSurface;
    private Surface recorderSurface;
    private MediaRecorder mediaRecorder;
    private List<CaptureRequest> highSpeedRequests;
    private List<CaptureRequest> highSpeedPreviewRequests;

    private String sessionId;
    private File videoFile;
    private File metadataFile;

    private volatile long meteredExposureNs = 8_000_000L;
    private volatile int meteredIso = 200;
    private volatile long actualExposureNs = TARGET_EXPOSURE_NS;
    private volatile int actualIso = 200;

    private volatile long scheduledStartNs = -1L;
    private volatile long startCallNs = -1L;
    private volatile long stopCallNs = -1L;
    private volatile long firstSensorTimestampNs = -1L;
    private volatile long lastUniqueSensorTimestampNs = -1L;
    private volatile long uniqueSensorFrames = 0L;
    private volatile long captureFailures = 0L;
    private volatile long encodedFrameCount = 0L;
    private volatile long encodedDurationUs = 0L;
    private volatile double encodedFps = 0.0;

    private volatile boolean recorderStarted = false;
    private volatile boolean closing = false;

    public CameraEngine(Activity activity, TextureView textureView, Listener listener) {
        this.activity = activity;
        this.textureView = textureView;
        this.listener = listener;
        this.cameraManager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        this.cameraThread = new HandlerThread("TkdMultiCamCamera");
        this.cameraThread.start();
        this.cameraHandler = new Handler(cameraThread.getLooper());
    }

    public State getState() {
        return state;
    }

    public String getRecordingPath() {
        return videoFile == null ? null : videoFile.getAbsolutePath();
    }

    public double getLastEncodedFps() {
        return encodedFps;
    }

    public long getLastEncodedFrameCount() {
        return encodedFrameCount;
    }

    public void arm(String newSessionId) {
        cameraHandler.post(() -> armInternal(newSessionId));
    }

    public void startAt(long localElapsedRealtimeNs) {
        cameraHandler.post(() -> scheduleStartInternal(localElapsedRealtimeNs));
    }

    public void stop() {
        cameraHandler.post(this::stopInternal);
    }

    public void shutdown() {
        cameraHandler.post(() -> {
            closeAll();
            cameraThread.quitSafely();
        });
    }

    private void armInternal(String newSessionId) {
        if (state != State.IDLE && state != State.ERROR) {
            status("Ignoring ARM while state=" + state);
            return;
        }
        if (activity.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            error("Camera permission is not granted");
            return;
        }

        closeAll();
        state = State.ARMING;
        closing = false;
        sessionId = sanitizeSessionId(newSessionId);
        scheduledStartNs = -1L;
        startCallNs = -1L;
        stopCallNs = -1L;
        firstSensorTimestampNs = -1L;
        lastUniqueSensorTimestampNs = -1L;
        uniqueSensorFrames = 0L;
        captureFailures = 0L;
        encodedFrameCount = 0L;
        encodedDurationUs = 0L;
        encodedFps = 0.0;
        meteredExposureNs = 8_000_000L;
        meteredIso = 200;

        status("ARMING camera 0 · 1080p120 · autofocus · shutter target 1/500");

        Runnable open = () -> cameraHandler.post(this::openCameraForMetering);
        if (textureView.isAvailable()) {
            open.run();
        } else {
            activity.runOnUiThread(() -> textureView.setSurfaceTextureListener(
                    new TextureView.SurfaceTextureListener() {
                        @Override
                        public void onSurfaceTextureAvailable(
                                SurfaceTexture surface, int width, int height) {
                            textureView.setSurfaceTextureListener(null);
                            open.run();
                        }

                        @Override
                        public void onSurfaceTextureSizeChanged(
                                SurfaceTexture surface, int width, int height) {}

                        @Override
                        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
                            return false;
                        }

                        @Override
                        public void onSurfaceTextureUpdated(SurfaceTexture surface) {}
                    }));
        }
    }

    private void openCameraForMetering() {
        try {
            characteristics = cameraManager.getCameraCharacteristics(CAMERA_ID);
            validateHighSpeedMode(characteristics);

            SurfaceTexture st = textureView.getSurfaceTexture();
            if (st == null) {
                error("Preview surface is unavailable");
                return;
            }
            st.setDefaultBufferSize(WIDTH, HEIGHT);
            previewSurface = new Surface(st);

            cameraManager.openCamera(CAMERA_ID, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice camera) {
                    cameraDevice = camera;
                    createMeteringSession();
                }

                @Override
                public void onDisconnected(CameraDevice camera) {
                    camera.close();
                    if (!closing) error("Camera disconnected");
                }

                @Override
                public void onError(CameraDevice camera, int errorCode) {
                    camera.close();
                    if (!closing) error("Camera error " + errorCode + " (" + cameraErrorName(errorCode) + ")");
                }
            }, cameraHandler);
        } catch (Exception e) {
            error("Open camera failed: " + describe(e));
        }
    }

    private void validateHighSpeedMode(CameraCharacteristics c) {
        StreamConfigurationMap map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) {
            throw new IllegalStateException("No stream configuration map");
        }

        Size target = new Size(WIDTH, HEIGHT);
        boolean sizeFound = Arrays.asList(map.getHighSpeedVideoSizes()).contains(target);
        if (!sizeFound) {
            throw new IllegalStateException("1080p is not advertised as high-speed");
        }

        boolean fpsFound = false;
        for (Range<Integer> range : map.getHighSpeedVideoFpsRangesFor(target)) {
            if (range.getLower() == FPS && range.getUpper() == FPS) {
                fpsFound = true;
                break;
            }
        }
        if (!fpsFound) {
            throw new IllegalStateException("1080p120 fixed range is not advertised");
        }
    }

    private void createMeteringSession() {
        try {
            cameraDevice.createCaptureSession(
                    List.of(previewSurface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            previewSession = session;
                            try {
                                CaptureRequest.Builder builder =
                                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                                builder.addTarget(previewSurface);
                                builder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
                                builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON);
                                builder.set(CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
                                builder.set(CaptureRequest.CONTROL_AWB_MODE,
                                        CaptureRequest.CONTROL_AWB_MODE_AUTO);
                                disableStabilization(builder);

                                session.setRepeatingRequest(
                                        builder.build(), meteringCaptureCallback, cameraHandler);
                                status("Metering exposure/focus for 1.2 s...");
                                cameraHandler.postDelayed(
                                        CameraEngine.this::transitionToHighSpeed, 1200L);
                            } catch (Exception e) {
                                error("Metering request failed: " + describe(e));
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            error("Normal preview session configuration failed");
                        }
                    },
                    cameraHandler);
        } catch (Exception e) {
            error("Metering session failed: " + describe(e));
        }
    }

    private final CameraCaptureSession.CaptureCallback meteringCaptureCallback =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(
                        CameraCaptureSession session,
                        CaptureRequest request,
                        TotalCaptureResult result) {
                    Long exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
                    Integer iso = result.get(CaptureResult.SENSOR_SENSITIVITY);
                    if (exposure != null && exposure > 0) {
                        meteredExposureNs = exposure;
                    }
                    if (iso != null && iso > 0) {
                        meteredIso = iso;
                    }
                }
            };

    private void transitionToHighSpeed() {
        if (state != State.ARMING || cameraDevice == null) {
            return;
        }

        try {
            if (previewSession != null) {
                try {
                    previewSession.stopRepeating();
                } catch (Exception ignored) {}
                previewSession.close();
                previewSession = null;
            }

            chooseManualExposure();
            prepareRecorder();
            createHighSpeedSession();
        } catch (Exception e) {
            error("High-speed preparation failed: " + describe(e));
        }
    }

    private void chooseManualExposure() {
        Range<Long> exposureRange =
                characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE);
        Range<Integer> isoRange =
                characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE);

        long target = TARGET_EXPOSURE_NS;
        if (exposureRange != null) {
            target = Math.max(exposureRange.getLower(), Math.min(target, exposureRange.getUpper()));
        }

        double lightProduct = (double) meteredExposureNs * (double) meteredIso;
        int targetIso = (int) Math.round(lightProduct / Math.max(1.0, target));

        if (isoRange != null) {
            targetIso = Math.max(isoRange.getLower(), Math.min(targetIso, isoRange.getUpper()));
        } else {
            targetIso = Math.max(50, Math.min(targetIso, 6400));
        }

        actualExposureNs = target;
        actualIso = targetIso;

        status(String.format(Locale.US,
                "Metered %.2f ms ISO %d -> locking %.2f ms (~1/%d) ISO %d",
                meteredExposureNs / 1_000_000.0,
                meteredIso,
                actualExposureNs / 1_000_000.0,
                Math.round(1_000_000_000.0 / actualExposureNs),
                actualIso));
    }

    private void prepareRecorder() throws Exception {
        File base = activity.getExternalFilesDir(Environment.DIRECTORY_MOVIES);
        if (base == null) {
            base = activity.getFilesDir();
        }
        File dir = new File(base, "TKDPoomsae");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("Cannot create recording directory");
        }

        String device = Build.MODEL.replaceAll("[^A-Za-z0-9._-]", "_");
        videoFile = new File(dir, sessionId + "-" + device + "-cam0-1080p120.mp4");
        metadataFile = new File(dir, sessionId + "-" + device + "-cam0-1080p120.json");

        mediaRecorder = Build.VERSION.SDK_INT >= 31
                ? new MediaRecorder(activity)
                : new MediaRecorder();

        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        // Keep capture and playback cadence identical. Some Samsung builds otherwise
        // negotiate a ~30 fps output timeline even while the high-speed sensor session
        // is running at 120 fps.
        mediaRecorder.setCaptureRate(FPS);
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        mediaRecorder.setVideoSize(WIDTH, HEIGHT);
        mediaRecorder.setVideoFrameRate(FPS);
        mediaRecorder.setVideoEncodingBitRate(24_000_000);
        mediaRecorder.setOrientationHint(computeOrientationHint());
        mediaRecorder.setOutputFile(videoFile.getAbsolutePath());
        mediaRecorder.prepare();
        recorderSurface = mediaRecorder.getSurface();
    }

    private int computeOrientationHint() {
        Integer sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        if (sensorOrientation == null) {
            return 0;
        }

        int rotation = activity.getDisplay() == null
                ? Surface.ROTATION_0
                : activity.getDisplay().getRotation();
        int deviceDegrees;
        switch (rotation) {
            case Surface.ROTATION_90:
                deviceDegrees = 90;
                break;
            case Surface.ROTATION_180:
                deviceDegrees = 180;
                break;
            case Surface.ROTATION_270:
                deviceDegrees = 270;
                break;
            default:
                deviceDegrees = 0;
                break;
        }
        return (sensorOrientation - deviceDegrees + 360) % 360;
    }

    private void createHighSpeedSession() {
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
                                error("Configured session is not constrained high-speed");
                                session.close();
                                return;
                            }
                            highSpeedSession =
                                    (CameraConstrainedHighSpeedCaptureSession) session;
                            buildHighSpeedRequests();
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            error("High-speed session configuration failed");
                        }
                    },
                    cameraHandler);
        } catch (Exception e) {
            error("Create high-speed session failed: " + describe(e));
        }
    }

    private void buildHighSpeedRequests() {
        try {
            CaptureRequest.Builder recordBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            recordBuilder.addTarget(previewSurface);
            recordBuilder.addTarget(recorderSurface);
            applyHighSpeedControls(recordBuilder, new Range<>(FPS, FPS));

            highSpeedRequests =
                    highSpeedSession.createHighSpeedRequestList(recordBuilder.build());

            // Keep a live preview while the camera is ARMED/READY. A preview-only
            // high-speed request is allowed to run at a lower display cadence while
            // the encoder surface stays configured and ready for START.
            Range<Integer> previewRange = choosePreviewHighSpeedRange();
            CaptureRequest.Builder previewBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            previewBuilder.addTarget(previewSurface);
            applyHighSpeedControls(previewBuilder, previewRange);

            highSpeedPreviewRequests =
                    highSpeedSession.createHighSpeedRequestList(previewBuilder.build());

            highSpeedSession.setRepeatingBurst(
                    highSpeedPreviewRequests, previewCaptureCallback, cameraHandler);

            state = State.READY;
            String detail = String.format(Locale.US,
                    "READY · live preview · cam0 1920x1080@120 · AF continuous · shutter ~1/%d · ISO %d",
                    Math.round(1_000_000_000.0 / actualExposureNs),
                    actualIso);
            status(detail);
            activity.runOnUiThread(() -> listener.onCameraReady(detail));
        } catch (Exception e) {
            error("Build high-speed request failed: " + describe(e));
        }
    }

    private void applyHighSpeedControls(
            CaptureRequest.Builder builder, Range<Integer> fpsRange) {
        builder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
        builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange);
        builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF);
        builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, actualExposureNs);
        builder.set(CaptureRequest.SENSOR_SENSITIVITY, actualIso);
        builder.set(CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
        builder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO);
        disableStabilization(builder);
    }

    private Range<Integer> choosePreviewHighSpeedRange() {
        try {
            StreamConfigurationMap map =
                    characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map != null) {
                Size target = new Size(WIDTH, HEIGHT);
                Range<Integer>[] ranges = map.getHighSpeedVideoFpsRangesFor(target);
                Range<Integer> best = null;
                for (Range<Integer> range : ranges) {
                    if (range.getUpper() == FPS && range.getLower() < FPS) {
                        if (best == null || range.getLower() < best.getLower()) {
                            best = range;
                        }
                    }
                }
                if (best != null) return best;
            }
        } catch (Exception ignored) {
        }
        return new Range<>(FPS, FPS);
    }

    private final CameraCaptureSession.CaptureCallback previewCaptureCallback =
            new CameraCaptureSession.CaptureCallback() {};

    private void disableStabilization(CaptureRequest.Builder builder) {
        int[] videoModes =
                characteristics == null ? null :
                        characteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES);
        if (contains(videoModes, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)) {
            builder.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF);
        }

        int[] oisModes =
                characteristics == null ? null :
                        characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
        if (contains(oisModes, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)) {
            builder.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF);
        }
    }

    private static boolean contains(int[] values, int target) {
        if (values == null) return false;
        for (int value : values) {
            if (value == target) return true;
        }
        return false;
    }

    private void scheduleStartInternal(long targetNs) {
        if (state != State.READY || highSpeedSession == null || mediaRecorder == null) {
            error("START requested while camera is not READY");
            return;
        }

        scheduledStartNs = targetNs;
        long now = SystemClock.elapsedRealtimeNanos();
        long remainingNs = targetNs - now;
        if (remainingNs <= 0) {
            startRecordingNow();
            return;
        }

        long delayMs = Math.max(0L, remainingNs / 1_000_000L - 3L);
        status(String.format(Locale.US,
                "START scheduled in %.1f ms", remainingNs / 1_000_000.0));

        cameraHandler.postDelayed(() -> {
            while (SystemClock.elapsedRealtimeNanos() < scheduledStartNs) {
                Thread.onSpinWait();
            }
            startRecordingNow();
        }, delayMs);
    }

    private void startRecordingNow() {
        if (state != State.READY) {
            return;
        }

        try {
            startCallNs = SystemClock.elapsedRealtimeNanos();
            firstSensorTimestampNs = -1L;
            lastUniqueSensorTimestampNs = -1L;
            uniqueSensorFrames = 0L;
            captureFailures = 0L;

            // Match the sequence that was verified on the S21 probe: switch the
            // constrained session to the fixed 120 fps record burst first, then start
            // MediaRecorder. Starting the recorder while the lower-rate armed preview
            // burst is still active can make Samsung negotiate a ~30 fps MP4 timeline.
            try {
                highSpeedSession.stopRepeating();
            } catch (Exception ignored) {
            }
            highSpeedSession.setRepeatingBurst(
                    highSpeedRequests, recordingCaptureCallback, cameraHandler);
            mediaRecorder.start();
            recorderStarted = true;
            state = State.RECORDING;

            long deltaNs = scheduledStartNs > 0 ? startCallNs - scheduledStartNs : 0L;
            status(String.format(Locale.US,
                    "RECORDING · local start delta %.3f ms",
                    deltaNs / 1_000_000.0));
            activity.runOnUiThread(() -> listener.onCameraStarted(startCallNs));
        } catch (Exception e) {
            error("Start recording failed: " + describe(e));
        }
    }

    private final CameraCaptureSession.CaptureCallback recordingCaptureCallback =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(
                        CameraCaptureSession session,
                        CaptureRequest request,
                        TotalCaptureResult result) {
                    Long ts = result.get(CaptureResult.SENSOR_TIMESTAMP);
                    if (ts == null) return;

                    if (ts != lastUniqueSensorTimestampNs) {
                        if (firstSensorTimestampNs < 0) {
                            firstSensorTimestampNs = ts;
                        }
                        lastUniqueSensorTimestampNs = ts;
                        uniqueSensorFrames++;
                    }
                }

                @Override
                public void onCaptureFailed(
                        CameraCaptureSession session,
                        CaptureRequest request,
                        CaptureFailure failure) {
                    captureFailures++;
                }
            };

    private void stopInternal() {
        if (state == State.IDLE) {
            return;
        }

        state = State.STOPPING;
        closing = true;
        stopCallNs = SystemClock.elapsedRealtimeNanos();

        try {
            if (highSpeedSession != null) {
                try {
                    highSpeedSession.stopRepeating();
                } catch (Exception ignored) {}
                try {
                    highSpeedSession.abortCaptures();
                } catch (Exception ignored) {}
            }

            if (recorderStarted && mediaRecorder != null) {
                mediaRecorder.stop();
            }
        } catch (Exception e) {
            status("Recorder stop warning: " + describe(e));
        } finally {
            recorderStarted = false;
        }

        verifyRecordedVideo();
        writeMetadata();
        String videoPath = publishVideoToGallery();
        if (videoPath == null && videoFile != null) {
            videoPath = videoFile.getAbsolutePath();
        }
        String jsonPath = metadataFile == null ? null : metadataFile.getAbsolutePath();
        final String finalVideoPath = videoPath;
        final String finalJsonPath = jsonPath;

        closeAll();
        state = State.IDLE;
        closing = false;

        status("STOPPED · " + (finalVideoPath == null ? "no video" : finalVideoPath));
        activity.runOnUiThread(() -> listener.onCameraStopped(finalVideoPath, finalJsonPath));
    }

    private void verifyRecordedVideo() {
        encodedFrameCount = 0L;
        encodedDurationUs = 0L;
        encodedFps = 0.0;

        if (videoFile == null || !videoFile.exists() || videoFile.length() == 0) {
            status("MP4 verification skipped: no file");
            return;
        }

        MediaExtractor extractor = new MediaExtractor();
        try {
            extractor.setDataSource(videoFile.getAbsolutePath());

            int videoTrack = -1;
            for (int i = 0; i < extractor.getTrackCount(); i++) {
                MediaFormat format = extractor.getTrackFormat(i);
                String mime = format.getString(MediaFormat.KEY_MIME);
                if (mime != null && mime.startsWith("video/")) {
                    videoTrack = i;
                    break;
                }
            }
            if (videoTrack < 0) {
                status("MP4 verification: no video track");
                return;
            }

            extractor.selectTrack(videoTrack);
            long firstUs = -1L;
            long lastUs = -1L;
            long frames = 0L;

            while (true) {
                long sampleUs = extractor.getSampleTime();
                if (sampleUs < 0) break;
                if (firstUs < 0) firstUs = sampleUs;
                lastUs = sampleUs;
                frames++;
                if (!extractor.advance()) break;
            }

            encodedFrameCount = frames;
            if (frames > 1 && lastUs > firstUs) {
                encodedDurationUs = lastUs - firstUs;
                encodedFps = (frames - 1) * 1_000_000.0 / encodedDurationUs;
            }

            status(String.format(Locale.US,
                    "MP4 verified: %d frames · %.2f fps",
                    encodedFrameCount, encodedFps));
        } catch (Exception e) {
            status("MP4 verification warning: " + describe(e));
        } finally {
            try { extractor.release(); } catch (Exception ignored) {}
        }
    }

    private String publishVideoToGallery() {
        if (videoFile == null || !videoFile.exists() || videoFile.length() == 0) {
            status("Gallery publish skipped: no video file");
            return null;
        }

        if (Build.VERSION.SDK_INT < 29) {
            status("Gallery publish uses app path on Android < 10");
            return videoFile.getAbsolutePath();
        }

        ContentResolver resolver = activity.getContentResolver();
        ContentValues values = new ContentValues();
        values.put(MediaStore.Video.Media.DISPLAY_NAME, videoFile.getName());
        values.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4");
        values.put(MediaStore.Video.Media.RELATIVE_PATH,
                Environment.DIRECTORY_MOVIES + "/TKDPoomsae");
        values.put(MediaStore.Video.Media.IS_PENDING, 1);

        Uri uri = null;
        try {
            uri = resolver.insert(
                    MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values);
            if (uri == null) {
                status("Gallery publish failed: MediaStore insert returned null");
                return null;
            }

            try (InputStream in = new FileInputStream(videoFile);
                 OutputStream out = resolver.openOutputStream(uri, "w")) {
                if (out == null) {
                    throw new IllegalStateException("MediaStore output stream is null");
                }
                byte[] buffer = new byte[1024 * 1024];
                int read;
                while ((read = in.read(buffer)) >= 0) {
                    out.write(buffer, 0, read);
                }
                out.flush();
            }

            ContentValues done = new ContentValues();
            done.put(MediaStore.Video.Media.IS_PENDING, 0);
            resolver.update(uri, done, null, null);

            status("Published to Gallery: Movies/TKDPoomsae/" + videoFile.getName());
            return uri.toString();
        } catch (Exception e) {
            status("Gallery publish warning: " + describe(e));
            if (uri != null) {
                try { resolver.delete(uri, null, null); } catch (Exception ignored) {}
            }
            return null;
        }
    }

    private void writeMetadata() {
        if (metadataFile == null) return;

        try {
            JSONObject j = new JSONObject();
            j.put("session_id", sessionId);
            j.put("manufacturer", Build.MANUFACTURER);
            j.put("model", Build.MODEL);
            j.put("camera_id", CAMERA_ID);
            j.put("width", WIDTH);
            j.put("height", HEIGHT);
            j.put("fps_requested", FPS);
            j.put("shutter_ns", actualExposureNs);
            j.put("iso", actualIso);
            j.put("metered_exposure_ns", meteredExposureNs);
            j.put("metered_iso", meteredIso);
            j.put("scheduled_start_elapsed_ns", scheduledStartNs);
            j.put("start_call_elapsed_ns", startCallNs);
            j.put("stop_call_elapsed_ns", stopCallNs);
            j.put("first_sensor_timestamp_ns", firstSensorTimestampNs);
            j.put("last_sensor_timestamp_ns", lastUniqueSensorTimestampNs);
            j.put("unique_sensor_frames", uniqueSensorFrames);
            j.put("capture_failures", captureFailures);
            j.put("encoded_frame_count", encodedFrameCount);
            j.put("encoded_duration_us", encodedDurationUs);
            j.put("encoded_fps", encodedFps);
            j.put("video_path", videoFile == null ? JSONObject.NULL : videoFile.getAbsolutePath());

            if (uniqueSensorFrames > 1
                    && firstSensorTimestampNs > 0
                    && lastUniqueSensorTimestampNs > firstSensorTimestampNs) {
                double seconds =
                        (lastUniqueSensorTimestampNs - firstSensorTimestampNs) / 1_000_000_000.0;
                j.put("sensor_fps_estimate", (uniqueSensorFrames - 1) / seconds);
            }

            try (FileWriter writer = new FileWriter(metadataFile)) {
                writer.write(j.toString(2));
            }
        } catch (Exception e) {
            status("Metadata write warning: " + describe(e));
        }
    }

    private void closeAll() {
        try {
            if (previewSession != null) previewSession.close();
        } catch (Exception ignored) {}
        previewSession = null;

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

        highSpeedRequests = null;
        highSpeedPreviewRequests = null;
    }

    private void status(String message) {
        activity.runOnUiThread(() -> listener.onCameraStatus(message));
    }

    private void error(String message) {
        state = State.ERROR;
        status("ERROR: " + message);
        activity.runOnUiThread(() -> listener.onCameraError(message));
        closeAll();
        closing = false;
    }

    private static String describe(Exception e) {
        String msg = e.getMessage();
        return e.getClass().getSimpleName() + (msg == null ? "" : ": " + msg);
    }

    private static String sanitizeSessionId(String id) {
        if (id == null || id.isBlank()) {
            return "session-" + System.currentTimeMillis();
        }
        return id.replaceAll("[^A-Za-z0-9._-]", "_");
    }

    private static String cameraErrorName(int error) {
        switch (error) {
            case CameraDevice.StateCallback.ERROR_CAMERA_IN_USE:
                return "ERROR_CAMERA_IN_USE";
            case CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE:
                return "ERROR_MAX_CAMERAS_IN_USE";
            case CameraDevice.StateCallback.ERROR_CAMERA_DISABLED:
                return "ERROR_CAMERA_DISABLED";
            case CameraDevice.StateCallback.ERROR_CAMERA_DEVICE:
                return "ERROR_CAMERA_DEVICE";
            case CameraDevice.StateCallback.ERROR_CAMERA_SERVICE:
                return "ERROR_CAMERA_SERVICE";
            default:
                return "UNKNOWN";
        }
    }
}
