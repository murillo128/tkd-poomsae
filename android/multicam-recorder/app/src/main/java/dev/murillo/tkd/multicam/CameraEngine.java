package dev.murillo.tkd.multicam;

import android.Manifest;
import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
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
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Range;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;

import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
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

    private String sessionId;
    private File videoFile;
    private File metadataFile;

    private volatile long meteredExposureNs = 0L;
    private volatile int meteredIso = 0;

    private volatile long recorderStartElapsedNs = -1L;
    private volatile long scheduledStartNs = -1L;
    private volatile long officialStartElapsedNs = -1L;
    private volatile long stopCallNs = -1L;

    private volatile long firstSensorTimestampNs = -1L;
    private volatile long lastUniqueSensorTimestampNs = -1L;
    private volatile long uniqueSensorFrames = 0L;
    private volatile long captureFailures = 0L;

    private volatile long encodedFrameCount = 0L;
    private volatile long encodedDurationUs = 0L;
    private volatile double encodedFps = 0.0;

    private volatile boolean recorderStarted = false;
    private volatile boolean officialRecordingStarted = false;
    private volatile boolean closing = false;

    public CameraEngine(
            Activity activity,
            TextureView textureView,
            Listener listener) {

        this.activity = activity;
        this.textureView = textureView;
        this.listener = listener;
        this.cameraManager =
                (CameraManager)
                        activity.getSystemService(
                                Context.CAMERA_SERVICE);

        this.cameraThread =
                new HandlerThread(
                        "TkdMultiCamCamera");

        this.cameraThread.start();

        this.cameraHandler =
                new Handler(
                        cameraThread.getLooper());
    }

    public State getState() {
        return state;
    }

    public double getLastEncodedFps() {
        return encodedFps;
    }

    public long getLastEncodedFrameCount() {
        return encodedFrameCount;
    }

    public void arm(String newSessionId) {
        cameraHandler.post(
                () -> armInternal(newSessionId));
    }

    public void startAt(
            long localElapsedRealtimeNs) {

        cameraHandler.post(
                () -> scheduleOfficialStart(
                        localElapsedRealtimeNs));
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

    private void armInternal(
            String newSessionId) {

        if (state != State.IDLE
                && state != State.ERROR) {
            status(
                    "Ignoring ARM while state="
                            + state);
            return;
        }

        if (activity.checkSelfPermission(
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {

            error(
                    "Camera permission is not granted");
            return;
        }

        closeAll();

        state = State.ARMING;
        closing = false;

        sessionId =
                sanitizeSessionId(newSessionId);

        recorderStartElapsedNs = -1L;
        scheduledStartNs = -1L;
        officialStartElapsedNs = -1L;
        stopCallNs = -1L;

        firstSensorTimestampNs = -1L;
        lastUniqueSensorTimestampNs = -1L;
        uniqueSensorFrames = 0L;
        captureFailures = 0L;

        encodedFrameCount = 0L;
        encodedDurationUs = 0L;
        encodedFps = 0.0;

        recorderStarted = false;
        officialRecordingStarted = false;

        status(
                "ARMING · probing exposure/focus");

        Runnable open =
                () -> cameraHandler.post(
                        this::openCameraForMetering);

        if (textureView.isAvailable()) {
            open.run();
        } else {
            activity.runOnUiThread(() ->
                    textureView.setSurfaceTextureListener(
                            new TextureView.SurfaceTextureListener() {
                                @Override
                                public void onSurfaceTextureAvailable(
                                        SurfaceTexture surface,
                                        int width,
                                        int height) {

                                    textureView.setSurfaceTextureListener(
                                            null);

                                    open.run();
                                }

                                @Override
                                public void onSurfaceTextureSizeChanged(
                                        SurfaceTexture surface,
                                        int width,
                                        int height) {}

                                @Override
                                public boolean onSurfaceTextureDestroyed(
                                        SurfaceTexture surface) {
                                    return false;
                                }

                                @Override
                                public void onSurfaceTextureUpdated(
                                        SurfaceTexture surface) {}
                            }));
        }
    }

    private void openCameraForMetering() {
        try {
            characteristics =
                    cameraManager
                            .getCameraCharacteristics(
                                    CAMERA_ID);

            validateHighSpeedMode(
                    characteristics);

            SurfaceTexture st =
                    textureView
                            .getSurfaceTexture();

            if (st == null) {
                error(
                        "Preview surface unavailable");
                return;
            }

            st.setDefaultBufferSize(
                    WIDTH,
                    HEIGHT);

            previewSurface =
                    new Surface(st);

            cameraManager.openCamera(
                    CAMERA_ID,
                    new CameraDevice.StateCallback() {
                        @Override
                        public void onOpened(
                                CameraDevice camera) {
                            cameraDevice = camera;
                            createMeteringSession();
                        }

                        @Override
                        public void onDisconnected(
                                CameraDevice camera) {
                            camera.close();

                            if (!closing) {
                                error(
                                        "Camera disconnected");
                            }
                        }

                        @Override
                        public void onError(
                                CameraDevice camera,
                                int errorCode) {
                            camera.close();

                            if (!closing) {
                                error(
                                        "Camera error "
                                                + errorCode
                                                + " ("
                                                + cameraErrorName(
                                                        errorCode)
                                                + ")");
                            }
                        }
                    },
                    cameraHandler);
        } catch (Exception e) {
            error(
                    "Open camera failed: "
                            + describe(e));
        }
    }

    private void validateHighSpeedMode(
            CameraCharacteristics c) {

        StreamConfigurationMap map =
                c.get(
                        CameraCharacteristics
                                .SCALER_STREAM_CONFIGURATION_MAP);

        if (map == null) {
            throw new IllegalStateException(
                    "No stream configuration map");
        }

        Size target =
                new Size(
                        WIDTH,
                        HEIGHT);

        if (!Arrays.asList(
                map.getHighSpeedVideoSizes())
                .contains(target)) {

            throw new IllegalStateException(
                    "1080p high-speed not advertised");
        }

        boolean fixed120 = false;

        for (Range<Integer> range
                : map.getHighSpeedVideoFpsRangesFor(
                        target)) {

            if (range.getLower() == FPS
                    && range.getUpper() == FPS) {
                fixed120 = true;
                break;
            }
        }

        if (!fixed120) {
            throw new IllegalStateException(
                    "1080p120 fixed range not advertised");
        }
    }

    private void createMeteringSession() {
        try {
            cameraDevice.createCaptureSession(
                    List.of(previewSurface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(
                                CameraCaptureSession session) {

                            previewSession = session;

                            try {
                                CaptureRequest.Builder builder =
                                        cameraDevice
                                                .createCaptureRequest(
                                                        CameraDevice
                                                                .TEMPLATE_PREVIEW);

                                builder.addTarget(
                                        previewSurface);

                                builder.set(
                                        CaptureRequest.CONTROL_MODE,
                                        CaptureRequest
                                                .CONTROL_MODE_AUTO);

                                builder.set(
                                        CaptureRequest.CONTROL_AE_MODE,
                                        CaptureRequest
                                                .CONTROL_AE_MODE_ON);

                                builder.set(
                                        CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest
                                                .CONTROL_AF_MODE_CONTINUOUS_VIDEO);

                                builder.set(
                                        CaptureRequest.CONTROL_AWB_MODE,
                                        CaptureRequest
                                                .CONTROL_AWB_MODE_AUTO);

                                session.setRepeatingRequest(
                                        builder.build(),
                                        meteringCallback,
                                        cameraHandler);

                                cameraHandler.postDelayed(
                                        CameraEngine.this
                                                ::enterProvenHighSpeedPath,
                                        1000L);
                            } catch (Exception e) {
                                error(
                                        "Metering failed: "
                                                + describe(e));
                            }
                        }

                        @Override
                        public void onConfigureFailed(
                                CameraCaptureSession session) {

                            error(
                                    "Metering session failed");
                        }
                    },
                    cameraHandler);
        } catch (Exception e) {
            error(
                    "Metering session failed: "
                            + describe(e));
        }
    }

    private final CameraCaptureSession.CaptureCallback
            meteringCallback =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(
                        CameraCaptureSession session,
                        CaptureRequest request,
                        TotalCaptureResult result) {

                    Long exposure =
                            result.get(
                                    CaptureResult
                                            .SENSOR_EXPOSURE_TIME);

                    Integer iso =
                            result.get(
                                    CaptureResult
                                            .SENSOR_SENSITIVITY);

                    if (exposure != null) {
                        meteredExposureNs =
                                exposure;
                    }

                    if (iso != null) {
                        meteredIso = iso;
                    }
                }
            };

    private void enterProvenHighSpeedPath() {
        if (state != State.ARMING
                || cameraDevice == null) {
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

            prepareRecorderExactlyLikeProbe();
            createHighSpeedSession();
        } catch (Exception e) {
            error(
                    "High-speed setup failed: "
                            + describe(e));
        }
    }

    private void prepareRecorderExactlyLikeProbe()
            throws Exception {

        File base =
                activity.getExternalFilesDir(
                        Environment.DIRECTORY_MOVIES);

        if (base == null) {
            base = activity.getFilesDir();
        }

        File dir =
                new File(
                        base,
                        "TKDPoomsae");

        if (!dir.exists()
                && !dir.mkdirs()) {
            throw new IllegalStateException(
                    "Cannot create output directory");
        }

        String device =
                Build.MODEL.replaceAll(
                        "[^A-Za-z0-9._-]",
                        "_");

        videoFile =
                new File(
                        dir,
                        sessionId
                                + "-"
                                + device
                                + "-cam0-1080p120.mp4");

        metadataFile =
                new File(
                        dir,
                        sessionId
                                + "-"
                                + device
                                + "-cam0-1080p120.json");

        mediaRecorder =
                Build.VERSION.SDK_INT >= 31
                        ? new MediaRecorder(activity)
                        : new MediaRecorder();

        mediaRecorder.setVideoSource(
                MediaRecorder.VideoSource.SURFACE);

        mediaRecorder.setOutputFormat(
                MediaRecorder.OutputFormat.MPEG_4);

        mediaRecorder.setVideoEncoder(
                MediaRecorder.VideoEncoder.H264);

        mediaRecorder.setVideoSize(
                WIDTH,
                HEIGHT);

        mediaRecorder.setVideoFrameRate(
                FPS);

        mediaRecorder.setVideoEncodingBitRate(
                24_000_000);

        mediaRecorder.setOrientationHint(
                computeOrientationHint());

        mediaRecorder.setOutputFile(
                videoFile.getAbsolutePath());

        mediaRecorder.prepare();

        recorderSurface =
                mediaRecorder.getSurface();
    }

    private int computeOrientationHint() {
        Integer sensorOrientation =
                characteristics.get(
                        CameraCharacteristics
                                .SENSOR_ORIENTATION);

        if (sensorOrientation == null) {
            return 0;
        }

        int rotation =
                activity.getDisplay() == null
                        ? Surface.ROTATION_0
                        : activity
                                .getDisplay()
                                .getRotation();

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

        return (sensorOrientation
                - deviceDegrees
                + 360) % 360;
    }

    private void createHighSpeedSession() {
        try {
            List<Surface> outputs =
                    new ArrayList<>();

            outputs.add(
                    previewSurface);

            outputs.add(
                    recorderSurface);

            cameraDevice
                    .createConstrainedHighSpeedCaptureSession(
                            outputs,
                            new CameraCaptureSession.StateCallback() {
                                @Override
                                public void onConfigured(
                                        CameraCaptureSession session) {

                                    if (!(session
                                            instanceof
                                            CameraConstrainedHighSpeedCaptureSession)) {

                                        error(
                                                "Not a constrained high-speed session");

                                        session.close();
                                        return;
                                    }

                                    highSpeedSession =
                                            (CameraConstrainedHighSpeedCaptureSession)
                                                    session;

                                    startHighSpeedPreroll();
                                }

                                @Override
                                public void onConfigureFailed(
                                        CameraCaptureSession session) {

                                    error(
                                            "High-speed session configuration failed");
                                }
                            },
                            cameraHandler);
        } catch (Exception e) {
            error(
                    "Create high-speed session failed: "
                            + describe(e));
        }
    }

    private void startHighSpeedPreroll() {
        try {
            CaptureRequest.Builder builder =
                    cameraDevice
                            .createCaptureRequest(
                                    CameraDevice
                                            .TEMPLATE_RECORD);

            builder.addTarget(
                    previewSurface);

            builder.addTarget(
                    recorderSurface);

            // This intentionally mirrors the probe that measured 116-120 fps
            // on this exact S21. Keep AE/AWB automatic and only force the
            // constrained fixed high-speed FPS range.
            builder.set(
                    CaptureRequest.CONTROL_MODE,
                    CaptureRequest.CONTROL_MODE_AUTO);

            builder.set(
                    CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                    new Range<>(FPS, FPS));

            builder.set(
                    CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest
                            .CONTROL_AF_MODE_CONTINUOUS_VIDEO);

            highSpeedRequests =
                    highSpeedSession
                            .createHighSpeedRequestList(
                                    builder.build());

            firstSensorTimestampNs = -1L;
            lastUniqueSensorTimestampNs = -1L;
            uniqueSensorFrames = 0L;
            captureFailures = 0L;

            // Exact ordering from the successful probe:
            // high-speed burst first, then MediaRecorder.start().
            highSpeedSession.setRepeatingBurst(
                    highSpeedRequests,
                    recordingCallback,
                    cameraHandler);

            mediaRecorder.start();

            recorderStarted = true;
            recorderStartElapsedNs =
                    SystemClock.elapsedRealtimeNanos();

            state = State.READY;

            String details =
                    String.format(
                            Locale.US,
                            "FHD120 buffer active · AF · AE auto · ISO %d",
                            meteredIso);

            status(
                    "READY · " + details);

            activity.runOnUiThread(
                    () -> listener.onCameraReady(
                            details));
        } catch (Exception e) {
            error(
                    "High-speed pre-roll failed: "
                            + describe(e));
        }
    }

    private void scheduleOfficialStart(
            long targetNs) {

        if (state != State.READY
                || !recorderStarted) {

            error(
                    "START requested while camera is not READY");
            return;
        }

        scheduledStartNs = targetNs;

        long remainingNs =
                targetNs
                        - SystemClock
                                .elapsedRealtimeNanos();

        if (remainingNs <= 0) {
            markOfficialStart();
            return;
        }

        long delayMs =
                Math.max(
                        0L,
                        remainingNs
                                / 1_000_000L
                                - 3L);

        status(
                String.format(
                        Locale.US,
                        "START in %.0f ms · 120fps buffer running",
                        remainingNs / 1_000_000.0));

        cameraHandler.postDelayed(() -> {
            while (SystemClock
                    .elapsedRealtimeNanos()
                    < scheduledStartNs) {
                Thread.onSpinWait();
            }

            markOfficialStart();
        }, delayMs);
    }

    private void markOfficialStart() {
        if (state != State.READY) {
            return;
        }

        officialStartElapsedNs =
                SystemClock
                        .elapsedRealtimeNanos();

        officialRecordingStarted = true;
        state = State.RECORDING;

        long preRollNs =
                recorderStartElapsedNs > 0
                        ? officialStartElapsedNs
                                - recorderStartElapsedNs
                        : 0L;

        status(
                String.format(
                        Locale.US,
                        "RECORDING · FHD120 · preroll %.2f s",
                        preRollNs
                                / 1_000_000_000.0));

        activity.runOnUiThread(
                () -> listener.onCameraStarted(
                        officialStartElapsedNs));
    }

    private final CameraCaptureSession.CaptureCallback
            recordingCallback =
            new CameraCaptureSession.CaptureCallback() {

                @Override
                public void onCaptureCompleted(
                        CameraCaptureSession session,
                        CaptureRequest request,
                        TotalCaptureResult result) {

                    Long ts =
                            result.get(
                                    CaptureResult
                                            .SENSOR_TIMESTAMP);

                    if (ts == null) return;

                    if (ts
                            != lastUniqueSensorTimestampNs) {

                        if (firstSensorTimestampNs < 0) {
                            firstSensorTimestampNs =
                                    ts;
                        }

                        lastUniqueSensorTimestampNs =
                                ts;

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

        stopCallNs =
                SystemClock.elapsedRealtimeNanos();

        try {
            if (highSpeedSession != null) {
                try {
                    highSpeedSession
                            .stopRepeating();
                } catch (Exception ignored) {}

                try {
                    highSpeedSession
                            .abortCaptures();
                } catch (Exception ignored) {}
            }

            if (recorderStarted
                    && mediaRecorder != null) {

                mediaRecorder.stop();
            }
        } catch (Exception e) {
            status(
                    "Recorder stop warning: "
                            + describe(e));
        } finally {
            recorderStarted = false;
        }

        verifyRecordedVideo();
        writeMetadata();

        String videoPath =
                publishVideoToGallery();

        if (videoPath == null
                && videoFile != null) {

            videoPath =
                    videoFile.getAbsolutePath();
        }

        String jsonPath =
                metadataFile == null
                        ? null
                        : metadataFile
                                .getAbsolutePath();

        final String finalVideoPath =
                videoPath;

        final String finalJsonPath =
                jsonPath;

        closeAll();

        state = State.IDLE;
        closing = false;

        status(
                String.format(
                        Locale.US,
                        "SAVED · %.1f fps · %d frames",
                        encodedFps,
                        encodedFrameCount));

        activity.runOnUiThread(
                () -> listener.onCameraStopped(
                        finalVideoPath,
                        finalJsonPath));
    }

    private void verifyRecordedVideo() {
        encodedFrameCount = 0L;
        encodedDurationUs = 0L;
        encodedFps = 0.0;

        if (videoFile == null
                || !videoFile.exists()
                || videoFile.length() == 0) {
            return;
        }

        MediaExtractor extractor =
                new MediaExtractor();

        try {
            extractor.setDataSource(
                    videoFile.getAbsolutePath());

            int videoTrack = -1;

            for (int i = 0;
                 i < extractor.getTrackCount();
                 i++) {

                MediaFormat format =
                        extractor.getTrackFormat(i);

                String mime =
                        format.getString(
                                MediaFormat.KEY_MIME);

                if (mime != null
                        && mime.startsWith(
                                "video/")) {

                    videoTrack = i;
                    break;
                }
            }

            if (videoTrack < 0) {
                return;
            }

            extractor.selectTrack(
                    videoTrack);

            long firstUs = -1L;
            long lastUs = -1L;

            while (true) {
                long sampleUs =
                        extractor.getSampleTime();

                if (sampleUs < 0) {
                    break;
                }

                if (firstUs < 0) {
                    firstUs = sampleUs;
                }

                lastUs = sampleUs;
                encodedFrameCount++;

                if (!extractor.advance()) {
                    break;
                }
            }

            if (encodedFrameCount > 1
                    && lastUs > firstUs) {

                encodedDurationUs =
                        lastUs - firstUs;

                encodedFps =
                        (encodedFrameCount - 1)
                                * 1_000_000.0
                                / encodedDurationUs;
            }
        } catch (Exception e) {
            status(
                    "MP4 verify warning: "
                            + describe(e));
        } finally {
            try {
                extractor.release();
            } catch (Exception ignored) {}
        }
    }

    private void writeMetadata() {
        if (metadataFile == null) return;

        try {
            JSONObject j =
                    new JSONObject();

            j.put("session_id", sessionId);
            j.put("manufacturer", Build.MANUFACTURER);
            j.put("model", Build.MODEL);
            j.put("camera_id", CAMERA_ID);
            j.put("width", WIDTH);
            j.put("height", HEIGHT);
            j.put("fps_requested", FPS);
            j.put(
                    "capture_backend",
                    "MediaRecorder proven high-speed path");
            j.put(
                    "exposure_mode",
                    "AE_AUTO");
            j.put(
                    "metered_exposure_ns",
                    meteredExposureNs);
            j.put(
                    "metered_iso",
                    meteredIso);
            j.put(
                    "recorder_start_elapsed_ns",
                    recorderStartElapsedNs);
            j.put(
                    "scheduled_start_elapsed_ns",
                    scheduledStartNs);
            j.put(
                    "official_start_elapsed_ns",
                    officialStartElapsedNs);
            j.put(
                    "stop_elapsed_ns",
                    stopCallNs);

            long preRollNs =
                    recorderStartElapsedNs > 0
                            && officialStartElapsedNs > 0
                            ? officialStartElapsedNs
                                    - recorderStartElapsedNs
                            : 0L;

            j.put(
                    "pre_roll_ns",
                    preRollNs);

            j.put(
                    "first_sensor_timestamp_ns",
                    firstSensorTimestampNs);

            j.put(
                    "last_sensor_timestamp_ns",
                    lastUniqueSensorTimestampNs);

            j.put(
                    "unique_sensor_frames",
                    uniqueSensorFrames);

            j.put(
                    "capture_failures",
                    captureFailures);

            j.put(
                    "encoded_frame_count",
                    encodedFrameCount);

            j.put(
                    "encoded_duration_us",
                    encodedDurationUs);

            j.put(
                    "encoded_fps",
                    encodedFps);

            if (uniqueSensorFrames > 1
                    && firstSensorTimestampNs > 0
                    && lastUniqueSensorTimestampNs
                            > firstSensorTimestampNs) {

                double seconds =
                        (lastUniqueSensorTimestampNs
                                - firstSensorTimestampNs)
                                / 1_000_000_000.0;

                j.put(
                        "sensor_fps_estimate",
                        (uniqueSensorFrames - 1)
                                / seconds);
            }

            try (FileWriter writer =
                         new FileWriter(
                                 metadataFile)) {

                writer.write(
                        j.toString(2));
            }
        } catch (Exception e) {
            status(
                    "Metadata warning: "
                            + describe(e));
        }
    }

    private String publishVideoToGallery() {
        if (videoFile == null
                || !videoFile.exists()
                || videoFile.length() == 0) {
            return null;
        }

        if (Build.VERSION.SDK_INT < 29) {
            return videoFile
                    .getAbsolutePath();
        }

        ContentResolver resolver =
                activity
                        .getContentResolver();

        ContentValues values =
                new ContentValues();

        values.put(
                MediaStore.Video.Media
                        .DISPLAY_NAME,
                videoFile.getName());

        values.put(
                MediaStore.Video.Media
                        .MIME_TYPE,
                "video/mp4");

        values.put(
                MediaStore.Video.Media
                        .RELATIVE_PATH,
                Environment
                        .DIRECTORY_MOVIES
                        + "/TKDPoomsae");

        values.put(
                MediaStore.Video.Media
                        .IS_PENDING,
                1);

        Uri uri = null;

        try {
            uri =
                    resolver.insert(
                            MediaStore.Video.Media
                                    .EXTERNAL_CONTENT_URI,
                            values);

            if (uri == null) {
                return null;
            }

            try (InputStream in =
                         new FileInputStream(
                                 videoFile);
                 OutputStream out =
                         resolver.openOutputStream(
                                 uri,
                                 "w")) {

                if (out == null) {
                    return null;
                }

                byte[] buffer =
                        new byte[1024 * 1024];

                int read;

                while ((read = in.read(
                        buffer)) >= 0) {

                    out.write(
                            buffer,
                            0,
                            read);
                }

                out.flush();
            }

            ContentValues done =
                    new ContentValues();

            done.put(
                    MediaStore.Video.Media
                            .IS_PENDING,
                    0);

            resolver.update(
                    uri,
                    done,
                    null,
                    null);

            return uri.toString();
        } catch (Exception e) {
            status(
                    "Gallery warning: "
                            + describe(e));

            if (uri != null) {
                try {
                    resolver.delete(
                            uri,
                            null,
                            null);
                } catch (Exception ignored) {}
            }

            return null;
        }
    }

    private void closeAll() {
        try {
            if (previewSession != null) {
                previewSession.close();
            }
        } catch (Exception ignored) {}

        previewSession = null;

        try {
            if (highSpeedSession != null) {
                highSpeedSession.close();
            }
        } catch (Exception ignored) {}

        highSpeedSession = null;

        try {
            if (cameraDevice != null) {
                cameraDevice.close();
            }
        } catch (Exception ignored) {}

        cameraDevice = null;

        try {
            if (mediaRecorder != null) {
                mediaRecorder.release();
            }
        } catch (Exception ignored) {}

        mediaRecorder = null;

        try {
            if (recorderSurface != null) {
                recorderSurface.release();
            }
        } catch (Exception ignored) {}

        recorderSurface = null;

        try {
            if (previewSurface != null) {
                previewSurface.release();
            }
        } catch (Exception ignored) {}

        previewSurface = null;
        highSpeedRequests = null;
    }

    private void status(String message) {
        activity.runOnUiThread(
                () -> listener.onCameraStatus(
                        message));
    }

    private void error(String message) {
        state = State.ERROR;

        status(
                "ERROR · " + message);

        activity.runOnUiThread(
                () -> listener.onCameraError(
                        message));

        closeAll();
        closing = false;
    }

    private static String describe(
            Exception e) {

        String msg =
                e.getMessage();

        return e.getClass()
                .getSimpleName()
                + (msg == null
                        ? ""
                        : ": " + msg);
    }

    private static String sanitizeSessionId(
            String id) {

        if (id == null
                || id.isBlank()) {

            return "session-"
                    + System
                            .currentTimeMillis();
        }

        return id.replaceAll(
                "[^A-Za-z0-9._-]",
                "_");
    }

    private static String cameraErrorName(
            int error) {

        switch (error) {
            case CameraDevice.StateCallback
                    .ERROR_CAMERA_IN_USE:
                return "ERROR_CAMERA_IN_USE";

            case CameraDevice.StateCallback
                    .ERROR_MAX_CAMERAS_IN_USE:
                return "ERROR_MAX_CAMERAS_IN_USE";

            case CameraDevice.StateCallback
                    .ERROR_CAMERA_DISABLED:
                return "ERROR_CAMERA_DISABLED";

            case CameraDevice.StateCallback
                    .ERROR_CAMERA_DEVICE:
                return "ERROR_CAMERA_DEVICE";

            case CameraDevice.StateCallback
                    .ERROR_CAMERA_SERVICE:
                return "ERROR_CAMERA_SERVICE";

            default:
                return "UNKNOWN";
        }
    }
}
