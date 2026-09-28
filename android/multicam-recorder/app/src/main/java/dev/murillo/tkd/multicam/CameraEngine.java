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
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/** Continuous high-speed capture; ARM output is private working data, not a saved video. */
public final class CameraEngine {
    public static final String CAMERA_ID = "0";
    public static final int WIDTH = 1920;
    public static final int HEIGHT = 1080;
    public static final int FPS = 120;
    private static final long ARM_LIMIT_MS = 180_000L;
    public enum State { IDLE, ARMING, READY, RECORDING, STOPPING, ERROR }
    public interface Listener {
        void onCameraStatus(String status);
        void onCameraReady(String details);
        void onCameraStarted(long localStartCallNs);
        void onCameraStopped(String videoPath, String metadataPath);
        void onCameraError(String error);
        default void onCameraDiscarded(String detail) {}
    }
    private final Activity activity;
    private final TextureView textureView;
    private final Listener listener;
    private final CameraManager cameraManager;
    private final HandlerThread cameraThread;
    private final Handler cameraHandler;
    private final SessionEpoch epoch = new SessionEpoch();
    private final LightMonitor lightMonitor = new LightMonitor();
    private volatile State state = State.IDLE;
    private CameraCharacteristics characteristics;
    private CameraDevice cameraDevice;
    private CameraCaptureSession previewSession;
    private CameraConstrainedHighSpeedCaptureSession highSpeedSession;
    private Surface previewSurface, recorderSurface;
    private MediaRecorder mediaRecorder;
    private List<CaptureRequest> highSpeedRequests;
    private WarmupStore warmup;
    private RecordingTrim.Result trimResult;
    private Runnable armTimeout, startTask;
    private String sessionId;
    private File videoFile, metadataFile;
    private RecordingOrientation.Choice pendingOrientation, orientationAtArm, orientationAtStart;
    private volatile int savedRotationDegrees = -1;
    private volatile boolean orientationVerified;
    private volatile String orientationProblem, trimSummary = "No final video yet.";
    private volatile long meteredExposureNs, recorderStartElapsedNs = -1, scheduledStartNs = -1;
    private volatile long officialStartElapsedNs = -1, stopCallNs = -1;
    private volatile int meteredIso;
    private volatile long firstSensorTimestampNs = -1, lastUniqueSensorTimestampNs = -1;
    private volatile long uniqueSensorFrames, captureFailures, encodedFrameCount, encodedDurationUs;
    private volatile double encodedFps;
    private volatile boolean recorderStarted, officialRecordingStarted, closing;

    public CameraEngine(Activity activity, TextureView textureView, Listener listener) {
        this.activity = activity;
        this.textureView = textureView;
        this.listener = listener;
        cameraManager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        cameraThread = new HandlerThread("TkdMultiCamCamera");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
    }
    public State getState() { return state; }
    public double getLastEncodedFps() { return encodedFps; }
    public long getLastEncodedFrameCount() { return encodedFrameCount; }
    public LightMonitor.Snapshot getLightSnapshot(long elapsedNs) { return lightMonitor.snapshot(elapsedNs); }
    public String getTrimSummary() { return trimSummary; }
    public boolean hasOrientationWarning() { return orientationProblem != null; }
    public String getOrientationSummary() {
        return "Playback orientation: " + savedRotationDegrees + " degrees clockwise. "
                + (orientationVerified ? "MP4 matrix verified." : "MP4 matrix NOT verified.")
                + (orientationProblem == null ? "" : " Warning: " + orientationProblem);
    }
    public void arm(String id) { arm(id, null); }
    public void arm(String id, RecordingOrientation.Choice orientation) {
        cameraHandler.post(() -> {
            if (state != State.IDLE && state != State.ERROR) return;
            pendingOrientation = orientation;
            armInternal(id);
        });
    }
    public void startAt(long targetNs) { startAt(targetNs, null); }
    public void startAt(long targetNs, RecordingOrientation.Choice orientation) {
        cameraHandler.post(() -> {
            if (state != State.READY) return;
            orientationAtStart = orientation;
            scheduleOfficialStart(targetNs);
        });
    }
    public void stop() { cameraHandler.post(this::stopInternal); }
    public void shutdown() {
        cameraHandler.post(() -> {
            if (state != State.IDLE && state != State.ERROR) stopInternal();
            epoch.advance(); cancelTimers(); closeAll();
            try { discardWarmup(false); } catch (IOException e) { status(e.getMessage()); }
            cameraThread.quitSafely();
        });
    }
    private boolean isArming(long token) {
        return epoch.matches(token) && state == State.ARMING && !closing;
    }
    private void cancelTimers() {
        if (armTimeout != null) cameraHandler.removeCallbacks(armTimeout);
        if (startTask != null) cameraHandler.removeCallbacks(startTask);
        armTimeout = null; startTask = null;
    }
    private void discardWarmup(boolean force) throws IOException {
        if (warmup == null) return;
        if (force) warmup.discard(); else warmup.discardUnlessStarted();
        warmup = null;
    }
    private void armInternal(String id) {
        if (state != State.IDLE && state != State.ERROR) return;
        if (activity.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            error("Camera permission is not granted"); return;
        }
        closeAll(); cancelTimers();
        try { discardWarmup(false); } catch (IOException e) { error(e.getMessage()); return; }
        final long token = epoch.advance();
        state = State.ARMING; closing = false;
        sessionId = sanitizeSessionId(id);
        lightMonitor.reset(sessionId);
        orientationAtArm = pendingOrientation; pendingOrientation = null; orientationAtStart = null;
        savedRotationDegrees = -1; orientationVerified = false; orientationProblem = null;
        recorderStartElapsedNs = scheduledStartNs = officialStartElapsedNs = stopCallNs = -1;
        firstSensorTimestampNs = lastUniqueSensorTimestampNs = -1;
        uniqueSensorFrames = captureFailures = encodedFrameCount = encodedDurationUs = 0;
        encodedFps = 0; recorderStarted = officialRecordingStarted = false;
        trimResult = null;
        trimSummary = "ARM is preview/measurement. A private temporary is discarded unless START is pressed.";
        status("ARMING · preparing exposure/focus");
        armTimeout = () -> {
            if (epoch.matches(token) && (state == State.ARMING || state == State.READY)) {
                status("ARM timed out after 3 minutes; discarding temporary."); stopInternal();
            }
        };
        cameraHandler.postDelayed(armTimeout, ARM_LIMIT_MS);
        Runnable open = () -> cameraHandler.post(() -> { if (isArming(token)) openCameraForMetering(); });
        if (textureView.isAvailable()) open.run();
        else activity.runOnUiThread(() -> textureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
                textureView.setSurfaceTextureListener(null); open.run();
            }
            public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {}
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) { return false; }
            public void onSurfaceTextureUpdated(SurfaceTexture surface) {}
        }));
    }
    private void openCameraForMetering() {
        final long token = epoch.current();
        if (!isArming(token)) return;
        try {
            characteristics = cameraManager.getCameraCharacteristics(CAMERA_ID);
            validateHighSpeedMode(characteristics);
            SurfaceTexture st = textureView.getSurfaceTexture();
            if (st == null) { error("Preview surface unavailable"); return; }
            st.setDefaultBufferSize(WIDTH, HEIGHT);
            previewSurface = new Surface(st);
            cameraManager.openCamera(CAMERA_ID, new CameraDevice.StateCallback() {
                public void onOpened(CameraDevice camera) {
                    if (!isArming(token)) { camera.close(); return; }
                    cameraDevice = camera; createMeteringSession();
                }
                public void onDisconnected(CameraDevice camera) {
                    camera.close();
                    if (epoch.matches(token) && !closing) error("Camera disconnected");
                }
                public void onError(CameraDevice camera, int code) {
                    camera.close();
                    if (epoch.matches(token) && !closing) error("Camera error " + code);
                }
            }, cameraHandler);
        } catch (Exception e) { error("Open camera failed: " + describe(e)); }
    }
    private void validateHighSpeedMode(CameraCharacteristics c) {
        StreamConfigurationMap map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) throw new IllegalStateException("No stream configuration map");
        Size target = new Size(WIDTH, HEIGHT);
        if (!Arrays.asList(map.getHighSpeedVideoSizes()).contains(target))
            throw new IllegalStateException("1080p high-speed not advertised");
        boolean fixed120 = false;
        for (Range<Integer> range : map.getHighSpeedVideoFpsRangesFor(target))
            if (range.getLower() == FPS && range.getUpper() == FPS) { fixed120 = true; break; }
        if (!fixed120) throw new IllegalStateException("1080p120 fixed range not advertised");
    }
    private void createMeteringSession() {
        final long token = epoch.current();
        if (!isArming(token)) return;
        try {
            cameraDevice.createCaptureSession(List.of(previewSurface), new CameraCaptureSession.StateCallback() {
                public void onConfigured(CameraCaptureSession session) {
                    if (!isArming(token)) { session.close(); return; }
                    previewSession = session;
                    try {
                        CaptureRequest.Builder builder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        builder.addTarget(previewSurface);
                        builder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
                        builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON);
                        builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
                        builder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO);
                        session.setRepeatingRequest(builder.build(), meteringCallback, cameraHandler);
                        cameraHandler.postDelayed(() -> { if (isArming(token)) enterProvenHighSpeedPath(); }, 1000L);
                    } catch (Exception e) { error("Metering failed: " + describe(e)); }
                }
                public void onConfigureFailed(CameraCaptureSession session) {
                    if (isArming(token)) error("Metering session failed");
                }
            }, cameraHandler);
        } catch (Exception e) { error("Metering session failed: " + describe(e)); }
    }
    private final CameraCaptureSession.CaptureCallback meteringCallback = new CameraCaptureSession.CaptureCallback() {
        public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
            Long exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
            Integer iso = result.get(CaptureResult.SENSOR_SENSITIVITY);
            if (exposure != null) meteredExposureNs = exposure;
            if (iso != null) meteredIso = iso;
        }
    };
    private void enterProvenHighSpeedPath() {
        if (state != State.ARMING || cameraDevice == null) return;
        try {
            if (previewSession != null) {
                try { previewSession.stopRepeating(); } catch (Exception ignored) {}
                previewSession.close(); previewSession = null;
            }
            prepareRecorderExactlyLikeProbe(); createHighSpeedSession();
        } catch (Exception e) { error("High-speed setup failed: " + describe(e)); }
    }
    private void prepareRecorderExactlyLikeProbe() throws Exception {
        File base = activity.getExternalFilesDir(Environment.DIRECTORY_MOVIES);
        if (base == null) base = activity.getFilesDir();
        File dir = new File(base, "TKDPoomsae");
        if (!dir.exists() && !dir.mkdirs()) throw new IOException("Cannot create output directory");
        String device = Build.MODEL.replaceAll("[^A-Za-z0-9._-]", "_");
        videoFile = new File(dir, sessionId + "-" + device + "-cam0-1080p120.mp4");
        metadataFile = new File(dir, sessionId + "-" + device + "-cam0-1080p120.json");
        mediaRecorder = Build.VERSION.SDK_INT >= 31 ? new MediaRecorder(activity) : new MediaRecorder();
        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        mediaRecorder.setVideoSize(WIDTH, HEIGHT);
        mediaRecorder.setVideoFrameRate(FPS);
        mediaRecorder.setVideoEncodingBitRate(24_000_000);
        mediaRecorder.setOrientationHint(computeOrientationHint());
        if (videoFile.exists()) throw new IOException("Recording output already exists");
        warmup = new WarmupStore(new File(activity.getFilesDir(), "capture-temporary"));
        mediaRecorder.setOutputFile(warmup.file().getAbsolutePath());
        mediaRecorder.prepare();
        recorderSurface = mediaRecorder.getSurface();
    }
    private int computeOrientationHint() {
        if (orientationAtArm != null) return orientationAtArm.clockwiseDegrees;
        Integer sensor = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        if (sensor == null) return 0;
        int degrees = activity.getWindowManager().getDefaultDisplay().getRotation() * 90;
        return (sensor - degrees + 360) % 360;
    }
    private void createHighSpeedSession() {
        final long token = epoch.current();
        if (!isArming(token)) return;
        try {
            List<Surface> outputs = new ArrayList<>();
            outputs.add(previewSurface);
            outputs.add(recorderSurface);
            cameraDevice.createConstrainedHighSpeedCaptureSession(outputs, new CameraCaptureSession.StateCallback() {
                public void onConfigured(CameraCaptureSession session) {
                    if (!isArming(token)) { session.close(); return; }
                    if (!(session instanceof CameraConstrainedHighSpeedCaptureSession)) {
                        session.close(); error("Not a constrained high-speed session"); return;
                    }
                    highSpeedSession = (CameraConstrainedHighSpeedCaptureSession) session;
                    startHighSpeedPreroll();
                }
                public void onConfigureFailed(CameraCaptureSession session) {
                    if (isArming(token)) error("High-speed session configuration failed");
                }
            }, cameraHandler);
        } catch (Exception e) { error("Create high-speed session failed: " + describe(e)); }
    }
    private void startHighSpeedPreroll() {
        try {
            CaptureRequest.Builder builder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            builder.addTarget(previewSurface);
            builder.addTarget(recorderSurface);
            builder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
            builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(FPS, FPS));
            builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
            highSpeedRequests = highSpeedSession.createHighSpeedRequestList(builder.build());
            firstSensorTimestampNs = lastUniqueSensorTimestampNs = -1;
            uniqueSensorFrames = captureFailures = 0;
            // Keep the working burst-before-recorder ordering and automatic exposure.
            highSpeedSession.setRepeatingBurst(highSpeedRequests, recordingCallback, cameraHandler);
            mediaRecorder.start();
            recorderStarted = true;
            recorderStartElapsedNs = SystemClock.elapsedRealtimeNanos();
            state = State.READY;
            String details = "FHD120 preview ready · AF · AE auto";
            status("READY · " + details);
            activity.runOnUiThread(() -> listener.onCameraReady(details));
        } catch (Exception e) { error("High-speed warm-up failed: " + describe(e)); }
    }
    private void scheduleOfficialStart(long targetNs) {
        if (state != State.READY || !recorderStarted) { error("Camera is not READY"); return; }
        scheduledStartNs = targetNs;
        long remaining = targetNs - SystemClock.elapsedRealtimeNanos();
        if (remaining <= 0) { markOfficialStart(); return; }
        final long token = epoch.current();
        startTask = () -> {
            if (!epoch.matches(token) || state != State.READY) return;
            while (SystemClock.elapsedRealtimeNanos() < scheduledStartNs) Thread.onSpinWait();
            markOfficialStart();
        };
        cameraHandler.postDelayed(startTask, Math.max(0L, remaining / 1_000_000L - 3L));
    }
    private void markOfficialStart() {
        if (state != State.READY) return;
        try {
            if (warmup == null) throw new IOException("No prepared capture");
            warmup.markStarted();
        } catch (IOException e) { error(e.getMessage()); return; }
        cancelTimers();
        officialStartElapsedNs = SystemClock.elapsedRealtimeNanos();
        officialRecordingStarted = true; state = State.RECORDING;
        status("RECORDING · FHD120 · private warm-up will be trimmed");
        activity.runOnUiThread(() -> listener.onCameraStarted(officialStartElapsedNs));
    }
    private final CameraCaptureSession.CaptureCallback recordingCallback = new CameraCaptureSession.CaptureCallback() {
        public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
            Long ts = result.get(CaptureResult.SENSOR_TIMESTAMP);
            if (recorderStarted) {
                Long exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
                Integer iso = result.get(CaptureResult.SENSOR_SENSITIVITY);
                lightMonitor.observe(ts == null ? -1L : ts, SystemClock.elapsedRealtimeNanos(),
                        exposure == null ? -1L : exposure, iso == null ? -1 : iso);
            }
            if (ts == null) return;
            if (ts != lastUniqueSensorTimestampNs) {
                if (firstSensorTimestampNs < 0) firstSensorTimestampNs = ts;
                lastUniqueSensorTimestampNs = ts; uniqueSensorFrames++;
            }
        }
        public void onCaptureFailed(CameraCaptureSession session, CaptureRequest request, CaptureFailure failure) {
            captureFailures++;
        }
    };
    private void stopInternal() {
        if (state == State.IDLE || state == State.STOPPING) return;
        state = State.STOPPING; closing = true;
        epoch.advance(); cancelTimers();
        stopCallNs = SystemClock.elapsedRealtimeNanos();
        final boolean keepRecording = officialRecordingStarted;
        Exception recorderFailure = null;
        try {
            if (highSpeedSession != null) {
                try { highSpeedSession.stopRepeating(); } catch (Exception ignored) {}
                try { highSpeedSession.abortCaptures(); } catch (Exception ignored) {}
            }
            if (recorderStarted && mediaRecorder != null) mediaRecorder.stop();
        } catch (Exception failure) { recorderFailure = failure; }
        finally { recorderStarted = false; closeAll(); }
        if (!keepRecording) {
            try { discardWarmup(true); } catch (IOException failure) { error(failure.getMessage()); return; }
            videoFile = null; metadataFile = null;
            trimSummary = "ARM cancelled: private temporary deleted. No video was saved or published.";
            state = State.IDLE; closing = false;
            status(trimSummary);
            activity.runOnUiThread(() -> listener.onCameraDiscarded(trimSummary));
            return;
        }
        try {
            if (recorderFailure != null) throw new IOException("Recorder finalization failed", recorderFailure);
            if (warmup == null || !warmup.hasStarted()) throw new IOException("Missing started temporary");
            long startUs = Math.max(0L, (officialStartElapsedNs - recorderStartElapsedNs) / 1000L);
            status("SAVING · lossless trim from START (sync-frame boundary)");
            trimResult = RecordingTrim.remux(warmup.file(), videoFile, startUs);
            finalizeStorageOrientation();
            verifyRecordedVideo();
            if (encodedFrameCount < 2 || encodedFps <= 0) throw new IOException("Final video verification failed");
            trimSummary = String.format(Locale.US,
                    "Removed %.3f s of warm-up. Kept %.3f s before START for the preceding keyframe. "
                    + "No re-encoding. START offset uses the recorder-start clock estimate, not frame-exact sensor alignment.",
                    trimResult.discardedUs / 1e6, trimResult.retainedLeadUs / 1e6);
            writeMetadata();
            try { discardWarmup(true); }
            catch (IOException cleanup) { trimSummary += " Temporary cleanup warning: " + cleanup.getMessage(); }
        } catch (Exception failure) {
            String recovery = warmup == null ? "unavailable" : warmup.file().getAbsolutePath();
            error("Cannot save trimmed recording: " + failure.getMessage()
                    + ". No Gallery item published. Private recovery file: " + recovery);
            return;
        }
        String galleryPath = publishVideoToGallery();
        final String path = galleryPath != null ? galleryPath : videoFile.getAbsolutePath();
        final String jsonPath = metadataFile.getAbsolutePath();
        state = State.IDLE; closing = false;
        status(String.format(Locale.US, "SAVED · %.1f fps · %.3f s keyframe lead", encodedFps, trimResult.retainedLeadUs / 1e6));
        activity.runOnUiThread(() -> listener.onCameraStopped(path, jsonPath));
    }
    private void verifyRecordedVideo() {
        encodedFrameCount = encodedDurationUs = 0; encodedFps = 0;
        if (videoFile == null || !videoFile.exists() || videoFile.length() == 0) return;
        MediaExtractor ex = new MediaExtractor();
        try {
            ex.setDataSource(videoFile.getAbsolutePath()); int track = -1;
            for (int i = 0; i < ex.getTrackCount(); i++) {
                String mime = ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME);
                if (mime != null && mime.startsWith("video/")) { track = i; break; }
            }
            if (track < 0) return;
            ex.selectTrack(track); long min = Long.MAX_VALUE, max = Long.MIN_VALUE;
            while (ex.getSampleTrackIndex() >= 0) {
                long pts = ex.getSampleTime(); min = Math.min(min, pts); max = Math.max(max, pts);
                encodedFrameCount++; if (!ex.advance()) break;
            }
            if (encodedFrameCount > 1 && max > min) {
                encodedDurationUs = max - min;
                encodedFps = (encodedFrameCount - 1) * 1_000_000.0 / encodedDurationUs;
            }
        } catch (Exception e) { status("MP4 verification warning: " + describe(e)); }
        finally { ex.release(); }
    }
    private void finalizeStorageOrientation() {
        RecordingOrientation.Choice choice = officialRecordingStarted && orientationAtStart != null ? orientationAtStart : orientationAtArm;
        if (choice == null) choice = new RecordingOrientation.Choice(computeOrientationHint(), "legacy_display_fallback", -1, true);
        savedRotationDegrees = choice.clockwiseDegrees;
        orientationProblem = choice.uncertain ? "Physical orientation unavailable/ambiguous; used " + choice.source : null;
        try {
            if (videoFile == null || !videoFile.isFile()) throw new IOException("No finalized MP4");
            Mp4Orientation.set(videoFile, savedRotationDegrees); orientationVerified = true;
        } catch (IOException e) {
            orientationVerified = false;
            orientationProblem = "Cannot update rotation metadata: " + e.getMessage();
            status("Orientation warning: " + orientationProblem);
        }
    }
    private JSONObject orientationJson(RecordingOrientation.Choice choice) throws Exception {
        JSONObject j = new JSONObject();
        if (choice != null) j.put("clockwise_degrees", choice.clockwiseDegrees).put("source", choice.source)
                .put("measurement_age_ns", choice.measurementAgeNs).put("uncertain", choice.uncertain);
        return j;
    }
    private void writeMetadata() throws Exception {
        if (metadataFile == null) throw new IOException("Missing sidecar path");
        JSONObject j = new JSONObject();
        j.put("session_id", sessionId);
        j.put("light_monitor", LightJson.summary(lightMonitor.summary()));
        j.put("output_rotation_cw", savedRotationDegrees);
        j.put("output_rotation_verified", orientationVerified);
        j.put("output_rotation_warning", orientationProblem == null ? JSONObject.NULL : orientationProblem);
        j.put("orientation_at_arm", orientationJson(orientationAtArm));
        j.put("orientation_at_start", orientationJson(orientationAtStart));
        j.put("rotation_basis", "START");
        j.put("manufacturer", Build.MANUFACTURER); j.put("model", Build.MODEL);
        j.put("camera_id", CAMERA_ID); j.put("width", WIDTH); j.put("height", HEIGHT); j.put("fps_requested", FPS);
        j.put("capture_backend", "MediaRecorder continuous warm-up + lossless trim");
        j.put("exposure_mode", "AE_AUTO"); j.put("metered_exposure_ns", meteredExposureNs); j.put("metered_iso", meteredIso);
        j.put("recorder_start_elapsed_ns", recorderStartElapsedNs); j.put("scheduled_start_elapsed_ns", scheduledStartNs);
        j.put("official_start_elapsed_ns", officialStartElapsedNs); j.put("stop_elapsed_ns", stopCallNs);
        j.put("internal_warmup_ns", Math.max(0, officialStartElapsedNs - recorderStartElapsedNs));
        j.put("pre_roll_ns", trimResult == null ? 0 : trimResult.retainedLeadUs * 1000L);
        if (trimResult != null) {
            j.put("trim_mode", "previous_sync_without_reencoding");
            j.put("start_time_basis", "elapsed_since_recorder_start_estimate");
            j.put("requested_start_in_warmup_us", trimResult.requestedFromFirstUs);
            j.put("discarded_warmup_us", trimResult.discardedUs);
            j.put("start_in_final_video_us", trimResult.retainedLeadUs);
            j.put("first_retained_source_pts_us", trimResult.firstSourcePtsUs);
            j.put("sample_payload_sha256", trimResult.sampleDigest);
            j.put("max_pts_rounding_us", trimResult.maxPtsRoundingUs);
        }
        j.put("first_sensor_timestamp_ns", firstSensorTimestampNs); j.put("last_sensor_timestamp_ns", lastUniqueSensorTimestampNs);
        j.put("unique_sensor_frames", uniqueSensorFrames); j.put("capture_failures", captureFailures);
        j.put("encoded_frame_count", encodedFrameCount); j.put("encoded_duration_us", encodedDurationUs); j.put("encoded_fps", encodedFps);
        if (uniqueSensorFrames > 1 && firstSensorTimestampNs > 0 && lastUniqueSensorTimestampNs > firstSensorTimestampNs)
            j.put("sensor_fps_estimate", (uniqueSensorFrames - 1) * 1e9 / (lastUniqueSensorTimestampNs - firstSensorTimestampNs));
        try (FileWriter writer = new FileWriter(metadataFile)) { writer.write(j.toString(2)); }
    }
    private String publishVideoToGallery() {
        if (videoFile == null || !videoFile.exists() || videoFile.length() == 0) return null;
        if (Build.VERSION.SDK_INT < 29) return videoFile.getAbsolutePath();
        ContentResolver resolver = activity.getContentResolver(); ContentValues values = new ContentValues();
        values.put(MediaStore.Video.Media.DISPLAY_NAME, videoFile.getName());
        values.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4");
        values.put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/TKDPoomsae");
        values.put(MediaStore.Video.Media.IS_PENDING, 1);
        Uri uri = null;
        try {
            uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values);
            if (uri == null) throw new IOException("MediaStore insert failed");
            try (InputStream in = new FileInputStream(videoFile); OutputStream out = resolver.openOutputStream(uri, "w")) {
                if (out == null) throw new IOException("MediaStore output stream missing");
                byte[] buffer = new byte[1024 * 1024]; int n;
                while ((n = in.read(buffer)) >= 0) out.write(buffer, 0, n);
                out.flush();
            }
            ContentValues done = new ContentValues(); done.put(MediaStore.Video.Media.IS_PENDING, 0);
            if (resolver.update(uri, done, null, null) != 1) throw new IOException("MediaStore publication failed");
            return uri.toString();
        } catch (Exception e) {
            status("Gallery warning: " + describe(e));
            if (uri != null) try { resolver.delete(uri, null, null); } catch (Exception ignored) {}
            return null;
        }
    }
    private void closeAll() {
        try { if (previewSession != null) previewSession.close(); } catch (Exception ignored) {}
        previewSession = null;
        try { if (highSpeedSession != null) highSpeedSession.close(); } catch (Exception ignored) {}
        highSpeedSession = null;
        try { if (cameraDevice != null) cameraDevice.close(); } catch (Exception ignored) {}
        cameraDevice = null;
        try { if (mediaRecorder != null) mediaRecorder.release(); } catch (Exception ignored) {}
        mediaRecorder = null; recorderStarted = false;
        try { if (recorderSurface != null) recorderSurface.release(); } catch (Exception ignored) {}
        recorderSurface = null;
        try { if (previewSurface != null) previewSurface.release(); } catch (Exception ignored) {}
        previewSurface = null; highSpeedRequests = null;
    }
    private void error(String message) {
        epoch.advance(); cancelTimers(); closing = true; state = State.ERROR;
        try {
            if (highSpeedSession != null) highSpeedSession.stopRepeating();
            if (recorderStarted && mediaRecorder != null) mediaRecorder.stop();
        } catch (Exception ignored) {}
        closeAll();
        if (warmup != null && warmup.hasStarted()) message += "; private recovery: " + warmup.file().getAbsolutePath();
        try { discardWarmup(false); } catch (IOException e) { message += "; " + e.getMessage(); }
        final String reported = message;
        status("ERROR · " + reported);
        activity.runOnUiThread(() -> listener.onCameraError(reported));
        closing = false;
    }
    private void status(String message) { activity.runOnUiThread(() -> listener.onCameraStatus(message)); }
    private static String describe(Exception e) { return e.getClass().getSimpleName() + ": " + e.getMessage(); }
    private static String sanitizeSessionId(String id) {
        return id == null || id.isBlank() ? "session-" + System.currentTimeMillis() : id.replaceAll("[^A-Za-z0-9._-]", "_");
    }
}
