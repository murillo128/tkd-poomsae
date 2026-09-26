package dev.murillo.tkd.multicam;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.content.SharedPreferences;
import java.io.PrintWriter;
import java.io.StringWriter;
import android.view.Gravity;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public final class MainActivity extends Activity
        implements CameraEngine.Listener, NetworkCoordinator.Listener {

    private static final int CAMERA_PERMISSION_REQUEST = 2001;

    private TextureView textureView;
    private TextView roleText;
    private TextView networkText;
    private TextView cameraText;
    private TextView peersText;
    private TextView sessionText;

    private LinearLayout controllerPanel;
    private LinearLayout cameraPanel;
    private CheckBox recordLocal;
    private Button armButton;
    private Button startButton;
    private Button stopButton;

    private CameraEngine cameraEngine;
    private NetworkCoordinator network;

    private NetworkCoordinator.Role role;
    private boolean localReady;
    private boolean localRecording;
    private String currentSessionId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        installCrashRecorder();
        buildUi();

        cameraEngine = null;
        network = null;

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST);
        }

        role = null;
        roleText.setText("Role: choose CONTROLLER or CAMERA");
        controllerPanel.setVisibility(View.GONE);
        cameraPanel.setVisibility(View.GONE);
        networkText.setText("Network: not started");
        showPreviousCrashIfAny();
    }

    private void installCrashRecorder() {
        final Thread.UncaughtExceptionHandler previous =
                Thread.getDefaultUncaughtExceptionHandler();

        Thread.setDefaultUncaughtExceptionHandler((thread, throwable) -> {
            try {
                StringWriter sw = new StringWriter();
                throwable.printStackTrace(new PrintWriter(sw));
                getSharedPreferences("crash", MODE_PRIVATE)
                        .edit()
                        .putString("last_crash", sw.toString())
                        .apply();
            } catch (Exception ignored) {
            }

            if (previous != null) {
                previous.uncaughtException(thread, throwable);
            }
        });
    }

    private void showPreviousCrashIfAny() {
        SharedPreferences prefs = getSharedPreferences("crash", MODE_PRIVATE);
        String crash = prefs.getString("last_crash", null);
        if (crash == null || crash.isEmpty()) {
            return;
        }

        prefs.edit().remove("last_crash").apply();
        networkText.setText("Previous crash captured. Tap COPY CRASH below.");

        Button copyCrash = new Button(this);
        copyCrash.setText("COPY PREVIOUS CRASH");
        copyCrash.setOnClickListener(v -> {
            android.content.ClipboardManager cm =
                    (android.content.ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
            cm.setPrimaryClip(android.content.ClipData.newPlainText("TKD MultiCam crash", crash));
            Toast.makeText(this, "Crash copied", Toast.LENGTH_SHORT).show();
        });

        ((LinearLayout) networkText.getParent()).addView(copyCrash, 5);
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(14), dp(14), dp(14), dp(18));
        scroll.addView(root);

        TextView title = new TextView(this);
        title.setText("TKD MultiCam 120");
        title.setTextSize(25);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("Camera 0 · 1920×1080 · 120 fps · AF continuous · shutter target 1/500");
        subtitle.setTextSize(13);
        subtitle.setGravity(Gravity.CENTER_HORIZONTAL);
        subtitle.setPadding(0, dp(4), 0, dp(10));
        root.addView(subtitle);

        LinearLayout roleButtons = new LinearLayout(this);
        roleButtons.setOrientation(LinearLayout.HORIZONTAL);

        Button controller = new Button(this);
        controller.setText("CONTROLLER");
        controller.setOnClickListener(v -> chooseController());
        roleButtons.addView(controller, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        Button camera = new Button(this);
        camera.setText("CAMERA");
        camera.setOnClickListener(v -> chooseCamera());
        roleButtons.addView(camera, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        root.addView(roleButtons);

        roleText = statusLine(root, "Role: —");
        networkText = statusLine(root, "Network: —");
        cameraText = statusLine(root, "Camera: idle");
        sessionText = statusLine(root, "Session: —");

        textureView = new TextureView(this);
        root.addView(textureView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(230)));

        controllerPanel = new LinearLayout(this);
        controllerPanel.setOrientation(LinearLayout.VERTICAL);
        controllerPanel.setPadding(0, dp(8), 0, 0);

        recordLocal = new CheckBox(this);
        recordLocal.setText("Record on this controller too");
        recordLocal.setChecked(true);
        controllerPanel.addView(recordLocal);

        LinearLayout row1 = new LinearLayout(this);
        row1.setOrientation(LinearLayout.HORIZONTAL);

        Button discover = new Button(this);
        discover.setText("DISCOVER");
        discover.setOnClickListener(v -> {
            network.discoverNow();
            updatePeers();
        });
        row1.addView(discover, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        armButton = new Button(this);
        armButton.setText("ARM ALL");
        armButton.setOnClickListener(v -> armAll());
        row1.addView(armButton, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        controllerPanel.addView(row1);

        LinearLayout row2 = new LinearLayout(this);
        row2.setOrientation(LinearLayout.HORIZONTAL);

        startButton = new Button(this);
        startButton.setText("START +3s");
        startButton.setOnClickListener(v -> startAll());
        row2.addView(startButton, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        stopButton = new Button(this);
        stopButton.setText("STOP ALL");
        stopButton.setOnClickListener(v -> stopAll());
        row2.addView(stopButton, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        controllerPanel.addView(row2);

        TextView peerTitle = new TextView(this);
        peerTitle.setText("Discovered cameras");
        peerTitle.setTextSize(17);
        peerTitle.setPadding(0, dp(10), 0, dp(2));
        controllerPanel.addView(peerTitle);

        peersText = new TextView(this);
        peersText.setTextSize(13);
        peersText.setText("None yet.");
        peersText.setTextIsSelectable(true);
        controllerPanel.addView(peersText);

        root.addView(controllerPanel);

        cameraPanel = new LinearLayout(this);
        cameraPanel.setOrientation(LinearLayout.VERTICAL);
        cameraPanel.setPadding(0, dp(10), 0, 0);

        TextView cameraHelp = new TextView(this);
        cameraHelp.setText(
                "Camera mode: leave this phone on the same Wi‑Fi. "
                        + "The controller will discover it, ARM it, and schedule START/STOP.");
        cameraHelp.setTextSize(14);
        cameraPanel.addView(cameraHelp);

        Button cameraStop = new Button(this);
        cameraStop.setText("LOCAL EMERGENCY STOP");
        cameraStop.setOnClickListener(v -> cameraEngine.stop());
        cameraPanel.addView(cameraStop);

        root.addView(cameraPanel);

        TextView notes = new TextView(this);
        notes.setPadding(0, dp(12), 0, 0);
        notes.setTextSize(12);
        notes.setText(
                "Recordings are stored locally under the app Movies/TKDPoomsae directory. "
                        + "Each MP4 gets a JSON sidecar with camera timing/exposure metadata.");
        root.addView(notes);

        setContentView(scroll);
    }

    private TextView statusLine(LinearLayout parent, String text) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextSize(14);
        tv.setPadding(0, dp(2), 0, dp(2));
        parent.addView(tv);
        return tv;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void ensureCoreObjects() {
        if (cameraEngine == null) {
            cameraEngine = new CameraEngine(this, textureView, this);
        }
        if (network == null) {
            network = new NetworkCoordinator(this, this);
        }
    }

    private void chooseController() {
        try {
            role = NetworkCoordinator.Role.CONTROLLER;
            ensureCoreObjects();
        roleText.setText("Role: CONTROLLER");
        controllerPanel.setVisibility(View.VISIBLE);
        cameraPanel.setVisibility(View.GONE);

        localReady = false;
        localRecording = false;
        currentSessionId = null;

        if (network != null) {
            network.start(NetworkCoordinator.Role.CONTROLLER);
        }
            updatePeers();
        } catch (Throwable t) {
            handleUiFailure("Controller init", t);
        }
    }

    private void chooseCamera() {
        try {
            role = NetworkCoordinator.Role.CAMERA;
            ensureCoreObjects();
        roleText.setText("Role: CAMERA");
        controllerPanel.setVisibility(View.GONE);
        cameraPanel.setVisibility(View.VISIBLE);

        localReady = false;
        localRecording = false;
        currentSessionId = null;

        if (network != null) {
            network.start(NetworkCoordinator.Role.CAMERA);
        }
            cameraText.setText("Camera: waiting for controller ARM");
        } catch (Throwable t) {
            handleUiFailure("Camera init", t);
        }
    }

    private void handleUiFailure(String where, Throwable t) {
        StringWriter sw = new StringWriter();
        t.printStackTrace(new PrintWriter(sw));
        String full = where + "\n" + sw;
        networkText.setText("ERROR: " + where + " · " + t.getClass().getSimpleName()
                + ": " + t.getMessage());

        android.content.ClipboardManager cm =
                (android.content.ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        cm.setPrimaryClip(android.content.ClipData.newPlainText("TKD MultiCam error", full));
        Toast.makeText(this, "Error copied to clipboard", Toast.LENGTH_LONG).show();
    }

    private void armAll() {
        if (!ensureCameraPermission()) return;

        currentSessionId = new SimpleDateFormat(
                "yyyyMMdd-HHmmss", Locale.US).format(new Date());
        sessionText.setText("Session: " + currentSessionId);

        localReady = !recordLocal.isChecked();
        localRecording = false;

        List<NetworkCoordinator.Peer> peers = network.getPeers();
        if (peers.isEmpty() && !recordLocal.isChecked()) {
            Toast.makeText(this,
                    "No remote cameras discovered and local recording is disabled.",
                    Toast.LENGTH_LONG).show();
            return;
        }

        network.armAll(currentSessionId);

        if (recordLocal.isChecked()) {
            cameraEngine.arm(currentSessionId);
        } else {
            cameraText.setText("Camera: local recording disabled");
        }

        for (NetworkCoordinator.Peer peer : peers) {
            peer.ready = false;
        }
        updatePeers();
    }

    private void startAll() {
        if (currentSessionId == null) {
            Toast.makeText(this, "ARM ALL first.", Toast.LENGTH_SHORT).show();
            return;
        }

        List<NetworkCoordinator.Peer> peers = network.getPeers();
        boolean remotesReady = true;
        for (NetworkCoordinator.Peer peer : peers) {
            if (!peer.ready) {
                remotesReady = false;
                break;
            }
        }

        if (!localReady || !remotesReady) {
            Toast.makeText(this,
                    "Not all cameras are READY yet.",
                    Toast.LENGTH_LONG).show();
            updatePeers();
            return;
        }

        final long leadMs = 3000L;
        long targetControllerNs =
                SystemClock.elapsedRealtimeNanos() + leadMs * 1_000_000L;

        network.startAll(targetControllerNs, leadMs);

        if (recordLocal.isChecked()) {
            cameraEngine.startAt(targetControllerNs);
        }

        sessionText.setText(
                "Session: " + currentSessionId + " · START scheduled +3 s");
    }

    private void stopAll() {
        network.stopAll();
        if (recordLocal.isChecked()
                && cameraEngine.getState() != CameraEngine.State.IDLE) {
            cameraEngine.stop();
        }
        sessionText.setText(
                "Session: " + (currentSessionId == null ? "—" : currentSessionId)
                        + " · STOP sent");
    }

    private boolean ensureCameraPermission() {
        if (checkSelfPermission(Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            return true;
        }
        requestPermissions(
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_REQUEST);
        return false;
    }

    private void updatePeers() {
        if (network == null || peersText == null) return;

        runOnUiThread(() -> {
            List<NetworkCoordinator.Peer> peers = network.getPeers();
            if (peers.isEmpty()) {
                peersText.setText("None yet. Keep the other phones in CAMERA mode on the same Wi‑Fi.");
                return;
            }

            StringBuilder sb = new StringBuilder();
            for (NetworkCoordinator.Peer peer : peers) {
                sb.append(peer.name == null ? peer.id : peer.name);
                sb.append("\n  ");
                sb.append(peer.address == null ? "?" : peer.address.getHostAddress());
                sb.append(" · ");
                sb.append(peer.status);
                if (peer.hasSync) {
                    sb.append(String.format(
                            Locale.US,
                            " · sync RTT %.2f ms",
                            peer.rttMs()));
                } else {
                    sb.append(" · sync pending");
                }
                sb.append("\n");
            }
            peersText.setText(sb.toString().trim());
        });
    }

    // CameraEngine.Listener

    @Override
    public void onCameraStatus(String status) {
        cameraText.setText("Camera: " + status);
    }

    @Override
    public void onCameraReady(String details) {
        localReady = true;
        cameraText.setText("Camera: " + details);

        if (role == NetworkCoordinator.Role.CAMERA) {
            network.sendReady(details);
        }
    }

    @Override
    public void onCameraStarted(long localStartCallNs) {
        localRecording = true;
        cameraText.setText(String.format(
                Locale.US,
                "Camera: RECORDING · startCall %.3f s",
                localStartCallNs / 1_000_000_000.0));

        if (role == NetworkCoordinator.Role.CAMERA) {
            network.sendStarted(localStartCallNs);
        }
    }

    @Override
    public void onCameraStopped(String videoPath, String metadataPath) {
        localReady = false;
        localRecording = false;
        cameraText.setText(
                "Camera: STOPPED\nVideo: " + videoPath + "\nMetadata: " + metadataPath);

        if (role == NetworkCoordinator.Role.CAMERA) {
            network.sendStopped(videoPath);
        }
    }

    @Override
    public void onCameraError(String error) {
        localReady = false;
        localRecording = false;
        cameraText.setText("Camera: ERROR · " + error);

        if (role == NetworkCoordinator.Role.CAMERA) {
            network.sendCameraError(error);
        }
    }

    // NetworkCoordinator.Listener

    @Override
    public void onPeersChanged() {
        updatePeers();
    }

    @Override
    public void onNetworkStatus(String status) {
        runOnUiThread(() -> networkText.setText("Network: " + status));
    }

    @Override
    public void onArmCommand(String sessionId) {
        if (role != NetworkCoordinator.Role.CAMERA) return;

        currentSessionId = sessionId;
        runOnUiThread(() -> sessionText.setText("Session: " + sessionId));
        cameraEngine.arm(sessionId);
    }

    @Override
    public void onStartCommand(long localTargetNs, long fallbackDelayMs) {
        if (role != NetworkCoordinator.Role.CAMERA) return;

        long target = localTargetNs;
        if (target <= 0) {
            target = SystemClock.elapsedRealtimeNanos()
                    + fallbackDelayMs * 1_000_000L;
        }
        cameraEngine.startAt(target);
    }

    @Override
    public void onStopCommand() {
        if (role != NetworkCoordinator.Role.CAMERA) return;
        cameraEngine.stop();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST
                && (grantResults.length == 0
                || grantResults[0] != PackageManager.PERMISSION_GRANTED)) {
            Toast.makeText(
                    this,
                    "Camera permission is required to record.",
                    Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onDestroy() {
        if (network != null) network.stop();
        if (cameraEngine != null) cameraEngine.shutdown();
        super.onDestroy();
    }
}
