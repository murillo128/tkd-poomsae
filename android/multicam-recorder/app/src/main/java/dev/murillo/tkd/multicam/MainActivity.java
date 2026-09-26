package dev.murillo.tkd.multicam;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.TextureView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Space;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public final class MainActivity extends Activity
        implements CameraEngine.Listener, NetworkCoordinator.Listener {

    private static final int CAMERA_PERMISSION_REQUEST = 2001;

    private static final int BG = Color.rgb(7, 17, 30);
    private static final int PANEL = Color.rgb(13, 31, 49);
    private static final int PANEL_ALT = Color.rgb(17, 39, 60);
    private static final int CYAN = Color.rgb(28, 220, 240);
    private static final int CYAN_SOFT = Color.rgb(37, 160, 184);
    private static final int GREEN = Color.rgb(26, 222, 154);
    private static final int RED = Color.rgb(255, 73, 83);
    private static final int TEXT = Color.rgb(244, 248, 252);
    private static final int MUTED = Color.rgb(157, 179, 202);
    private static final int BORDER = Color.rgb(31, 82, 110);

    private TextureView textureView;
    private TextView roleText;
    private TextView networkText;
    private TextView cameraText;
    private TextView peersText;
    private TextView sessionText;
    private TextView peerTitle;

    private LinearLayout controllerPanel;
    private LinearLayout cameraPanel;
    private Switch recordLocal;
    private Button controllerButton;
    private Button cameraButton;
    private Button armButton;
    private Button startButton;
    private Button stopButton;
    private Button discoverButton;

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
        configureWindow();
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
        roleText.setText("Choose role");
        networkText.setText("Not started");
        cameraText.setText("Idle");
        sessionText.setText("No active session");
        showPreviousCrashIfAny();
    }

    private void configureWindow() {
        Window window = getWindow();
        window.setStatusBarColor(BG);
        window.setNavigationBarColor(BG);
        if (android.os.Build.VERSION.SDK_INT >= 23) {
            window.getDecorView().setSystemUiVisibility(0);
        }
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
                        .commit();
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
        networkText.setText("Previous crash captured");

        Button copyCrash = actionButton("COPY CRASH", CYAN_SOFT);
        copyCrash.setOnClickListener(v -> {
            ClipboardManager cm =
                    (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
            cm.setPrimaryClip(ClipData.newPlainText("TKD MultiCam crash", crash));
            Toast.makeText(this, "Crash copied", Toast.LENGTH_SHORT).show();
        });
        controllerPanel.addView(copyCrash, matchWrap(0, dp(8)));
    }

    private void buildUi() {
        boolean landscape =
                getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(BG);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(14), dp(18), dp(18));
        scroll.addView(root);

        root.addView(buildHeader(landscape), matchWrap(0, dp(10)));

        if (landscape) {
            buildLandscape(root);
        } else {
            buildPortrait(root);
        }

        setContentView(scroll);
    }

    private View buildHeader(boolean landscape) {
        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);

        ImageView icon = new ImageView(this);
        icon.setImageResource(R.drawable.tkd_multicam_icon);
        icon.setScaleType(ImageView.ScaleType.CENTER_CROP);
        GradientDrawable iconBg = rounded(PANEL_ALT, dp(14), BORDER, dp(1));
        icon.setBackground(iconBg);
        header.addView(icon, new LinearLayout.LayoutParams(dp(landscape ? 52 : 48), dp(landscape ? 52 : 48)));

        LinearLayout titleBox = new LinearLayout(this);
        titleBox.setOrientation(LinearLayout.VERTICAL);
        titleBox.setPadding(dp(12), 0, 0, 0);

        TextView title = new TextView(this);
        title.setText("TKD MultiCam 120");
        title.setTextColor(TEXT);
        title.setTextSize(landscape ? 24 : 23);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        titleBox.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("SYNCHRONIZED POOMSAE RECORDING");
        subtitle.setTextColor(CYAN);
        subtitle.setTextSize(10);
        subtitle.setLetterSpacing(0.16f);
        titleBox.addView(subtitle);

        header.addView(titleBox, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        LinearLayout roleChooser = new LinearLayout(this);
        roleChooser.setOrientation(LinearLayout.HORIZONTAL);

        controllerButton = roleButton("CONTROLLER");
        controllerButton.setOnClickListener(v -> chooseController());
        roleChooser.addView(controllerButton, new LinearLayout.LayoutParams(landscape ? dp(150) : 0, dp(48), landscape ? 0f : 1f));

        cameraButton = roleButton("CAMERA");
        cameraButton.setOnClickListener(v -> chooseCamera());
        LinearLayout.LayoutParams cameraLp = new LinearLayout.LayoutParams(landscape ? dp(120) : 0, dp(48), landscape ? 0f : 1f);
        cameraLp.setMargins(dp(8), 0, 0, 0);
        roleChooser.addView(cameraButton, cameraLp);

        if (landscape) {
            header.addView(roleChooser);
        } else {
            rootRoleChooser = roleChooser;
        }

        return header;
    }

    private LinearLayout rootRoleChooser;

    private void buildPortrait(LinearLayout root) {
        if (rootRoleChooser != null) {
            root.addView(rootRoleChooser, matchWrap(0, dp(10)));
        }

        root.addView(buildPreview(dp(260)), matchWrap(0, dp(10)));

        LinearLayout statusGrid = new LinearLayout(this);
        statusGrid.setOrientation(LinearLayout.VERTICAL);

        LinearLayout statusRow1 = new LinearLayout(this);
        statusRow1.setOrientation(LinearLayout.HORIZONTAL);
        roleText = statusCard(statusRow1, "ROLE", "Choose role");
        networkText = statusCard(statusRow1, "NETWORK", "Not started");
        statusGrid.addView(statusRow1, matchWrap(0, dp(6)));

        LinearLayout statusRow2 = new LinearLayout(this);
        statusRow2.setOrientation(LinearLayout.HORIZONTAL);
        cameraText = statusCard(statusRow2, "CAMERA", "Idle");
        sessionText = statusCard(statusRow2, "SESSION", "No active session");
        statusGrid.addView(statusRow2);

        root.addView(statusGrid, matchWrap(0, dp(10)));

        buildControllerPanel();
        root.addView(controllerPanel, matchWrap(0, dp(10)));

        buildCameraPanel();
        root.addView(cameraPanel, matchWrap(0, dp(10)));

        root.addView(buildPeersPanel(), matchWrap(0, 0));
    }

    private void buildLandscape(LinearLayout root) {
        LinearLayout mainRow = new LinearLayout(this);
        mainRow.setOrientation(LinearLayout.HORIZONTAL);
        mainRow.setGravity(Gravity.TOP);

        LinearLayout left = new LinearLayout(this);
        left.setOrientation(LinearLayout.VERTICAL);
        left.addView(buildPreview(dp(320)), matchWrap(0, dp(10)));

        LinearLayout statusRow = new LinearLayout(this);
        statusRow.setOrientation(LinearLayout.HORIZONTAL);
        roleText = statusCard(statusRow, "ROLE", "Choose role");
        networkText = statusCard(statusRow, "NETWORK", "Not started");
        cameraText = statusCard(statusRow, "CAMERA", "Idle");
        sessionText = statusCard(statusRow, "SESSION", "No active session");
        left.addView(statusRow);

        mainRow.addView(left, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.55f));

        LinearLayout right = new LinearLayout(this);
        right.setOrientation(LinearLayout.VERTICAL);
        right.setPadding(dp(14), 0, 0, 0);

        buildControllerPanel();
        right.addView(controllerPanel, matchWrap(0, dp(10)));

        buildCameraPanel();
        right.addView(cameraPanel, matchWrap(0, 0));

        mainRow.addView(right, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(mainRow, matchWrap(0, dp(12)));
        root.addView(buildPeersPanel(), matchWrap(0, 0));
    }

    private View buildPreview(int height) {
        FrameLayout frame = new FrameLayout(this);
        frame.setBackground(rounded(Color.rgb(3, 12, 22), dp(18), CYAN_SOFT, dp(1)));

        textureView = new TextureView(this);
        FrameLayout.LayoutParams previewLp =
                new FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, height);
        previewLp.setMargins(dp(3), dp(3), dp(3), dp(3));
        frame.addView(textureView, previewLp);

        TextView live = pill("●  LIVE PREVIEW", GREEN);
        FrameLayout.LayoutParams liveLp =
                new FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT);
        liveLp.gravity = Gravity.TOP | Gravity.START;
        liveLp.setMargins(dp(14), dp(12), 0, 0);
        frame.addView(live, liveLp);

        TextView mode = pill("FHD · 120 FPS · 1×", CYAN);
        FrameLayout.LayoutParams modeLp =
                new FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT);
        modeLp.gravity = Gravity.TOP | Gravity.END;
        modeLp.setMargins(0, dp(12), dp(14), 0);
        frame.addView(mode, modeLp);

        return frame;
    }

    private void buildControllerPanel() {
        controllerPanel = new LinearLayout(this);
        controllerPanel.setOrientation(LinearLayout.VERTICAL);
        controllerPanel.setPadding(dp(14), dp(14), dp(14), dp(14));
        controllerPanel.setBackground(rounded(PANEL, dp(18), BORDER, dp(1)));

        recordLocal = new Switch(this);
        recordLocal.setText("  Record on this controller too");
        recordLocal.setTextColor(TEXT);
        recordLocal.setTextSize(15);
        recordLocal.setChecked(true);
        controllerPanel.addView(recordLocal, matchWrap(0, dp(12)));

        LinearLayout row1 = new LinearLayout(this);
        row1.setOrientation(LinearLayout.HORIZONTAL);
        discoverButton = actionButton("⌁  DISCOVER", CYAN_SOFT);
        discoverButton.setOnClickListener(v -> {
            if (network != null) {
                network.discoverNow();
                updatePeers();
            }
        });
        row1.addView(discoverButton, weightedButton());

        armButton = actionButton("◉  ARM ALL", CYAN_SOFT);
        armButton.setOnClickListener(v -> armAll());
        LinearLayout.LayoutParams armLp = weightedButton();
        armLp.setMargins(dp(8), 0, 0, 0);
        row1.addView(armButton, armLp);
        controllerPanel.addView(row1, matchWrap(0, dp(8)));

        LinearLayout row2 = new LinearLayout(this);
        row2.setOrientation(LinearLayout.HORIZONTAL);
        startButton = actionButton("▶  START +3S", GREEN);
        startButton.setOnClickListener(v -> startAll());
        row2.addView(startButton, weightedButton());

        stopButton = actionButton("■  STOP ALL", RED);
        stopButton.setOnClickListener(v -> stopAll());
        LinearLayout.LayoutParams stopLp = weightedButton();
        stopLp.setMargins(dp(8), 0, 0, 0);
        row2.addView(stopButton, stopLp);
        controllerPanel.addView(row2);

        controllerPanel.setVisibility(View.GONE);
    }

    private void buildCameraPanel() {
        cameraPanel = new LinearLayout(this);
        cameraPanel.setOrientation(LinearLayout.VERTICAL);
        cameraPanel.setPadding(dp(16), dp(16), dp(16), dp(16));
        cameraPanel.setBackground(rounded(PANEL, dp(18), BORDER, dp(1)));

        TextView cameraTitle = new TextView(this);
        cameraTitle.setText("CAMERA NODE");
        cameraTitle.setTextColor(CYAN);
        cameraTitle.setTextSize(12);
        cameraTitle.setTypeface(Typeface.DEFAULT_BOLD);
        cameraPanel.addView(cameraTitle);

        TextView cameraHelp = new TextView(this);
        cameraHelp.setText(
                "Waiting for controller on the same Wi‑Fi. Keep the phone mounted and this preview visible.");
        cameraHelp.setTextColor(MUTED);
        cameraHelp.setTextSize(14);
        cameraHelp.setPadding(0, dp(6), 0, dp(14));
        cameraPanel.addView(cameraHelp);

        Button cameraStop = actionButton("■  LOCAL EMERGENCY STOP", RED);
        cameraStop.setOnClickListener(v -> {
            if (cameraEngine != null) cameraEngine.stop();
        });
        cameraPanel.addView(cameraStop);

        cameraPanel.setVisibility(View.GONE);
    }

    private View buildPeersPanel() {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(14), dp(12), dp(14), dp(14));
        panel.setBackground(rounded(PANEL, dp(18), BORDER, dp(1)));

        peerTitle = new TextView(this);
        peerTitle.setText("DISCOVERED CAMERAS");
        peerTitle.setTextColor(CYAN);
        peerTitle.setTextSize(14);
        peerTitle.setTypeface(Typeface.DEFAULT_BOLD);
        panel.addView(peerTitle);

        peersText = new TextView(this);
        peersText.setTextColor(MUTED);
        peersText.setTextSize(13);
        peersText.setPadding(0, dp(8), 0, 0);
        peersText.setText("No cameras yet. Put other phones in CAMERA mode on this Wi‑Fi.");
        peersText.setTextIsSelectable(true);
        panel.addView(peersText);

        return panel;
    }

    private TextView statusCard(LinearLayout parent, String label, String initial) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(10), dp(9), dp(10), dp(9));
        card.setBackground(rounded(PANEL_ALT, dp(14), BORDER, dp(1)));

        TextView l = new TextView(this);
        l.setText(label);
        l.setTextColor(CYAN);
        l.setTextSize(10);
        l.setTypeface(Typeface.DEFAULT_BOLD);
        card.addView(l);

        TextView value = new TextView(this);
        value.setText(initial);
        value.setTextColor(TEXT);
        value.setTextSize(12);
        value.setMaxLines(3);
        card.addView(value);

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        if (parent.getChildCount() > 0) lp.setMargins(dp(6), 0, 0, 0);
        parent.addView(card, lp);
        return value;
    }

    private TextView pill(String text, int accent) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextColor(TEXT);
        tv.setTextSize(11);
        tv.setTypeface(Typeface.DEFAULT_BOLD);
        tv.setPadding(dp(10), dp(5), dp(10), dp(5));
        tv.setBackground(rounded(Color.argb(220, 10, 23, 37), dp(20), accent, dp(1)));
        return tv;
    }

    private Button roleButton(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setTextColor(TEXT);
        b.setTextSize(12);
        b.setTypeface(Typeface.DEFAULT_BOLD);
        b.setAllCaps(false);
        b.setBackground(rounded(PANEL_ALT, dp(14), BORDER, dp(1)));
        return b;
    }

    private Button actionButton(String text, int accent) {
        Button b = new Button(this);
        b.setText(text);
        b.setTextColor(TEXT);
        b.setTextSize(13);
        b.setTypeface(Typeface.DEFAULT_BOLD);
        b.setAllCaps(false);
        b.setMinHeight(dp(54));
        b.setBackground(rounded(PANEL_ALT, dp(14), accent, dp(1)));
        return b;
    }

    private GradientDrawable rounded(int fill, int radius, int stroke, int strokeWidth) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(fill);
        g.setCornerRadius(radius);
        if (strokeWidth > 0) g.setStroke(strokeWidth, stroke);
        return g;
    }

    private LinearLayout.LayoutParams weightedButton() {
        return new LinearLayout.LayoutParams(0, dp(58), 1f);
    }

    private LinearLayout.LayoutParams matchWrap(int top, int bottom) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, top, 0, bottom);
        return lp;
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
            roleText.setText("CONTROLLER");
            controllerPanel.setVisibility(View.VISIBLE);
            cameraPanel.setVisibility(View.GONE);
            styleRoleButtons();

            localReady = false;
            localRecording = false;
            currentSessionId = null;

            network.start(NetworkCoordinator.Role.CONTROLLER);
            updatePeers();
        } catch (Throwable t) {
            handleUiFailure("Controller init", t);
        }
    }

    private void chooseCamera() {
        try {
            role = NetworkCoordinator.Role.CAMERA;
            ensureCoreObjects();
            roleText.setText("CAMERA");
            controllerPanel.setVisibility(View.GONE);
            cameraPanel.setVisibility(View.VISIBLE);
            styleRoleButtons();

            localReady = false;
            localRecording = false;
            currentSessionId = null;

            network.start(NetworkCoordinator.Role.CAMERA);
            cameraText.setText("Waiting for controller ARM");
        } catch (Throwable t) {
            handleUiFailure("Camera init", t);
        }
    }

    private void styleRoleButtons() {
        if (controllerButton == null || cameraButton == null) return;
        boolean controller = role == NetworkCoordinator.Role.CONTROLLER;
        controllerButton.setBackground(rounded(
                controller ? Color.rgb(11, 62, 81) : PANEL_ALT,
                dp(14), controller ? CYAN : BORDER, dp(1)));
        cameraButton.setBackground(rounded(
                controller ? PANEL_ALT : Color.rgb(11, 62, 81),
                dp(14), controller ? BORDER : CYAN, dp(1)));
    }

    private void handleUiFailure(String where, Throwable t) {
        StringWriter sw = new StringWriter();
        t.printStackTrace(new PrintWriter(sw));
        String full = where + "\n" + sw;
        networkText.setText("ERROR · " + t.getClass().getSimpleName());

        ClipboardManager cm =
                (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        cm.setPrimaryClip(ClipData.newPlainText("TKD MultiCam error", full));
        Toast.makeText(this, "Error copied to clipboard", Toast.LENGTH_LONG).show();
    }

    private void armAll() {
        if (!ensureCameraPermission()) return;

        currentSessionId = new SimpleDateFormat(
                "yyyyMMdd-HHmmss", Locale.US).format(new Date());
        sessionText.setText(currentSessionId + " · ARMING");

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
            cameraText.setText("Local recording disabled");
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

        sessionText.setText(currentSessionId + " · START +3s");
    }

    private void stopAll() {
        if (network != null) network.stopAll();
        if (recordLocal != null && recordLocal.isChecked()
                && cameraEngine != null
                && cameraEngine.getState() != CameraEngine.State.IDLE) {
            cameraEngine.stop();
        }
        sessionText.setText(
                (currentSessionId == null ? "No session" : currentSessionId)
                        + " · STOPPED");
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
            if (peerTitle != null) {
                peerTitle.setText("DISCOVERED CAMERAS (" + peers.size() + ")");
            }
            if (peers.isEmpty()) {
                peersText.setText("No cameras yet. Put other phones in CAMERA mode on this Wi‑Fi.");
                return;
            }

            StringBuilder sb = new StringBuilder();
            for (NetworkCoordinator.Peer peer : peers) {
                sb.append(peer.ready ? "● READY  " : "○ ");
                sb.append(peer.name == null ? peer.id : peer.name);
                sb.append("\n     ");
                sb.append(peer.address == null ? "?" : peer.address.getHostAddress());
                sb.append("  ·  ");
                sb.append(peer.status);
                if (peer.hasSync) {
                    sb.append(String.format(Locale.US, "  ·  RTT %.1f ms", peer.rttMs()));
                } else {
                    sb.append("  ·  sync pending");
                }
                sb.append("\n");
            }
            peersText.setText(sb.toString().trim());
            peersText.setTextColor(TEXT);
        });
    }

    @Override
    public void onCameraStatus(String status) {
        runOnUiThread(() -> cameraText.setText(status));
    }

    @Override
    public void onCameraReady(String details) {
        localReady = true;
        runOnUiThread(() -> cameraText.setText("READY · " + details));

        if (role == NetworkCoordinator.Role.CAMERA) {
            network.sendReady(details);
        }
    }

    @Override
    public void onCameraStarted(long localStartCallNs) {
        localRecording = true;
        runOnUiThread(() -> {
            cameraText.setText("RECORDING · 1080p120");
            sessionText.setText(currentSessionId + " · RECORDING");
        });

        if (role == NetworkCoordinator.Role.CAMERA) {
            network.sendStarted(localStartCallNs);
        }
    }

    @Override
    public void onCameraStopped(String videoPath, String metadataPath) {
        localReady = false;
        localRecording = false;
        runOnUiThread(() -> {
            cameraText.setText("STOPPED · saved to Gallery");
            sessionText.setText(
                    (currentSessionId == null ? "No session" : currentSessionId) + " · SAVED");
        });

        if (role == NetworkCoordinator.Role.CAMERA) {
            network.sendStopped(videoPath);
        }
    }

    @Override
    public void onCameraError(String error) {
        localReady = false;
        localRecording = false;
        runOnUiThread(() -> cameraText.setText("ERROR · " + error));

        if (role == NetworkCoordinator.Role.CAMERA) {
            network.sendCameraError(error);
        }
    }

    @Override
    public void onPeersChanged() {
        updatePeers();
    }

    @Override
    public void onNetworkStatus(String status) {
        runOnUiThread(() -> networkText.setText(status));
    }

    @Override
    public void onArmCommand(String sessionId) {
        if (role != NetworkCoordinator.Role.CAMERA) return;

        currentSessionId = sessionId;
        runOnUiThread(() -> sessionText.setText(sessionId + " · ARMING"));
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
