package dev.murillo.tkd.multicam;

import static dev.murillo.tkd.multicam.StudioUi.*;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.ActivityInfo;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.Typeface;
import android.net.ConnectivityManager;
import android.net.LinkAddress;
import android.net.LinkProperties;
import android.net.NetworkCapabilities;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.StatFs;
import android.os.SystemClock;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.text.style.StyleSpan;
import android.view.Gravity;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.net.Inet4Address;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Native, responsive capture dashboard. Recording and network engines are unchanged. */
public final class MainActivity extends Activity
        implements CameraEngine.Listener, NetworkCoordinator.Listener {
    private enum Phase { IDLE, ARMING, READY, SCHEDULED, RECORDING, STOPPING, SAVED, ERROR }
    private static final int CAMERA_PERMISSION = 2001;
    private StudioUi ui;
    private boolean wide, testMode, destroyed, localEnabled = true;
    private NetworkCoordinator.Role role = NetworkCoordinator.Role.CONTROLLER;
    private Phase phase = Phase.IDLE;
    private CameraEngine cameraEngine;
    private NetworkCoordinator network;
    private final ExecutorService networkIo = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final Set<String> participants = new HashSet<>();
    private String sessionId, videoUri, metadataPath;
    private String networkDetail = "Discovery has not started.", cameraDetail = "Camera idle.", sessionDetail = "No active session.";
    private long startedNs, scheduledNs;
    private double lastFps;
    private boolean networkError;
    private CameraPreview preview;
    private LinearLayout controllerActions, cameraActions, recordRow, peerContainer;
    private View startButton, armButton, stopButton;
    private Switch recordSwitch;
    private TextView previewState, previewCaption, fitButton, peerHeading, peerSummary;
    private View previewPlaceholder;
    private Tile roleTile, networkTile, cameraTile, sessionTile;
    private TextView controllerTab, cameraTab;
    private TextView lightIndicator;
    private LightMonitor.Snapshot lightSnapshot = LightMonitor.Snapshot.unknown("", "No capture yet");
    private long lastLightRefreshNs = -1;

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ui = new StudioUi(this);
        wide = getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
        testMode = (getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0
                && getIntent().getBooleanExtra("ui_test", false);
        String stored = saved == null ? getPreferences(MODE_PRIVATE).getString("role", "CONTROLLER") : saved.getString("role", "CONTROLLER");
        role = "CAMERA".equals(stored) ? NetworkCoordinator.Role.CAMERA : NetworkCoordinator.Role.CONTROLLER;
        localEnabled = saved == null ? getPreferences(MODE_PRIVATE).getBoolean("local", true) : saved.getBoolean("local", true);
        buildScreen();
        preview.setFill(saved != null && saved.getBoolean("fill", false));
        render();
        if (!testMode) startNetwork();
        main.post(ticker);
    }

    private void buildScreen() {
        LinearLayout shell = ui.column();
        shell.setTag("dashboard");
        shell.setBackground(ui.panel(BG, 0xff030d16, BG, 0));
        shell.addView(header(), new LinearLayout.LayoutParams(-1, ui.dp(wide ? 42 : 58)));
        if (wide) {
            LinearLayout body = ui.row();
            body.setGravity(Gravity.TOP);
            body.addView(viewfinder(), new LinearLayout.LayoutParams(0, -1, 1.48f));
            LinearLayout right = ui.column();
            right.setPadding(ui.dp(10), 0, 0, 0);
            right.addView(statusStrip(), margin(-1, ui.dp(54), 0, 5));
            right.addView(recordLocalRow(), margin(-1, ui.dp(38), 0, 5));
            right.addView(actionGrid(true));
            right.addView(cameraControls(true));
            ScrollView commands = new ScrollView(this);
            commands.setFillViewport(false);
            commands.setVerticalScrollBarEnabled(false);
            commands.addView(right, new ScrollView.LayoutParams(-1, -2));
            body.addView(commands, new LinearLayout.LayoutParams(0, -1, 1));
            shell.addView(body, margin(-1, 0, 7, 8, 1));
            shell.addView(peersPanel(true), new LinearLayout.LayoutParams(-1, ui.dp(80)));
        } else {
            LinearLayout content = ui.column();
            int width = getResources().getDisplayMetrics().widthPixels - ui.dp(28);
            // Bounded portrait viewfinder. FIT shows the full 9:16 capture with side bars;
            // FILL is an explicit, labelled crop, never an anisotropic resize.
            int height = Math.min(Math.round(width * 0.78f), ui.dp(310));
            content.addView(viewfinder(), margin(-1, height, 8, 9));
            content.addView(statusStrip(), margin(-1, ui.dp(76), 0, 9));
            content.addView(recordLocalRow(), margin(-1, ui.dp(57), 0, 9));
            content.addView(actionGrid(false), margin(-1, -2, 0, 12));
            content.addView(cameraControls(false), margin(-1, -2, 0, 12));
            content.addView(peersPanel(false), margin(-1, -2, 0, 14));
            LinearLayout footer = ui.row();
            footer.setPadding(ui.dp(6), ui.dp(10), ui.dp(6), ui.dp(12));
            TextView tagline = ui.line("Record better. Train smarter.", 10, MUTED, false);
            tagline.setLetterSpacing(0.08f);
            footer.addView(tagline, new LinearLayout.LayoutParams(0, -2, 1));
            footer.addView(ui.line("TKD MultiCam  1.8", 11, CYAN, true));
            content.addView(footer);
            ScrollView scroll = new ScrollView(this);
            scroll.setVerticalScrollBarEnabled(false);
            scroll.addView(content);
            shell.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));
        }
        setContentView(shell);
        if (Build.VERSION.SDK_INT >= 30) {
            getWindow().setDecorFitsSystemWindows(false);
            WindowInsetsController controller = getWindow().getInsetsController();
            if (controller != null) controller.setSystemBarsAppearance(0,
                    WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS | WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS);
        } else {
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                    | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION);
        }
        getWindow().setStatusBarColor(Color.TRANSPARENT);
        getWindow().setNavigationBarColor(Color.TRANSPARENT);
        shell.setOnApplyWindowInsetsListener((view, insets) -> {
            int left, top, right, bottom;
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets safe = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                left = safe.left; top = safe.top; right = safe.right; bottom = safe.bottom;
            } else {
                left = insets.getSystemWindowInsetLeft(); top = insets.getSystemWindowInsetTop();
                right = insets.getSystemWindowInsetRight(); bottom = insets.getSystemWindowInsetBottom();
            }
            view.setPadding(left + ui.dp(wide ? 10 : 14), top + ui.dp(5), right + ui.dp(wide ? 10 : 14), bottom + ui.dp(6));
            return insets;
        });
        shell.requestApplyInsets();
    }

    private View header() {
        LinearLayout bar = ui.row();
        ImageView brand = new ImageView(this);
        brand.setImageResource(R.drawable.studio_brand);
        brand.setScaleType(ImageView.ScaleType.FIT_CENTER);
        brand.setContentDescription("TKD MultiCam");
        bar.addView(brand, new LinearLayout.LayoutParams(ui.dp(wide ? 40 : 52), ui.dp(wide ? 40 : 52)));
        LinearLayout titles = ui.column();
        titles.setGravity(Gravity.CENTER_VERTICAL);
        titles.setPadding(ui.dp(9), 0, ui.dp(4), 0);
        TextView title = ui.line("", wide ? 22 : 24, TEXT, false);
        title.setTag("brand_title");
        title.setTypeface(Typeface.create("sans-serif-condensed", Typeface.ITALIC));
        SpannableString name = new SpannableString("TKD MultiCam 120");
        name.setSpan(new StyleSpan(Typeface.BOLD_ITALIC), 0, 3, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        name.setSpan(new ForegroundColorSpan(CYAN), 13, 16, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        title.setText(name);
        title.setAutoSizeTextTypeUniformWithConfiguration(16, wide ? 23 : 25, 1, android.util.TypedValue.COMPLEX_UNIT_SP);
        titles.addView(title, new LinearLayout.LayoutParams(-1, ui.dp(wide ? 26 : 29)));
        TextView subtitle = ui.line("SYNCHRONIZED POOMSAE RECORDING", wide ? 7 : 8, CYAN, false);
        subtitle.setLetterSpacing(0.16f);
        titles.addView(subtitle, margin(-1, -2, 3, 0));
        bar.addView(titles, new LinearLayout.LayoutParams(0, -1, 1));
        if (wide) {
            bar.addView(ui.icon("wifi", MUTED, 18));
            TextView battery = ui.line("  " + batteryPercent() + "   ", 11, MUTED, false);
            bar.addView(battery);
            LinearLayout segment = ui.row();
            segment.setPadding(ui.dp(2), ui.dp(2), ui.dp(2), ui.dp(2));
            segment.setBackground(ui.panel(0xff071c29, 0xff061421, LINE, 9));
            controllerTab = tab("CONTROLLER", NetworkCoordinator.Role.CONTROLLER);
            cameraTab = tab("CAMERA", NetworkCoordinator.Role.CAMERA);
            segment.addView(controllerTab, new LinearLayout.LayoutParams(ui.dp(103), ui.dp(34)));
            segment.addView(cameraTab, new LinearLayout.LayoutParams(ui.dp(84), ui.dp(34)));
            bar.addView(segment);
        }
        ImageView menu = ui.icon("menu", TEXT, 22);
        FrameLayout menuTarget = new FrameLayout(this);
        menuTarget.setTag("menu");
        menuTarget.addView(menu, new FrameLayout.LayoutParams(ui.dp(22), ui.dp(22), Gravity.CENTER));
        menuTarget.setContentDescription("Options and camera role");
        menuTarget.setOnClickListener(v -> menu());
        bar.addView(menuTarget, new LinearLayout.LayoutParams(ui.dp(36), -1));
        return bar;
    }

    private TextView tab(String name, NetworkCoordinator.Role newRole) {
        TextView text = ui.line(name, 11, CYAN, true);
        text.setGravity(Gravity.CENTER);
        text.setOnClickListener(v -> selectRole(newRole));
        return text;
    }

    private FrameLayout viewfinder() {
        FrameLayout frame = new FrameLayout(this);
        frame.setTag("preview_frame");
        frame.setBackground(ui.panel(0xff02111b, 0xff071d28, 0xff39b4c7, 13));
        frame.setClipToOutline(true);
        preview = new CameraPreview(this);
        // The parent owns the viewport height; don't add that height a second time
        // inside its margins, which clipped the bottom edge of the portrait texture.
        FrameLayout.LayoutParams camera = new FrameLayout.LayoutParams(-1, -1);
        camera.setMargins(ui.dp(2), ui.dp(2), ui.dp(2), ui.dp(2));
        frame.addView(preview, camera);
        LinearLayout placeholder = ui.column();
        placeholder.setGravity(Gravity.CENTER);
        placeholder.addView(ui.icon("camera", 0xff3a7387, 43));
        TextView hint = ui.text("Ready to frame your session", 14, 0xffadc3cf, false);
        hint.setGravity(Gravity.CENTER);
        placeholder.addView(hint, margin(-1, -2, 10, 4));
        TextView sub = ui.text("ARM starts the preview and pre-roll", 10, MUTED, false);
        sub.setGravity(Gravity.CENTER);
        placeholder.addView(sub);
        frame.addView(placeholder, new FrameLayout.LayoutParams(-1, -1));
        previewPlaceholder = placeholder;
        previewState = overlay("PREVIEW OFF", TEXT, 11);
        frame.addView(previewState, overlayAt(Gravity.TOP | Gravity.START, 10, 10));
        TextView format = overlay("FHD 120  ·  1.0×", TEXT, 10);
        frame.addView(format, overlayAt(Gravity.TOP | Gravity.END, 10, 10));
        lightIndicator = overlay("LIGHT UNKNOWN", MUTED, 10);
        lightIndicator.setTag("light_indicator");
        lightIndicator.setSingleLine(false);
        lightIndicator.setMaxLines(2);
        lightIndicator.setVisibility(View.GONE);
        lightIndicator.setOnClickListener(v -> details("Light / exposure", lightSnapshot.detail(), lightSnapshot.warning()));
        frame.addView(lightIndicator, overlayAt(Gravity.TOP | Gravity.START, 10, 46));
        frame.addOnLayoutChangeListener((v,l,t,r,b,ol,ot,or,ob) ->
                lightIndicator.setMaxWidth(Math.max(1, r-l-ui.dp(20))));
        previewCaption = overlay("Camera 0  ·  Auto exposure", MUTED, 10);
        frame.addView(previewCaption, overlayAt(Gravity.BOTTOM | Gravity.START, 10, 10));
        fitButton = overlay("FIT", CYAN, 10);
        fitButton.setGravity(Gravity.CENTER);
        fitButton.setTag("preview_fit");
        fitButton.setContentDescription("Preview fit or cropped fill. Does not change the recording.");
        FrameLayout.LayoutParams fit = new FrameLayout.LayoutParams(ui.dp(48), ui.dp(40), Gravity.BOTTOM | Gravity.END);
        fit.setMargins(0, 0, ui.dp(7), ui.dp(7));
        frame.addView(fitButton, fit);
        fitButton.setOnClickListener(v -> {
            preview.setFill(!preview.isFill());
            render();
        });
        return frame;
    }

    private TextView overlay(String text, int color, int size) {
        TextView t = ui.line(text, size, color, false);
        t.setPadding(ui.dp(9), ui.dp(6), ui.dp(9), ui.dp(6));
        t.setBackground(ui.touch(0xdd06131d, 0xdd06131d, 0x665ba0b5, 7));
        return t;
    }
    private FrameLayout.LayoutParams overlayAt(int gravity, int x, int y) {
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(-2, -2, gravity);
        lp.setMargins(ui.dp(x), ui.dp(y), ui.dp(x), ui.dp(y));
        return lp;
    }

    private View statusStrip() {
        LinearLayout strip = ui.row();
        roleTile = new Tile("Role", "role", this::roleDialog);
        networkTile = new Tile("Network", "wifi", () -> details("Network", networkDetail + "\n\n" + wifiInfo(), networkError));
        cameraTile = new Tile("Camera", "camera", () -> details("Camera 0", cameraDetail + "\n\n" + lightSnapshot.detail()
                + "\n\nRequested capture: 1920×1080 at 120 fps.\nContinuous autofocus; automatic exposure. No fixed 1/500 shutter is currently requested.\nPreview FIT/FILL changes only the display, not the saved video.", phase == Phase.ERROR || (lastFps > 0 && lastFps < 110)));
        sessionTile = new Tile("Session", "session", () -> details("Session", sessionDetail
                + "\n\nSession: " + (sessionId == null ? "None" : sessionId)
                + "\nARM already writes pre-roll. START marks the synchronized beginning in the JSON sidecar."
                + "\nStorage available: " + freeStorage()
                + "\nVideo: " + (videoUri == null ? "Not saved yet" : videoUri)
                + "\nMetadata: " + (metadataPath == null ? "Not saved yet" : metadataPath), phase == Phase.ERROR));
        Tile[] tiles = {roleTile, networkTile, cameraTile, sessionTile};
        for (int i = 0; i < tiles.length; i++) {
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, -1, 1);
            if (i > 0) lp.leftMargin = ui.dp(5);
            strip.addView(tiles[i].view, lp);
        }
        return strip;
    }

    private final class Tile {
        final FrameLayout view = new FrameLayout(MainActivity.this);
        final TextView value, small, badge;
        Tile(String title, String glyph, Runnable click) {
            view.setTag("status_" + title.toLowerCase(Locale.US));
            view.setBackground(ui.touch(PANEL, 0xff091c29, LINE, 9));
            LinearLayout content = ui.column();
            content.setPadding(ui.dp(wide ? 6 : 8), ui.dp(8), ui.dp(6), ui.dp(6));
            LinearLayout top = ui.row();
            top.addView(ui.icon(glyph, CYAN, wide ? 17 : 19));
            TextView label = ui.line(title, wide ? 9 : 10, MUTED, false);
            label.setPadding(ui.dp(4), 0, 0, 0);
            top.addView(label, new LinearLayout.LayoutParams(0, -2, 1));
            content.addView(top);
            value = ui.line("Idle", wide ? 10 : 11, TEXT, true);
            content.addView(value, margin(-1, -2, 6, 0));
            small = ui.line("", 9, MUTED, false);
            if (!wide) content.addView(small, margin(-1, -2, 5, 0));
            view.addView(content, new FrameLayout.LayoutParams(-1, -1));
            badge = ui.line("!", 10, Color.WHITE, true);
            badge.setGravity(Gravity.CENTER);
            badge.setBackground(ui.panel(RED, RED, RED, 9));
            FrameLayout.LayoutParams b = new FrameLayout.LayoutParams(ui.dp(16), ui.dp(16), Gravity.TOP | Gravity.END);
            b.setMargins(0, ui.dp(3), ui.dp(3), 0);
            view.addView(badge, b);
            badge.setVisibility(View.GONE);
            view.setFocusable(true);
            view.setOnClickListener(v -> click.run());
        }
        void show(String state, String detail, int color, boolean error) {
            value.setText(state); value.setTextColor(color); small.setText(detail);
            badge.setVisibility(error ? View.VISIBLE : View.GONE);
            view.setContentDescription(state + ". " + detail + ". Tap for details.");
        }
    }

    private View recordLocalRow() {
        recordRow = ui.row();
        recordRow.setPadding(ui.dp(10), ui.dp(5), ui.dp(11), ui.dp(5));
        recordRow.setBackground(ui.panel(PANEL, 0xff091b28, LINE, 10));
        recordSwitch = new Switch(this);
        recordSwitch.setTag("record_local");
        recordSwitch.setChecked(localEnabled);
        recordSwitch.setThumbTintList(new ColorStateList(new int[][]{{android.R.attr.state_checked},{}}, new int[]{Color.WHITE, MUTED}));
        recordSwitch.setTrackTintList(new ColorStateList(new int[][]{{android.R.attr.state_checked},{}}, new int[]{CYAN, 0xff254b5d}));
        recordSwitch.setContentDescription("Record on this controller too");
        recordRow.addView(recordSwitch, new LinearLayout.LayoutParams(ui.dp(47), -1));
        LinearLayout labels = ui.column();
        labels.setPadding(ui.dp(8), 0, ui.dp(4), 0);
        labels.setGravity(Gravity.CENTER_VERTICAL);
        labels.addView(ui.line(wide ? "Record on this device" : "Record on this controller too", wide ? 11 : 13, TEXT, true));
        if (!wide) labels.addView(ui.line("Also save a local video", 10, MUTED, false), margin(-1, -2, 4, 0));
        recordRow.addView(labels, new LinearLayout.LayoutParams(0, -1, 1));
        recordRow.addView(ui.icon("camera", 0xff7798ac, wide ? 19 : 23));
        recordSwitch.setOnCheckedChangeListener((button, checked) -> {
            localEnabled = checked;
            getPreferences(MODE_PRIVATE).edit().putBoolean("local", checked).apply();
            render();
        });
        return recordRow;
    }

    private View actionGrid(boolean compact) {
        controllerActions = ui.column();
        LinearLayout first = ui.row(), second = ui.row();
        View discover = ui.command("action_discover", "search", "DISCOVER", "Find cameras on Wi-Fi", CYAN, compact,
                () -> io(() -> { if (network != null) network.discoverNow(); }));
        armButton = ui.command("action_arm", "arm", "ARM ALL", "Prepare + start pre-roll", CYAN, compact, this::armAll);
        startButton = ui.command("action_start", "play", "START +3S", "Synchronized start", GREEN, compact, this::startAll);
        stopButton = ui.command("action_stop", "stop", "STOP ALL", "Finish and save videos", RED, compact, this::stopAll);
        first.addView(discover, new LinearLayout.LayoutParams(0, ui.dp(compact ? 50 : 64), 1));
        LinearLayout.LayoutParams a = new LinearLayout.LayoutParams(0, ui.dp(compact ? 50 : 64), 1); a.leftMargin = ui.dp(7);
        first.addView(armButton, a);
        second.addView(startButton, new LinearLayout.LayoutParams(0, ui.dp(compact ? 50 : 64), 1));
        LinearLayout.LayoutParams b = new LinearLayout.LayoutParams(0, ui.dp(compact ? 50 : 64), 1); b.leftMargin = ui.dp(7);
        second.addView(stopButton, b);
        controllerActions.addView(first, margin(-1, -2, 0, 7));
        controllerActions.addView(second);
        return controllerActions;
    }

    private View cameraControls(boolean compact) {
        cameraActions = ui.column();
        TextView help = ui.text("CAMERA NODE\nThe controller starts and stops this camera over Wi-Fi.", compact ? 11 : 13, MUTED, false);
        cameraActions.addView(help, margin(-1, -2, 3, 10));
        cameraActions.addView(ui.command("action_emergency", "stop", "STOP CAMERA", "Save this device's video", RED, compact,
                () -> { if (cameraEngine != null && active() && phase != Phase.STOPPING) { phase = Phase.STOPPING; cameraEngine.stop(); render(); } }),
                new LinearLayout.LayoutParams(-1, ui.dp(54)));
        return cameraActions;
    }

    private View peersPanel(boolean horizontal) {
        LinearLayout panel = ui.column();
        panel.setPadding(ui.dp(horizontal ? 10 : 0), ui.dp(horizontal ? 5 : 8), ui.dp(horizontal ? 10 : 0), ui.dp(horizontal ? 5 : 7));
        if (horizontal) panel.setBackground(ui.panel(PANEL, 0xff071925, LINE, 10));
        LinearLayout heading = ui.row();
        if (!horizontal) heading.addView(ui.icon("role", CYAN, 20));
        peerHeading = ui.line("Discovered cameras (0)", horizontal ? 11 : 13, horizontal ? CYAN : TEXT, true);
        peerHeading.setPadding(ui.dp(5), 0, 0, 0);
        heading.addView(peerHeading, new LinearLayout.LayoutParams(0, -2, 1));
        peerSummary = ui.line("Searching", 10, MUTED, false);
        heading.addView(peerSummary);
        panel.addView(heading, margin(-1, ui.dp(horizontal ? 16 : 20), 0, 5));
        peerContainer = horizontal ? ui.row() : ui.column();
        peerContainer.setTag("peer_list");
        if (horizontal) {
            HorizontalScrollView scroll = new HorizontalScrollView(this);
            scroll.setHorizontalScrollBarEnabled(false);
            scroll.addView(peerContainer);
            panel.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));
        } else panel.addView(peerContainer);
        return panel;
    }

    private void renderPeers() {
        if (peerContainer == null) return;
        List<NetworkCoordinator.Peer> peers = peers();
        peerHeading.setText("Discovered cameras (" + peers.size() + ")");
        long ready = 0;
        for (NetworkCoordinator.Peer peer : peers) if (peer.ready) ready++;
        peerSummary.setText(peers.isEmpty() ? "Local Wi-Fi" : ready + " / " + peers.size() + " ready");
        peerContainer.removeAllViews();
        if (peers.isEmpty()) {
            LinearLayout empty = ui.row();
            empty.setPadding(ui.dp(11), ui.dp(wide ? 6 : 10), ui.dp(11), ui.dp(wide ? 6 : 10));
            empty.addView(ui.icon("phone", 0xff54879b, 24));
            TextView hint = ui.text(role == NetworkCoordinator.Role.CONTROLLER
                    ? "No cameras yet. Choose CAMERA on the other phones."
                    : "This phone is a camera node. Waiting for the controller.", 11, MUTED, false);
            hint.setPadding(ui.dp(10), 0, 0, 0);
            empty.addView(hint);
            peerContainer.addView(empty);
            return;
        }
        for (NetworkCoordinator.Peer peer : peers) {
            String name = peer.name == null ? peer.id : peer.name;
            String ip = peer.address == null ? "IP unknown" : peer.address.getHostAddress();
            String rtt = peer.hasSync ? String.format(Locale.US, "RTT %.1f ms · Synced", peer.rttMs()) : "Clock sync pending";
            boolean rec = "recording".equalsIgnoreCase(peer.status);
            boolean err = peer.status != null && peer.status.toLowerCase(Locale.US).contains("error");
            int color = err ? RED : rec ? RED : peer.ready ? GREEN : CYAN;
            LightJson.Report report = peer.lightReport;
            LightMonitor.Snapshot light = report == null
                    ? LightMonitor.Snapshot.unknown("", "Camera has not supplied light metadata (older app or no active capture)")
                    : report.at(SystemClock.elapsedRealtimeNanos());
            int lightColor = lightColor(light);
            LinearLayout card = ui.row();
            card.setPadding(ui.dp(11), ui.dp(9), ui.dp(11), ui.dp(9));
            card.setBackground(ui.touch(PANEL, 0xff091c29, light.warning() ? lightColor : LINE, 10));
            card.addView(ui.icon("phone", MUTED, wide ? 22 : 29));
            LinearLayout info = ui.column(); info.setPadding(ui.dp(10), 0, ui.dp(8), 0);
            info.addView(ui.line(name, wide ? 10 : 12, TEXT, true));
            String lightLine = light.values() + " · " + light.label();
            info.addView(ui.line(report == null ? ip + " · Light —" : lightLine,
                    wide ? 9 : 10, report == null ? MUTED : lightColor, false), margin(-1, -2, 4, 0));
            card.addView(info, new LinearLayout.LayoutParams(0, -2, 1));
            card.addView(ui.line(err ? "! ERROR" : rec ? "● REC" : peer.ready ? "● READY" : "CAMERA", wide ? 9 : 11, color, true));
            card.setOnClickListener(v -> details(name, "Address: " + ip + "\n" + rtt + "\nStatus: " + peer.status
                    + "\n\n" + light.detail()
                    + "\n\nRemote live thumbnails and battery telemetry are not transmitted by this prototype.", err));
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(wide ? ui.dp(260) : -1, ui.dp(wide ? 48 : 68));
            if (peerContainer.getChildCount() > 0) { if (wide) lp.leftMargin = ui.dp(7); else lp.topMargin = ui.dp(6); }
            peerContainer.addView(card, lp);
        }
    }

    private boolean active() {
        return phase == Phase.ARMING || phase == Phase.READY || phase == Phase.SCHEDULED || phase == Phase.RECORDING || phase == Phase.STOPPING;
    }
    private boolean localViewEnabled() {
        return role == NetworkCoordinator.Role.CAMERA || localEnabled;
    }
    private void renderNetwork() {
        boolean cameraVisible = role == NetworkCoordinator.Role.CAMERA
                && networkDetail.toLowerCase(Locale.US).contains("camera visible");
        networkTile.show(networkError ? "Error" : cameraVisible ? "Visible" : peers().isEmpty() ? "Searching" : "Connected",
                "Local Wi-Fi", networkError ? RED : CYAN, networkError);
    }
    private void render() {
        if (destroyed || preview == null) return;
        boolean controller = role == NetworkCoordinator.Role.CONTROLLER;
        boolean localView = localViewEnabled();
        boolean hasLocalFrame = localView && cameraEngine != null;
        controllerActions.setVisibility(controller ? View.VISIBLE : View.GONE);
        cameraActions.setVisibility(controller ? View.GONE : View.VISIBLE);
        recordRow.setVisibility(controller ? View.VISIBLE : View.GONE);
        recordSwitch.setEnabled(!active());
        if (controllerTab != null) {
            controllerTab.setBackground(ui.touch(controller ? 0xff075265 : PANEL, controller ? 0xff073047 : PANEL, controller ? CYAN : PANEL, 7));
            cameraTab.setBackground(ui.touch(controller ? PANEL : 0xff075265, controller ? PANEL : 0xff073047, controller ? PANEL : CYAN, 7));
        }
        roleTile.show(controller ? "Controller" : "Camera", "Tap to change", CYAN, false);
        renderNetwork();
        String state = phase == Phase.READY ? "Ready" : phase == Phase.SCHEDULED ? "Countdown" : phase == Phase.RECORDING ? "Recording"
                : phase == Phase.STOPPING ? "Saving" : phase == Phase.SAVED ? "Saved" : phase == Phase.ARMING ? "Arming" : phase == Phase.ERROR ? "Error" : "Idle";
        boolean fpsProblem = phase == Phase.SAVED && lastFps > 0 && lastFps < 110;
        int color = phase == Phase.ERROR || fpsProblem ? RED : phase == Phase.READY || phase == Phase.SAVED ? GREEN : CYAN;
        cameraTile.show(localView ? state : "Remote only", !localView ? "Local camera off" : phase == Phase.SAVED && lastFps > 0
                ? String.format(Locale.US, "%.1f fps measured", lastFps) : "FHD120 · AF", color, phase == Phase.ERROR || fpsProblem);
        sessionTile.show(state, sessionId == null ? "No session" : sessionId.substring(Math.max(0, sessionId.length()-6)), color, phase == Phase.ERROR);
        boolean showLocalFrame = hasLocalFrame && (phase == Phase.READY || phase == Phase.SCHEDULED
                || phase == Phase.RECORDING || phase == Phase.STOPPING || phase == Phase.SAVED);
        previewPlaceholder.setVisibility(showLocalFrame ? View.GONE : View.VISIBLE);
        previewState.setText(!localView ? "CONTROLLER ONLY" : phase == Phase.RECORDING ? "● RECORDING"
                : phase == Phase.READY || phase == Phase.SCHEDULED ? "● LIVE · PRE-ROLL"
                : phase == Phase.ARMING ? "OPENING CAMERA" : phase == Phase.SAVED ? "LAST FRAME" : "PREVIEW OFF");
        previewState.setTextColor(phase == Phase.RECORDING ? RED : phase == Phase.READY ? GREEN : TEXT);
        fitButton.setText(preview.isFill() ? "FILL" : "FIT");
        previewCaption.setText(!localView ? "Local video disabled" : preview.isFill() ? "Cropped preview · recording unchanged" : "Full frame · auto exposure");
        armButton.setEnabled(!active());
        startButton.setEnabled(phase == Phase.READY);
        stopButton.setEnabled(active() && phase != Phase.STOPPING);
        armButton.setAlpha(active() ? 0.55f : 1f);
        startButton.setAlpha(phase == Phase.READY ? 1f : 0.65f);
        stopButton.setAlpha(active() ? 1f : 0.65f);
        preview.refreshTransform();
        renderPeers();
        updateLightIndicator();
    }

    private final Runnable ticker = new Runnable() {
        @Override public void run() {
            if (destroyed) return;
            if (phase == Phase.SCHEDULED) {
                double remaining = Math.max(0, (scheduledNs - SystemClock.elapsedRealtimeNanos()) / 1e9);
                previewState.setText(String.format(Locale.US, "START IN %.1f s%s", remaining, localViewEnabled() ? " · PRE-ROLL" : " · REMOTE"));
            } else if (phase == Phase.RECORDING) {
                long seconds = Math.max(0, (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000_000L);
                previewState.setText(String.format(Locale.US, localViewEnabled() ? "● REC  %02d:%02d" : "REMOTE REC  %02d:%02d", seconds/60, seconds%60));
            }
            updateLightTelemetry();
            main.postDelayed(this, 250);
        }
    };

    private static int lightColor(LightMonitor.Snapshot sample) {
        return !sample.known ? MUTED : sample.blurBand >= 2 ? RED : sample.warning() ? AMBER : GREEN;
    }

    private void updateLightTelemetry() {
        if (testMode) return;
        long now = SystemClock.elapsedRealtimeNanos();
        if (lastLightRefreshNs >= 0 && now-lastLightRefreshNs < LightMonitor.WINDOW_NS) return;
        lastLightRefreshNs = now;
        boolean live = localViewEnabled() && cameraEngine != null
                && (phase == Phase.ARMING || phase == Phase.READY || phase == Phase.SCHEDULED || phase == Phase.RECORDING);
        if (live) lightSnapshot = cameraEngine.getLightSnapshot(now);
        if (network != null) network.setLocalLight(live ? lightSnapshot : null);
        updateLightIndicator();
        if (role == NetworkCoordinator.Role.CONTROLLER) renderPeers(); // also expire remote telemetry
    }

    private void updateLightIndicator() {
        if (lightIndicator == null) return;
        boolean visible = localViewEnabled() && (active() || phase == Phase.SAVED);
        lightIndicator.setVisibility(visible ? View.VISIBLE : View.GONE);
        int color = lightColor(lightSnapshot);
        String label = lightSnapshot.label();
        if (!lightSnapshot.known && lightSnapshot.reason.toLowerCase(Locale.US).contains("stale"))
            label = "LIGHT DATA STALE";
        lightIndicator.setText((phase == Phase.SAVED ? "LAST · " : "") + label + "\n" + lightSnapshot.values());
        lightIndicator.setTextColor(color);
        lightIndicator.setBackground(ui.touch(0xee06131d, 0xee06131d, color, 7));
        lightIndicator.setContentDescription(label + ". " + lightSnapshot.values() + ". Tap for light details.");
        boolean captureError = phase == Phase.ERROR || (phase == Phase.SAVED && lastFps > 0 && lastFps < 110);
        boolean warning = visible && lightSnapshot.warning();
        cameraTile.badge.setVisibility(captureError || warning ? View.VISIBLE : View.GONE);
        int badgeColor = captureError ? RED : color;
        cameraTile.badge.setBackground(ui.panel(badgeColor, badgeColor, badgeColor, 9));
        if (visible && lightSnapshot.known) cameraTile.small.setText(lightSnapshot.values());
    }

    private void startNetwork() {
        if (testMode) return;
        if (network == null) network = new NetworkCoordinator(this, this);
        NetworkCoordinator.Role selected = role;
        io(() -> network.start(selected));
    }
    private void io(Runnable task) {
        if (!destroyed && !networkIo.isShutdown()) networkIo.execute(() -> {
            try { task.run(); } catch (Exception e) { onNetworkStatus("Error: " + e.getMessage()); }
        });
    }
    private List<NetworkCoordinator.Peer> peers() { return network == null ? Collections.emptyList() : network.getPeers(); }
    private void selectRole(NetworkCoordinator.Role newRole) {
        if (active()) { toast("Stop the session before changing role."); return; }
        if (role == newRole && network != null) return;
        role = newRole;
        getPreferences(MODE_PRIVATE).edit().putString("role", role.name()).apply();
        phase = Phase.IDLE; sessionId = null; networkError = false;
        startNetwork(); render();
    }
    private void armAll() {
        if (active()) return;
        List<NetworkCoordinator.Peer> peers = peers();
        if (!localEnabled && peers.isEmpty()) { toast("Choose a camera or enable local recording."); return; }
        if (localEnabled && checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.CAMERA}, CAMERA_PERMISSION); return;
        }
        participants.clear(); for (NetworkCoordinator.Peer peer : peers) participants.add(peer.id);
        sessionId = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
        phase = localEnabled ? Phase.ARMING : Phase.READY;
        lastFps = 0; videoUri = null; metadataPath = null;
        lightSnapshot = LightMonitor.Snapshot.unknown(sessionId, "Waiting for high-speed metadata");
        if (network != null) network.setLocalLight(null);
        sessionDetail = "Preparing cameras. Pre-roll is saved from ARM.";
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LOCKED);
        String id = sessionId;
        io(() -> { if (network != null) network.armAll(id); });
        if (localEnabled) {
            if (cameraEngine == null) cameraEngine = new CameraEngine(this, preview, this);
            cameraEngine.arm(id);
        }
        render();
    }
    private void startAll() {
        if (phase != Phase.READY) return;
        List<NetworkCoordinator.Peer> current = peers();
        Set<String> ids = new HashSet<>();
        for (NetworkCoordinator.Peer peer : current) {
            ids.add(peer.id);
            if (!peer.ready) { toast("Wait until all cameras are READY."); return; }
        }
        if (!ids.equals(participants)) { toast("The camera group changed. Stop and ARM again."); return; }
        scheduledNs = SystemClock.elapsedRealtimeNanos() + 3_000_000_000L;
        phase = Phase.SCHEDULED;
        long target = scheduledNs;
        io(() -> { if (network != null) network.startAll(target, 3000); });
        if (localEnabled && cameraEngine != null) cameraEngine.startAt(target);
        else main.postDelayed(() -> { if (phase == Phase.SCHEDULED) onCameraStarted(target); }, 3000);
        sessionDetail = "Synchronized start scheduled. This is not hardware genlock.";
        render();
    }
    private void stopAll() {
        if (!active() || phase == Phase.STOPPING) return;
        phase = Phase.STOPPING;
        sessionDetail = "STOP sent. Waiting for local file finalization.";
        io(() -> { if (network != null && role == NetworkCoordinator.Role.CONTROLLER) network.stopAll(); });
        if (cameraEngine != null && cameraEngine.getState() != CameraEngine.State.IDLE) cameraEngine.stop();
        else { phase = Phase.IDLE; sessionDetail = "STOP sent to remote cameras."; unlock(); }
        render();
    }
    private void unlock() { setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED); }

    @Override public void onCameraStatus(String status) {
        main.post(() -> { if (!destroyed) cameraDetail = status; });
    }
    @Override public void onCameraReady(String detail) {
        main.post(() -> {
            if (destroyed || phase == Phase.STOPPING) return;
            cameraDetail = detail; phase = Phase.READY; sessionDetail = "Cameras armed. Pre-roll is running; press START to mark the synchronized beginning.";
            render();
            if (role == NetworkCoordinator.Role.CAMERA) io(() -> network.sendReady(detail));
        });
    }
    @Override public void onCameraStarted(long localStartNs) {
        main.post(() -> {
            if (destroyed || phase == Phase.STOPPING) return;
            phase = Phase.RECORDING; startedNs = localStartNs; sessionDetail = "Recording. The saved video also contains pre-roll.";
            render();
            if (role == NetworkCoordinator.Role.CAMERA) io(() -> network.sendStarted(localStartNs));
        });
    }
    @Override public void onCameraStopped(String path, String metadata) {
        main.post(() -> {
            if (destroyed) return;
            if (cameraEngine != null) lightSnapshot = cameraEngine.getLightSnapshot(SystemClock.elapsedRealtimeNanos());
            if (network != null) network.setLocalLight(null);
            videoUri = path; metadataPath = metadata; lastFps = cameraEngine == null ? 0 : cameraEngine.getLastEncodedFps();
            phase = path == null ? Phase.ERROR : Phase.SAVED;
            sessionDetail = path == null ? "No video file was returned." : path.startsWith("content:") ? "Video published to Gallery / Movies / TKDPoomsae." : "Video saved in app storage; Gallery publication was not confirmed.";
            cameraDetail = String.format(Locale.US, "MP4: %.2f fps measured from timestamps.\nFrames: %d\n%s", lastFps,
                    cameraEngine == null ? 0 : cameraEngine.getLastEncodedFrameCount(), lastFps > 0 && lastFps < 110 ? "Warning: below the requested 120 fps." : "Capture completed.");
            render();
            if (role == NetworkCoordinator.Role.CAMERA) io(() -> network.sendStopped(path));
            unlock();
        });
    }
    @Override public void onCameraError(String error) {
        main.post(() -> {
            if (destroyed) return;
            cameraDetail = error; phase = Phase.ERROR; sessionDetail = "Camera error. Open the Camera status for details.";
            render();
            if (role == NetworkCoordinator.Role.CAMERA) io(() -> network.sendCameraError(error));
            unlock();
        });
    }
    @Override public void onPeersChanged() {
        main.post(() -> { if (!destroyed) { renderNetwork(); renderPeers(); } });
    }
    @Override public void onNetworkStatus(String status) {
        main.post(() -> {
            if (destroyed) return;
            networkDetail = status;
            String s = status.toLowerCase(Locale.US);
            networkError = s.contains("error") || s.contains("failed") || s.contains("warning") || s.contains("eperm");
            render();
        });
    }
    @Override public void onArmCommand(String id) {
        main.post(() -> {
            if (destroyed || role != NetworkCoordinator.Role.CAMERA) return;
            if (active()) return;
            if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                onCameraError("Camera permission is missing. Grant it on this phone."); return;
            }
            sessionId = id; phase = Phase.ARMING; lastFps = 0; videoUri = null; metadataPath = null;
            lightSnapshot = LightMonitor.Snapshot.unknown(id, "Waiting for high-speed metadata");
            if (network != null) network.setLocalLight(null);
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LOCKED);
            if (cameraEngine == null) cameraEngine = new CameraEngine(this, preview, this);
            cameraEngine.arm(id); render();
        });
    }
    @Override public void onStartCommand(long target, long fallbackMs) {
        main.post(() -> {
            if (destroyed || role != NetworkCoordinator.Role.CAMERA || phase != Phase.READY) return;
            scheduledNs = target > 0 ? target : SystemClock.elapsedRealtimeNanos() + fallbackMs * 1_000_000L;
            phase = Phase.SCHEDULED; cameraEngine.startAt(scheduledNs); render();
        });
    }
    @Override public void onStopCommand() { main.post(() -> { if (!destroyed && role == NetworkCoordinator.Role.CAMERA) stopAll(); }); }

    private void roleDialog() {
        new AlertDialog.Builder(this).setTitle("Device role")
                .setSingleChoiceItems(new String[]{"Controller — coordinate all cameras", "Camera — controlled by another phone"},
                        role == NetworkCoordinator.Role.CONTROLLER ? 0 : 1, (dialog, which) -> {
                            dialog.dismiss(); selectRole(which == 0 ? NetworkCoordinator.Role.CONTROLLER : NetworkCoordinator.Role.CAMERA);
                            if (which == 1 && checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
                                requestPermissions(new String[]{android.Manifest.permission.CAMERA}, CAMERA_PERMISSION);
                        }).setNegativeButton("Close", null).show();
    }
    private void menu() {
        new AlertDialog.Builder(this).setTitle("TKD MultiCam 1.8")
                .setItems(new String[]{"Change device role", "Preview: fit / cropped fill", "Open last saved video", "Diagnostics", "About capture"}, (d, n) -> {
                    if (n == 0) roleDialog();
                    if (n == 1) { preview.setFill(!preview.isFill()); render(); }
                    if (n == 2) openVideo();
                    if (n == 3) details("Diagnostics", networkDetail + "\n\n" + cameraDetail + "\n\n" + sessionDetail
                            + "\n\n" + getSharedPreferences("crash", MODE_PRIVATE).getString("last_crash", "No captured process crash."), networkError || phase == Phase.ERROR);
                    if (n == 4) details("Capture", "Camera 0 · 1080p120 target\nAutofocus · automatic exposure\n\nARM writes pre-roll. START records a synchronized time marker. STOP finalizes the MP4 and publishes it to Gallery.\n\nThis version keeps the verified v1.3 capture backend unchanged.\n\nPreview FIT shows the whole frame. FILL crops only the screen image. Neither stretches or changes the saved video.", false);
                }).setNegativeButton("Close", null).show();
    }
    private void details(String title, String message, boolean error) {
        TextView body = ui.text(message, 14, TEXT, false);
        body.setTextIsSelectable(true);
        body.setPadding(ui.dp(22), ui.dp(16), ui.dp(22), ui.dp(16));
        ScrollView scroll = new ScrollView(this); scroll.addView(body);
        new AlertDialog.Builder(this).setTitle((error ? "!  " : "") + title).setView(scroll)
                .setPositiveButton("Close", null).setNeutralButton("Copy", (d, w) -> {
                    ClipboardManager cm = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
                    cm.setPrimaryClip(ClipData.newPlainText(title, message));
                }).show();
    }
    private void openVideo() {
        if (active()) { toast("Stop the recording first."); return; }
        if (videoUri == null || !videoUri.startsWith("content:")) { toast("No Gallery video saved in this session."); return; }
        try { startActivity(new Intent(Intent.ACTION_VIEW).setDataAndType(Uri.parse(videoUri), "video/mp4").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)); }
        catch (Exception e) { details("Open video", e.toString(), true); }
    }
    private String batteryPercent() {
        Intent i = registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (i == null) return "—";
        int level = i.getIntExtra(BatteryManager.EXTRA_LEVEL, -1), scale = i.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
        return level < 0 || scale <= 0 ? "—" : Math.round(level * 100f / scale) + "%";
    }
    private String freeStorage() {
        try { File f = getExternalFilesDir(null); return String.format(Locale.US, "%.1f GB", new StatFs((f == null ? getFilesDir() : f).getPath()).getAvailableBytes() / 1e9); }
        catch (Exception e) { return "Unavailable"; }
    }
    private String wifiInfo() {
        try {
            ConnectivityManager cm = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
            NetworkCapabilities cap = cm.getNetworkCapabilities(cm.getActiveNetwork());
            if (cap == null || !cap.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) return "The active network is not Wi-Fi.";
            LinkProperties lp = cm.getLinkProperties(cm.getActiveNetwork());
            StringBuilder b = new StringBuilder("Wi-Fi active");
            if (lp != null) for (LinkAddress a : lp.getLinkAddresses()) if (a.getAddress() instanceof Inet4Address) b.append("\nIP: ").append(a.getAddress().getHostAddress());
            return b.toString();
        } catch (Exception e) { return "Wi-Fi details unavailable."; }
    }
    private void toast(String text) { Toast.makeText(this, text, Toast.LENGTH_LONG).show(); }
    private LinearLayout.LayoutParams margin(int width, int height, int top, int bottom) { return margin(width, height, top, bottom, 0); }
    private LinearLayout.LayoutParams margin(int width, int height, int top, int bottom, float weight) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(width, height, weight);
        lp.setMargins(0, ui.dp(top), 0, ui.dp(bottom)); return lp;
    }
    @Override public void onRequestPermissionsResult(int request, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (request == CAMERA_PERMISSION && (results.length == 0 || results[0] != PackageManager.PERMISSION_GRANTED)) toast("Camera permission is required only on devices that record.");
    }
    @Override protected void onSaveInstanceState(Bundle out) {
        out.putString("role", role.name()); out.putBoolean("local", localEnabled); out.putBoolean("fill", preview.isFill());
        super.onSaveInstanceState(out);
    }
    @Override protected void onStop() {
        if (!isChangingConfigurations() && active()) stopAll();
        super.onStop();
    }
    @Override protected void onDestroy() {
        destroyed = true; main.removeCallbacksAndMessages(null);
        if (network != null && !networkIo.isShutdown()) networkIo.execute(network::stop);
        networkIo.shutdown();
        if (cameraEngine != null) cameraEngine.shutdown();
        super.onDestroy();
    }
}
