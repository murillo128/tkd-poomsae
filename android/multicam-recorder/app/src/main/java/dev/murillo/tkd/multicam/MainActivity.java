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
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.graphics.Typeface;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.TextureView;
import android.view.Surface;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.text.TextUtils;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public final class MainActivity extends Activity
        implements CameraEngine.Listener, NetworkCoordinator.Listener {

    private static final int CAMERA_PERMISSION_REQUEST = 2001;

    private static final int BG = Color.rgb(5, 15, 27);
    private static final int PANEL = Color.rgb(10, 29, 46);
    private static final int PANEL_ALT = Color.rgb(13, 38, 58);
    private static final int CYAN = Color.rgb(22, 218, 239);
    private static final int CYAN_SOFT = Color.rgb(37, 151, 180);
    private static final int GREEN = Color.rgb(29, 225, 154);
    private static final int RED = Color.rgb(255, 72, 86);
    private static final int TEXT = Color.rgb(246, 249, 252);
    private static final int MUTED = Color.rgb(156, 178, 201);
    private static final int BORDER = Color.rgb(31, 83, 112);

    private boolean landscapeUi;

    private TextureView textureView;
    private TextView roleText;
    private TextView networkText;
    private TextView cameraText;
    private TextView sessionText;
    private TextView peerTitle;
    private TextView previewTelemetry;
    private LinearLayout peersContainer;

    private LinearLayout controllerPanel;
    private LinearLayout cameraPanel;
    private LinearLayout rootRoleChooser;
    private Switch recordLocal;
    private Button controllerButton;
    private Button cameraButton;

    private CameraEngine cameraEngine;
    private NetworkCoordinator network;

    private NetworkCoordinator.Role role;
    private boolean localReady;
    private String currentSessionId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        configureWindow();
        installCrashRecorder();
        buildUi();

        if (checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
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

        if (savedInstanceState != null) {
            String restored =
                    savedInstanceState.getString("role");
            if ("CONTROLLER".equals(restored)) {
                chooseController();
            } else if ("CAMERA".equals(restored)) {
                chooseCamera();
            }
        }
    }

    private void configureWindow() {
        Window window = getWindow();
        window.setStatusBarColor(BG);
        window.setNavigationBarColor(BG);
        if (BuildCompat.atLeast23()) {
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
        SharedPreferences prefs =
                getSharedPreferences("crash", MODE_PRIVATE);

        String crash =
                prefs.getString("last_crash", null);

        if (crash == null || crash.isEmpty()) return;

        prefs.edit().remove("last_crash").apply();
        networkText.setText("Previous crash captured");

        Button copyCrash =
                actionButton("COPY CRASH", CYAN_SOFT);

        copyCrash.setOnClickListener(v -> {
            ClipboardManager cm =
                    (ClipboardManager)
                            getSystemService(CLIPBOARD_SERVICE);

            cm.setPrimaryClip(
                    ClipData.newPlainText(
                            "TKD MultiCam crash",
                            crash));

            Toast.makeText(
                    this,
                    "Crash copied",
                    Toast.LENGTH_SHORT).show();
        });

        controllerPanel.addView(
                copyCrash,
                matchWrap(0, dp(6)));
    }

    private void buildUi() {
        landscapeUi =
                getResources()
                        .getConfiguration()
                        .orientation
                        == Configuration.ORIENTATION_LANDSCAPE;

        rootRoleChooser = null;

        LinearLayout root =
                new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);

        int topInset =
                systemDimen("status_bar_height");

        int navRightInset =
                landscapeUi
                        ? systemDimen("navigation_bar_width")
                        : 0;

        root.setPadding(
                dp(landscapeUi ? 10 : 18),
                topInset + dp(landscapeUi ? 4 : 8),
                navRightInset + dp(landscapeUi ? 12 : 18),
                dp(landscapeUi ? 8 : 18));

        root.addView(
                buildHeader(landscapeUi),
                matchWrap(0, dp(landscapeUi ? 5 : 10)));

        if (landscapeUi) {
            buildLandscape(root);
            setContentView(root);
        } else {
            buildPortrait(root);

            ScrollView scroll =
                    new ScrollView(this);
            scroll.setFillViewport(true);
            scroll.setBackgroundColor(BG);
            scroll.addView(root);
            setContentView(scroll);
        }
    }

    private View buildHeader(boolean landscape) {
        LinearLayout header =
                new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);

        ImageView icon =
                new ImageView(this);
        icon.setImageResource(
                R.drawable.tkd_multicam_icon);
        icon.setScaleType(
                ImageView.ScaleType.CENTER_CROP);
        icon.setBackground(
                rounded(
                        PANEL_ALT,
                        dp(12),
                        BORDER,
                        dp(1)));

        int iconSize =
                dp(landscape ? 34 : 48);

        header.addView(
                icon,
                new LinearLayout.LayoutParams(
                        iconSize,
                        iconSize));

        LinearLayout titleBox =
                new LinearLayout(this);
        titleBox.setOrientation(
                LinearLayout.VERTICAL);
        titleBox.setPadding(
                dp(landscape ? 8 : 12),
                0,
                0,
                0);

        TextView title =
                new TextView(this);
        title.setText("TKD MultiCam 120");
        title.setTextColor(TEXT);
        title.setTextSize(landscape ? 17 : 23);
        title.setTypeface(
                Typeface.DEFAULT_BOLD);
        titleBox.addView(title);

        TextView subtitle =
                new TextView(this);
        subtitle.setText(
                "SYNCHRONIZED POOMSAE RECORDING");
        subtitle.setTextColor(CYAN);
        subtitle.setTextSize(landscape ? 7 : 10);
        subtitle.setLetterSpacing(0.14f);
        titleBox.addView(subtitle);

        header.addView(
                titleBox,
                new LinearLayout.LayoutParams(
                        0,
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        1f));

        LinearLayout chooser =
                new LinearLayout(this);
        chooser.setOrientation(
                LinearLayout.HORIZONTAL);

        controllerButton =
                roleButton("CONTROLLER");
        controllerButton.setOnClickListener(
                v -> chooseController());

        chooser.addView(
                controllerButton,
                new LinearLayout.LayoutParams(
                        landscape ? dp(108) : 0,
                        dp(landscape ? 34 : 48),
                        landscape ? 0f : 1f));

        cameraButton =
                roleButton("CAMERA");
        cameraButton.setOnClickListener(
                v -> chooseCamera());

        LinearLayout.LayoutParams cameraLp =
                new LinearLayout.LayoutParams(
                        landscape ? dp(88) : 0,
                        dp(landscape ? 34 : 48),
                        landscape ? 0f : 1f);

        cameraLp.setMargins(dp(6), 0, 0, 0);
        chooser.addView(cameraButton, cameraLp);

        if (landscape) {
            header.addView(chooser);
        } else {
            rootRoleChooser = chooser;
        }

        return header;
    }

    private void buildPortrait(LinearLayout root) {
        if (rootRoleChooser != null) {
            root.addView(
                    rootRoleChooser,
                    matchWrap(0, dp(10)));
        }

        root.addView(
                buildPreview(portraitPreviewHeight()),
                matchWrap(0, dp(10)));

        LinearLayout statusGrid =
                new LinearLayout(this);
        statusGrid.setOrientation(
                LinearLayout.VERTICAL);

        LinearLayout row1 =
                new LinearLayout(this);
        row1.setOrientation(
                LinearLayout.HORIZONTAL);

        roleText =
                statusCard(
                        row1,
                        "ROLE",
                        "Choose role");

        networkText =
                statusCard(
                        row1,
                        "NETWORK",
                        "Not started");

        statusGrid.addView(
                row1,
                matchWrap(0, dp(6)));

        LinearLayout row2 =
                new LinearLayout(this);
        row2.setOrientation(
                LinearLayout.HORIZONTAL);

        cameraText =
                statusCard(
                        row2,
                        "CAMERA",
                        "Idle");

        sessionText =
                statusCard(
                        row2,
                        "SESSION",
                        "No active session");

        statusGrid.addView(row2);

        root.addView(
                statusGrid,
                matchWrap(0, dp(10)));

        buildControllerPanel();
        root.addView(
                controllerPanel,
                matchWrap(0, dp(10)));

        buildCameraPanel();
        root.addView(
                cameraPanel,
                matchWrap(0, dp(10)));

        root.addView(
                buildPeersPanel(),
                matchWrap(0, 0));
    }

    private void buildLandscape(LinearLayout root) {
        LinearLayout body =
                new LinearLayout(this);
        body.setOrientation(
                LinearLayout.HORIZONTAL);

        FrameLayout preview =
                (FrameLayout)
                        buildPreview(-1);

        body.addView(
                preview,
                new LinearLayout.LayoutParams(
                        0,
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        1.55f));

        LinearLayout commandColumn =
                new LinearLayout(this);
        commandColumn.setOrientation(
                LinearLayout.VERTICAL);
        commandColumn.setPadding(
                dp(10),
                0,
                0,
                0);

        LinearLayout statusGrid =
                new LinearLayout(this);
        statusGrid.setOrientation(
                LinearLayout.VERTICAL);

        LinearLayout statusRow1 =
                new LinearLayout(this);
        statusRow1.setOrientation(
                LinearLayout.HORIZONTAL);

        roleText =
                statusCard(
                        statusRow1,
                        "ROLE",
                        "Choose role");

        networkText =
                statusCard(
                        statusRow1,
                        "NETWORK",
                        "Not started");

        statusGrid.addView(
                statusRow1,
                matchWrap(0, dp(5)));

        LinearLayout statusRow2 =
                new LinearLayout(this);
        statusRow2.setOrientation(
                LinearLayout.HORIZONTAL);

        cameraText =
                statusCard(
                        statusRow2,
                        "CAMERA",
                        "Idle");

        sessionText =
                statusCard(
                        statusRow2,
                        "SESSION",
                        "No active session");

        statusGrid.addView(statusRow2);

        commandColumn.addView(
                statusGrid,
                matchWrap(0, dp(6)));

        buildControllerPanel();
        commandColumn.addView(
                controllerPanel,
                matchWrap(0, dp(6)));

        buildCameraPanel();
        commandColumn.addView(
                cameraPanel,
                matchWrap(0, dp(6)));

        commandColumn.addView(
                buildPeersPanel(),
                new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        0,
                        1f));

        body.addView(
                commandColumn,
                new LinearLayout.LayoutParams(
                        0,
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        1f));

        LinearLayout.LayoutParams bodyLp =
                new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        0,
                        1f);

        bodyLp.setMargins(
                0,
                dp(3),
                0,
                0);

        root.addView(body, bodyLp);
    }

    private View buildPreview(int height) {
        FrameLayout frame =
                new FrameLayout(this);

        frame.setBackground(
                gradientPanel(
                        Color.rgb(2, 10, 18),
                        Color.rgb(4, 22, 34),
                        dp(18),
                        CYAN_SOFT));

        textureView =
                new TextureView(this);

        textureView.addOnLayoutChangeListener(
                (v, left, top, right, bottom,
                 oldLeft, oldTop, oldRight, oldBottom) ->
                        updatePreviewTransform());

        textureView.post(this::updatePreviewTransform);

        int previewHeight =
                height > 0
                        ? height
                        : FrameLayout.LayoutParams.MATCH_PARENT;

        FrameLayout.LayoutParams previewLp =
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        previewHeight);

        previewLp.setMargins(
                dp(3),
                dp(3),
                dp(3),
                dp(3));

        frame.addView(
                textureView,
                previewLp);

        TextView live =
                pill(
                        "●  LIVE PREVIEW",
                        GREEN);

        FrameLayout.LayoutParams liveLp =
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.WRAP_CONTENT,
                        FrameLayout.LayoutParams.WRAP_CONTENT);

        liveLp.gravity =
                Gravity.TOP | Gravity.START;

        liveLp.setMargins(
                dp(14),
                dp(12),
                0,
                0);

        frame.addView(live, liveLp);

        TextView mode =
                pill(
                        "FHD · 120 FPS · CAM 0 · 1×",
                        CYAN);

        FrameLayout.LayoutParams modeLp =
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.WRAP_CONTENT,
                        FrameLayout.LayoutParams.WRAP_CONTENT);

        modeLp.gravity =
                Gravity.TOP | Gravity.END;

        modeLp.setMargins(
                0,
                dp(12),
                dp(14),
                0);

        frame.addView(mode, modeLp);

        previewTelemetry =
                pill(
                        "IDLE · AF CONTINUOUS · 1/500",
                        CYAN_SOFT);

        previewTelemetry.setTextSize(
                landscapeUi ? 8 : 10);

        FrameLayout.LayoutParams telemetryLp =
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.WRAP_CONTENT,
                        FrameLayout.LayoutParams.WRAP_CONTENT);

        telemetryLp.gravity =
                Gravity.BOTTOM | Gravity.START;

        telemetryLp.setMargins(
                dp(14),
                0,
                0,
                dp(12));

        frame.addView(
                previewTelemetry,
                telemetryLp);

        return frame;
    }

    private void buildControllerPanel() {
        controllerPanel =
                new LinearLayout(this);

        controllerPanel.setOrientation(
                LinearLayout.VERTICAL);

        controllerPanel.setPadding(
                dp(landscapeUi ? 8 : 14),
                dp(landscapeUi ? 7 : 14),
                dp(landscapeUi ? 8 : 14),
                dp(landscapeUi ? 7 : 14));

        controllerPanel.setBackground(
                gradientPanel(
                        PANEL,
                        Color.rgb(11, 39, 58),
                        dp(18),
                        BORDER));

        recordLocal =
                new Switch(this);

        recordLocal.setText(
                "  Record on this controller too");

        recordLocal.setTextColor(TEXT);
        recordLocal.setTextSize(
                landscapeUi ? 10 : 15);
        recordLocal.setChecked(true);

        controllerPanel.addView(
                recordLocal,
                matchWrap(
                        0,
                        dp(landscapeUi ? 5 : 12)));

        LinearLayout row1 =
                new LinearLayout(this);
        row1.setOrientation(
                LinearLayout.HORIZONTAL);

        Button discover =
                actionButton(
                        "⌁  DISCOVER",
                        CYAN_SOFT);

        discover.setOnClickListener(v -> {
            if (network != null) {
                network.discoverNow();
                updatePeers();
            }
        });

        row1.addView(
                discover,
                weightedButton());

        Button arm =
                actionButton(
                        "◉  ARM ALL",
                        CYAN);

        arm.setOnClickListener(
                v -> armAll());

        LinearLayout.LayoutParams armLp =
                weightedButton();

        armLp.setMargins(
                dp(7),
                0,
                0,
                0);

        row1.addView(arm, armLp);

        controllerPanel.addView(
                row1,
                matchWrap(
                        0,
                        dp(landscapeUi ? 5 : 8)));

        LinearLayout row2 =
                new LinearLayout(this);
        row2.setOrientation(
                LinearLayout.HORIZONTAL);

        Button start =
                actionButton(
                        "▶  START +3S",
                        GREEN);

        start.setOnClickListener(
                v -> startAll());

        row2.addView(
                start,
                weightedButton());

        Button stop =
                actionButton(
                        "■  STOP ALL",
                        RED);

        stop.setOnClickListener(
                v -> stopAll());

        LinearLayout.LayoutParams stopLp =
                weightedButton();

        stopLp.setMargins(
                dp(7),
                0,
                0,
                0);

        row2.addView(
                stop,
                stopLp);

        controllerPanel.addView(row2);
        controllerPanel.setVisibility(View.GONE);
    }

    private void buildCameraPanel() {
        cameraPanel =
                new LinearLayout(this);

        cameraPanel.setOrientation(
                LinearLayout.VERTICAL);

        cameraPanel.setPadding(
                dp(landscapeUi ? 8 : 16),
                dp(landscapeUi ? 8 : 16),
                dp(landscapeUi ? 8 : 16),
                dp(landscapeUi ? 8 : 16));

        cameraPanel.setBackground(
                gradientPanel(
                        PANEL,
                        Color.rgb(11, 39, 58),
                        dp(18),
                        BORDER));

        TextView title =
                new TextView(this);
        title.setText("CAMERA NODE");
        title.setTextColor(CYAN);
        title.setTextSize(
                landscapeUi ? 10 : 12);
        title.setTypeface(
                Typeface.DEFAULT_BOLD);
        cameraPanel.addView(title);

        TextView help =
                new TextView(this);

        help.setText(
                "Visible on local Wi‑Fi. Keep this phone mounted and the preview unobstructed.");

        help.setTextColor(MUTED);
        help.setTextSize(
                landscapeUi ? 9 : 14);

        help.setPadding(
                0,
                dp(landscapeUi ? 3 : 6),
                0,
                dp(landscapeUi ? 6 : 14));

        cameraPanel.addView(help);

        Button emergency =
                actionButton(
                        "■  LOCAL EMERGENCY STOP",
                        RED);

        emergency.setOnClickListener(v -> {
            if (cameraEngine != null) {
                cameraEngine.stop();
            }
        });

        cameraPanel.addView(emergency);
        cameraPanel.setVisibility(View.GONE);
    }

    private View buildPeersPanel() {
        LinearLayout panel =
                new LinearLayout(this);

        panel.setOrientation(
                LinearLayout.VERTICAL);

        panel.setPadding(
                dp(landscapeUi ? 8 : 14),
                dp(landscapeUi ? 6 : 12),
                dp(landscapeUi ? 8 : 14),
                dp(landscapeUi ? 6 : 14));

        panel.setBackground(
                gradientPanel(
                        PANEL,
                        Color.rgb(9, 35, 54),
                        dp(18),
                        BORDER));

        peerTitle =
                new TextView(this);

        peerTitle.setText(
                "DISCOVERED CAMERAS");

        peerTitle.setTextColor(CYAN);
        peerTitle.setTextSize(
                landscapeUi ? 10 : 14);

        peerTitle.setTypeface(
                Typeface.DEFAULT_BOLD);

        panel.addView(peerTitle);

        peersContainer =
                new LinearLayout(this);

        peersContainer.setOrientation(
                LinearLayout.VERTICAL);

        peersContainer.setPadding(
                0,
                dp(landscapeUi ? 3 : 8),
                0,
                0);

        TextView empty =
                new TextView(this);

        empty.setText(
                "No cameras yet. Put other phones in CAMERA mode on this Wi‑Fi.");

        empty.setTextColor(MUTED);
        empty.setTextSize(
                landscapeUi ? 9 : 13);

        peersContainer.addView(empty);
        panel.addView(peersContainer);

        return panel;
    }

    private TextView statusCard(
            LinearLayout parent,
            String label,
            String initial) {

        LinearLayout card =
                new LinearLayout(this);

        card.setOrientation(
                LinearLayout.VERTICAL);

        card.setPadding(
                dp(landscapeUi ? 6 : 10),
                dp(landscapeUi ? 4 : 9),
                dp(landscapeUi ? 6 : 10),
                dp(landscapeUi ? 4 : 9));

        card.setBackground(
                gradientPanel(
                        PANEL_ALT,
                        Color.rgb(12, 43, 64),
                        dp(14),
                        BORDER));

        TextView l =
                new TextView(this);

        l.setText(label);
        l.setTextColor(CYAN);
        l.setTextSize(
                landscapeUi ? 7 : 10);

        l.setTypeface(
                Typeface.DEFAULT_BOLD);

        card.addView(l);

        TextView value =
                new TextView(this);

        value.setText(initial);
        value.setTextColor(TEXT);
        value.setTextSize(
                landscapeUi ? 9 : 12);

        value.setMinLines(2);
        value.setMaxLines(2);
        value.setEllipsize(
                TextUtils.TruncateAt.END);
        value.setGravity(
                Gravity.TOP | Gravity.START);

        card.addView(value);

        LinearLayout.LayoutParams lp =
                new LinearLayout.LayoutParams(
                        0,
                        dp(landscapeUi ? 56 : 74),
                        1f);

        if (parent.getChildCount() > 0) {
            lp.setMargins(
                    dp(5),
                    0,
                    0,
                    0);
        }

        parent.addView(card, lp);
        return value;
    }

    private TextView pill(
            String text,
            int accent) {

        TextView tv =
                new TextView(this);

        tv.setText(text);
        tv.setTextColor(TEXT);
        tv.setTextSize(
                landscapeUi ? 8 : 11);

        tv.setTypeface(
                Typeface.DEFAULT_BOLD);

        tv.setPadding(
                dp(10),
                dp(5),
                dp(10),
                dp(5));

        tv.setBackground(
                rounded(
                        Color.argb(
                                226,
                                7,
                                21,
                                34),
                        dp(20),
                        accent,
                        dp(1)));

        return tv;
    }

    private Button roleButton(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setTextColor(TEXT);
        b.setTextSize(landscapeUi ? 9 : 12);
        b.setTypeface(Typeface.DEFAULT_BOLD);
        b.setAllCaps(false);
        b.setBackground(
                gradientPanel(
                        PANEL_ALT,
                        Color.rgb(12, 43, 64),
                        dp(14),
                        BORDER));
        return b;
    }

    private Button actionButton(
            String text,
            int accent) {

        Button b =
                new Button(this);

        b.setText(text);
        b.setTextColor(TEXT);
        b.setTextSize(
                landscapeUi ? 9 : 13);

        b.setTypeface(
                Typeface.DEFAULT_BOLD);

        b.setAllCaps(false);
        b.setMinHeight(
                dp(landscapeUi ? 36 : 54));

        int start;
        int end;

        if (accent == GREEN) {
            start = Color.rgb(0, 143, 117);
            end = Color.rgb(0, 86, 91);
        } else if (accent == RED) {
            start = Color.rgb(154, 35, 55);
            end = Color.rgb(86, 25, 43);
        } else if (accent == CYAN) {
            start = Color.rgb(0, 116, 145);
            end = Color.rgb(9, 66, 91);
        } else {
            start = Color.rgb(14, 89, 115);
            end = Color.rgb(11, 54, 77);
        }

        b.setBackground(
                gradientPanel(
                        start,
                        end,
                        dp(14),
                        accent));

        return b;
    }

    private GradientDrawable rounded(
            int fill,
            int radius,
            int stroke,
            int strokeWidth) {

        GradientDrawable g =
                new GradientDrawable();

        g.setColor(fill);
        g.setCornerRadius(radius);

        if (strokeWidth > 0) {
            g.setStroke(
                    strokeWidth,
                    stroke);
        }

        return g;
    }

    private GradientDrawable gradientPanel(
            int startColor,
            int endColor,
            int radius,
            int strokeColor) {

        GradientDrawable g =
                new GradientDrawable(
                        GradientDrawable.Orientation.TL_BR,
                        new int[]{
                                startColor,
                                endColor
                        });

        g.setCornerRadius(radius);
        g.setStroke(dp(1), strokeColor);

        return g;
    }

    private int mix(
            int a,
            int b,
            float amount) {

        float t =
                Math.max(
                        0f,
                        Math.min(
                                1f,
                                amount));

        return Color.rgb(
                Math.round(
                        Color.red(a)
                                + (Color.red(b) - Color.red(a)) * t),
                Math.round(
                        Color.green(a)
                                + (Color.green(b) - Color.green(a)) * t),
                Math.round(
                        Color.blue(a)
                                + (Color.blue(b) - Color.blue(a)) * t));
    }

    private LinearLayout.LayoutParams weightedButton() {
        return new LinearLayout.LayoutParams(
                0,
                dp(landscapeUi ? 38 : 58),
                1f);
    }

    private LinearLayout.LayoutParams matchWrap(
            int top,
            int bottom) {

        LinearLayout.LayoutParams lp =
                new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT);

        lp.setMargins(
                0,
                top,
                0,
                bottom);

        return lp;
    }

    private void updatePreviewTransform() {
        TextureView preview = textureView;
        if (preview == null) return;

        int viewWidth = preview.getWidth();
        int viewHeight = preview.getHeight();
        if (viewWidth <= 0 || viewHeight <= 0) return;

        int rotation =
                getDisplay() == null
                        ? Surface.ROTATION_0
                        : getDisplay().getRotation();

        Matrix matrix = new Matrix();
        RectF viewRect =
                new RectF(
                        0f,
                        0f,
                        viewWidth,
                        viewHeight);

        float centerX =
                viewRect.centerX();
        float centerY =
                viewRect.centerY();

        // Camera2Basic-style display transform. Portrait is intentionally left
        // unrotated: that is the orientation Samsung already presents correctly.
        // Landscape gets a display rotation plus a UNIFORM scale, so geometry is
        // never stretched and vertical lines stay vertical.
        if (rotation == Surface.ROTATION_90
                || rotation == Surface.ROTATION_270) {

            RectF bufferRect =
                    new RectF(
                            0f,
                            0f,
                            CameraEngine.HEIGHT,
                            CameraEngine.WIDTH);

            bufferRect.offset(
                    centerX - bufferRect.centerX(),
                    centerY - bufferRect.centerY());

            matrix.setRectToRect(
                    viewRect,
                    bufferRect,
                    Matrix.ScaleToFit.FILL);

            float scale =
                    Math.max(
                            (float) viewHeight
                                    / CameraEngine.HEIGHT,
                            (float) viewWidth
                                    / CameraEngine.WIDTH);

            matrix.postScale(
                    scale,
                    scale,
                    centerX,
                    centerY);

            float degrees =
                    rotation == Surface.ROTATION_90
                            ? -90f
                            : 90f;

            matrix.postRotate(
                    degrees,
                    centerX,
                    centerY);
        } else if (rotation == Surface.ROTATION_180) {
            matrix.postRotate(
                    180f,
                    centerX,
                    centerY);
        }

        preview.setTransform(matrix);
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);

        if (hasFocus && textureView != null) {
            textureView.post(
                    this::updatePreviewTransform);
        }
    }

    private int portraitPreviewHeight() {
        int width =
                getResources()
                        .getDisplayMetrics()
                        .widthPixels
                        - dp(36);

        return Math.round(
                width * 9f / 16f);
    }

    private int systemDimen(String name) {
        int id =
                getResources()
                        .getIdentifier(
                                name,
                                "dimen",
                                "android");

        return id > 0
                ? getResources()
                        .getDimensionPixelSize(id)
                : 0;
    }

    private int dp(int value) {
        return Math.round(
                value
                        * getResources()
                                .getDisplayMetrics()
                                .density);
    }

    private void ensureCoreObjects() {
        if (cameraEngine == null) {
            cameraEngine =
                    new CameraEngine(
                            this,
                            textureView,
                            this);
        }

        if (network == null) {
            network =
                    new NetworkCoordinator(
                            this,
                            this);
        }
    }

    private void chooseController() {
        try {
            role =
                    NetworkCoordinator.Role.CONTROLLER;

            ensureCoreObjects();

            roleText.setText("CONTROLLER");
            controllerPanel.setVisibility(View.VISIBLE);
            cameraPanel.setVisibility(View.GONE);
            styleRoleButtons();

            localReady = false;
            currentSessionId = null;

            network.start(
                    NetworkCoordinator.Role.CONTROLLER);

            updatePeers();
        } catch (Throwable t) {
            handleUiFailure(
                    "Controller init",
                    t);
        }
    }

    private void chooseCamera() {
        try {
            role =
                    NetworkCoordinator.Role.CAMERA;

            ensureCoreObjects();

            roleText.setText("CAMERA");
            controllerPanel.setVisibility(View.GONE);
            cameraPanel.setVisibility(View.VISIBLE);
            styleRoleButtons();

            localReady = false;
            currentSessionId = null;

            network.start(
                    NetworkCoordinator.Role.CAMERA);

            cameraText.setText(
                    "Waiting for controller ARM");
        } catch (Throwable t) {
            handleUiFailure(
                    "Camera init",
                    t);
        }
    }

    private void styleRoleButtons() {
        if (controllerButton == null
                || cameraButton == null) {
            return;
        }

        boolean controller =
                role
                        == NetworkCoordinator.Role.CONTROLLER;

        controllerButton.setBackground(
                gradientPanel(
                        controller
                                ? Color.rgb(0, 112, 137)
                                : PANEL_ALT,
                        controller
                                ? Color.rgb(5, 68, 91)
                                : Color.rgb(12, 43, 64),
                        dp(14),
                        controller
                                ? CYAN
                                : BORDER));

        cameraButton.setBackground(
                gradientPanel(
                        controller
                                ? PANEL_ALT
                                : Color.rgb(0, 112, 137),
                        controller
                                ? Color.rgb(12, 43, 64)
                                : Color.rgb(5, 68, 91),
                        dp(14),
                        controller
                                ? BORDER
                                : CYAN));
    }

    private void handleUiFailure(
            String where,
            Throwable t) {

        StringWriter sw =
                new StringWriter();

        t.printStackTrace(
                new PrintWriter(sw));

        String full =
                where + "\n" + sw;

        networkText.setText(
                "ERROR · "
                        + t.getClass()
                                .getSimpleName());

        ClipboardManager cm =
                (ClipboardManager)
                        getSystemService(
                                CLIPBOARD_SERVICE);

        cm.setPrimaryClip(
                ClipData.newPlainText(
                        "TKD MultiCam error",
                        full));

        Toast.makeText(
                this,
                "Error copied to clipboard",
                Toast.LENGTH_LONG).show();
    }

    private void armAll() {
        if (!ensureCameraPermission()) return;

        lockCurrentOrientation();

        currentSessionId =
                new SimpleDateFormat(
                        "yyyyMMdd-HHmmss",
                        Locale.US)
                        .format(new Date());

        sessionText.setText(
                currentSessionId + " · ARMING");

        localReady =
                !recordLocal.isChecked();

        List<NetworkCoordinator.Peer> peers =
                network.getPeers();

        if (peers.isEmpty()
                && !recordLocal.isChecked()) {

            Toast.makeText(
                    this,
                    "No remote cameras discovered and local recording is disabled.",
                    Toast.LENGTH_LONG).show();

            return;
        }

        network.armAll(currentSessionId);

        if (recordLocal.isChecked()) {
            cameraEngine.arm(currentSessionId);
        } else {
            cameraText.setText(
                    "Local recording disabled");
        }

        for (NetworkCoordinator.Peer peer : peers) {
            peer.ready = false;
        }

        updatePeers();
    }

    private void startAll() {
        if (currentSessionId == null) {
            Toast.makeText(
                    this,
                    "ARM ALL first.",
                    Toast.LENGTH_SHORT).show();
            return;
        }

        List<NetworkCoordinator.Peer> peers =
                network.getPeers();

        boolean remotesReady = true;

        for (NetworkCoordinator.Peer peer : peers) {
            if (!peer.ready) {
                remotesReady = false;
                break;
            }
        }

        if (!localReady || !remotesReady) {
            Toast.makeText(
                    this,
                    "Not all cameras are READY yet.",
                    Toast.LENGTH_LONG).show();

            updatePeers();
            return;
        }

        final long leadMs = 3000L;

        long targetControllerNs =
                SystemClock.elapsedRealtimeNanos()
                        + leadMs * 1_000_000L;

        network.startAll(
                targetControllerNs,
                leadMs);

        if (recordLocal.isChecked()) {
            cameraEngine.startAt(
                    targetControllerNs);
        }

        sessionText.setText(
                currentSessionId
                        + " · START +3s");
    }

    private void stopAll() {
        if (network != null) {
            network.stopAll();
        }

        if (recordLocal != null
                && recordLocal.isChecked()
                && cameraEngine != null
                && cameraEngine.getState()
                        != CameraEngine.State.IDLE) {

            cameraEngine.stop();
        }

        sessionText.setText(
                (currentSessionId == null
                        ? "No session"
                        : currentSessionId)
                        + " · STOPPED");

        unlockOrientation();
    }

    private boolean ensureCameraPermission() {
        if (checkSelfPermission(
                Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            return true;
        }

        requestPermissions(
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_REQUEST);

        return false;
    }

    private void updatePeers() {
        if (network == null
                || peersContainer == null) {
            return;
        }

        runOnUiThread(() -> {
            List<NetworkCoordinator.Peer> peers =
                    network.getPeers();

            if (peerTitle != null) {
                peerTitle.setText(
                        "DISCOVERED CAMERAS ("
                                + peers.size()
                                + ")");
            }

            peersContainer.removeAllViews();

            if (peers.isEmpty()) {
                TextView empty =
                        new TextView(this);

                empty.setText(
                        "No cameras yet. Put other phones in CAMERA mode on this Wi‑Fi.");

                empty.setTextColor(MUTED);
                empty.setTextSize(
                        landscapeUi ? 9 : 13);

                peersContainer.addView(empty);
                return;
            }

            for (NetworkCoordinator.Peer peer : peers) {
                LinearLayout card =
                        new LinearLayout(this);

                card.setOrientation(
                        LinearLayout.HORIZONTAL);

                card.setGravity(
                        Gravity.CENTER_VERTICAL);

                card.setPadding(
                        dp(landscapeUi ? 7 : 11),
                        dp(landscapeUi ? 5 : 9),
                        dp(landscapeUi ? 7 : 11),
                        dp(landscapeUi ? 5 : 9));

                card.setBackground(
                        gradientPanel(
                                Color.rgb(8, 27, 43),
                                Color.rgb(10, 43, 61),
                                dp(12),
                                peer.ready
                                        ? GREEN
                                        : BORDER));

                TextView indicator =
                        new TextView(this);

                indicator.setText("●");
                indicator.setTextColor(
                        peer.ready
                                ? GREEN
                                : CYAN_SOFT);

                indicator.setTextSize(
                        landscapeUi ? 11 : 15);

                card.addView(indicator);

                LinearLayout details =
                        new LinearLayout(this);

                details.setOrientation(
                        LinearLayout.VERTICAL);

                details.setPadding(
                        dp(7),
                        0,
                        0,
                        0);

                TextView name =
                        new TextView(this);

                name.setText(
                        peer.name == null
                                ? peer.id
                                : peer.name);

                name.setTextColor(TEXT);
                name.setTextSize(
                        landscapeUi ? 9 : 13);

                name.setTypeface(
                        Typeface.DEFAULT_BOLD);

                details.addView(name);

                TextView meta =
                        new TextView(this);

                String address =
                        peer.address == null
                                ? "?"
                                : peer.address
                                        .getHostAddress();

                String sync =
                        peer.hasSync
                                ? String.format(
                                        Locale.US,
                                        "RTT %.1f ms",
                                        peer.rttMs())
                                : "sync pending";

                meta.setText(
                        address
                                + "  ·  "
                                + sync
                                + "  ·  "
                                + peer.status);

                meta.setTextColor(MUTED);
                meta.setTextSize(
                        landscapeUi ? 7 : 11);

                meta.setMaxLines(1);
                details.addView(meta);

                card.addView(
                        details,
                        new LinearLayout.LayoutParams(
                                0,
                                LinearLayout.LayoutParams.WRAP_CONTENT,
                                1f));

                TextView state =
                        new TextView(this);

                state.setText(
                        peer.ready
                                ? "READY"
                                : "CAMERA");

                state.setTextColor(
                        peer.ready
                                ? GREEN
                                : CYAN);

                state.setTextSize(
                        landscapeUi ? 8 : 11);

                state.setTypeface(
                        Typeface.DEFAULT_BOLD);

                card.addView(state);

                LinearLayout.LayoutParams lp =
                        new LinearLayout.LayoutParams(
                                LinearLayout.LayoutParams.MATCH_PARENT,
                                dp(landscapeUi ? 48 : 66));

                if (peersContainer.getChildCount() > 0) {
                    lp.setMargins(
                            0,
                            dp(4),
                            0,
                            0);
                }

                peersContainer.addView(
                        card,
                        lp);
            }
        });
    }

    @Override
    public void onCameraStatus(String status) {
        runOnUiThread(() -> {
            cameraText.setText(status);

            if (previewTelemetry != null) {
                if (status.startsWith("RECORDING")) {
                    previewTelemetry.setText(
                            "● REC · FHD120 · 1/500 · AF");

                    previewTelemetry.setTextColor(GREEN);
                } else if (status.startsWith("READY")) {
                    previewTelemetry.setText(
                            "READY · FHD120 · AF · 1/500");

                    previewTelemetry.setTextColor(TEXT);
                }
            }
        });
    }

    @Override
    public void onCameraReady(String details) {
        localReady = true;

        runOnUiThread(() -> {
            updatePreviewTransform();

            cameraText.setText(
                    "READY · " + details);

            if (previewTelemetry != null) {
                previewTelemetry.setText(
                        "READY · FHD120 · AF · 1/500");
                previewTelemetry.setTextColor(TEXT);
            }
        });

        if (role
                == NetworkCoordinator.Role.CAMERA) {
            network.sendReady(details);
        }
    }

    @Override
    public void onCameraStarted(
            long localStartCallNs) {

        runOnUiThread(() -> {
            cameraText.setText(
                    "RECORDING · FHD120");

            sessionText.setText(
                    currentSessionId
                            + " · RECORDING");

            if (previewTelemetry != null) {
                previewTelemetry.setText(
                        "● REC · FHD120 · 1/500 · AF");
                previewTelemetry.setTextColor(GREEN);
            }
        });

        if (role
                == NetworkCoordinator.Role.CAMERA) {
            network.sendStarted(
                    localStartCallNs);
        }
    }

    @Override
    public void onCameraStopped(
            String videoPath,
            String metadataPath) {

        localReady = false;

        runOnUiThread(() -> {
            double fps =
                    cameraEngine == null
                            ? 0.0
                            : cameraEngine
                                    .getLastEncodedFps();

            if (fps > 0.0) {
                cameraText.setText(
                        String.format(
                                Locale.US,
                                "SAVED · MP4 %.1f fps",
                                fps));

                if (previewTelemetry != null) {
                    previewTelemetry.setText(
                            String.format(
                                    Locale.US,
                                    "SAVED · %.1f FPS · GALLERY",
                                    fps));

                    previewTelemetry.setTextColor(
                            fps >= 110.0
                                    ? GREEN
                                    : RED);
                }
            } else {
                cameraText.setText(
                        "SAVED · fps unavailable");
            }

            sessionText.setText(
                    (currentSessionId == null
                            ? "No session"
                            : currentSessionId)
                            + " · SAVED");
        });

        if (role
                == NetworkCoordinator.Role.CAMERA) {
            network.sendStopped(videoPath);
        }

        unlockOrientation();
    }

    @Override
    public void onCameraError(String error) {
        localReady = false;

        runOnUiThread(() -> {
            cameraText.setText(
                    "ERROR · " + error);

            if (previewTelemetry != null) {
                previewTelemetry.setText(
                        "ERROR · CHECK CAMERA");
                previewTelemetry.setTextColor(RED);
            }
        });

        if (role
                == NetworkCoordinator.Role.CAMERA) {
            network.sendCameraError(error);
        }
    }

    @Override
    public void onPeersChanged() {
        updatePeers();
    }

    @Override
    public void onNetworkStatus(
            String status) {

        runOnUiThread(
                () -> networkText.setText(status));
    }

    @Override
    public void onArmCommand(
            String sessionId) {

        if (role
                != NetworkCoordinator.Role.CAMERA) {
            return;
        }

        lockCurrentOrientation();

        currentSessionId = sessionId;

        runOnUiThread(
                () -> sessionText.setText(
                        sessionId + " · ARMING"));

        cameraEngine.arm(sessionId);
    }

    @Override
    public void onStartCommand(
            long localTargetNs,
            long fallbackDelayMs) {

        if (role
                != NetworkCoordinator.Role.CAMERA) {
            return;
        }

        long target =
                localTargetNs;

        if (target <= 0) {
            target =
                    SystemClock.elapsedRealtimeNanos()
                            + fallbackDelayMs
                                    * 1_000_000L;
        }

        cameraEngine.startAt(target);
    }

    @Override
    public void onStopCommand() {
        if (role
                != NetworkCoordinator.Role.CAMERA) {
            return;
        }

        cameraEngine.stop();
    }

    private void lockCurrentOrientation() {
        if (android.os.Build.VERSION.SDK_INT >= 18) {
            setRequestedOrientation(
                    ActivityInfo.SCREEN_ORIENTATION_LOCKED);
        }
    }

    private void unlockOrientation() {
        setRequestedOrientation(
                ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED);
    }

    @Override
    protected void onSaveInstanceState(
            Bundle outState) {

        if (role != null) {
            outState.putString(
                    "role",
                    role.name());
        }

        super.onSaveInstanceState(outState);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            String[] permissions,
            int[] grantResults) {

        super.onRequestPermissionsResult(
                requestCode,
                permissions,
                grantResults);

        if (requestCode
                == CAMERA_PERMISSION_REQUEST
                && (grantResults.length == 0
                || grantResults[0]
                        != PackageManager.PERMISSION_GRANTED)) {

            Toast.makeText(
                    this,
                    "Camera permission is required to record.",
                    Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onDestroy() {
        if (network != null) {
            network.stop();
        }

        if (cameraEngine != null) {
            cameraEngine.shutdown();
        }

        super.onDestroy();
    }

    private static final class BuildCompat {
        static boolean atLeast23() {
            return android.os.Build.VERSION.SDK_INT >= 23;
        }
    }
}
