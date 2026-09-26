package dev.murillo.tkd.multicam;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

public final class BootstrapActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setGravity(Gravity.CENTER_HORIZONTAL);
        root.setPadding(32, 32, 32, 32);
        scroll.addView(root);

        TextView title = new TextView(this);
        title.setText("TKD MultiCam bootstrap v0.7");
        title.setTextSize(24);
        title.setGravity(Gravity.CENTER);
        root.addView(title, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        TextView info = new TextView(this);
        info.setText("Launcher OK. Camera/network code is loaded only after OPEN MULTICAM.");
        info.setTextSize(16);
        info.setGravity(Gravity.CENTER);
        info.setPadding(0, 24, 0, 24);
        root.addView(info, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        SharedPreferences prefs = getSharedPreferences("crash", MODE_PRIVATE);
        String crash = prefs.getString("last_crash", null);

        if (crash != null && !crash.isEmpty()) {
            TextView crashTitle = new TextView(this);
            crashTitle.setText("LAST MULTICAM CRASH");
            crashTitle.setTextSize(18);
            crashTitle.setPadding(0, 16, 0, 8);
            root.addView(crashTitle);

            TextView crashText = new TextView(this);
            crashText.setText(crash);
            crashText.setTextSize(11);
            crashText.setTextIsSelectable(true);
            crashText.setPadding(8, 8, 8, 8);
            root.addView(crashText, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT));

            Button copy = new Button(this);
            copy.setText("COPY CRASH");
            copy.setOnClickListener(v -> {
                ClipboardManager cm =
                        (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
                cm.setPrimaryClip(
                        ClipData.newPlainText("TKD MultiCam crash", crash));
                Toast.makeText(this, "Crash copied", Toast.LENGTH_SHORT).show();
            });
            root.addView(copy);

            Button clear = new Button(this);
            clear.setText("CLEAR CRASH");
            clear.setOnClickListener(v -> {
                prefs.edit().remove("last_crash").apply();
                recreate();
            });
            root.addView(clear);
        }

        Button open = new Button(this);
        open.setText("OPEN MULTICAM");
        open.setOnClickListener(v -> {
            // Clear an older crash so the next bootstrap shows only the new one.
            prefs.edit().remove("last_crash").apply();

            Intent intent = new Intent();
            intent.setClassName(
                    getPackageName(),
                    "dev.murillo.tkd.multicam.MainActivity");
            try {
                startActivity(intent);
            } catch (Throwable t) {
                java.io.StringWriter sw = new java.io.StringWriter();
                t.printStackTrace(new java.io.PrintWriter(sw));
                prefs.edit().putString("last_crash", sw.toString()).commit();
                recreate();
            }
        });
        root.addView(open, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        setContentView(scroll);
    }
}
