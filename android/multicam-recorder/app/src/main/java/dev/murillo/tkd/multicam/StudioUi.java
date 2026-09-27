package dev.murillo.tkd.multicam;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

/** Small native-view styling helpers, shared by portrait and landscape dashboards. */
final class StudioUi {
    static final int BG = 0xff06131e, PANEL = 0xff0b202f, LINE = 0xff224659;
    static final int TEXT = 0xffedf6fb, MUTED = 0xff97afbf, CYAN = 0xff19d5e8;
    static final int GREEN = 0xff20e5ac, RED = 0xffff5568, AMBER = 0xffffc66b;
    final Context context;
    StudioUi(Context context) { this.context = context; }
    int dp(float v) { return Math.round(v * context.getResources().getDisplayMetrics().density); }

    TextView text(String value, float size, int color, boolean bold) {
        TextView t = new TextView(context);
        t.setText(value);
        t.setTextSize(size);
        t.setTextColor(color);
        t.setIncludeFontPadding(false);
        t.setTypeface(Typeface.create("sans-serif", bold ? Typeface.BOLD : Typeface.NORMAL));
        return t;
    }
    TextView line(String value, float size, int color, boolean bold) {
        TextView t = text(value, size, color, bold);
        t.setSingleLine(true);
        t.setEllipsize(TextUtils.TruncateAt.END);
        return t;
    }
    LinearLayout column() {
        LinearLayout v = new LinearLayout(context);
        v.setOrientation(LinearLayout.VERTICAL);
        return v;
    }
    LinearLayout row() {
        LinearLayout v = new LinearLayout(context);
        v.setOrientation(LinearLayout.HORIZONTAL);
        v.setGravity(Gravity.CENTER_VERTICAL);
        return v;
    }
    GradientDrawable panel(int top, int bottom, int border, float radius) {
        GradientDrawable d = new GradientDrawable(GradientDrawable.Orientation.TL_BR,
                new int[]{top, bottom});
        d.setCornerRadius(dp(radius));
        d.setStroke(Math.max(1, dp(0.7f)), border);
        return d;
    }
    Drawable touch(int top, int bottom, int border, float radius) {
        return new RippleDrawable(ColorStateList.valueOf(0x3341ecff),
                panel(top, bottom, border, radius), panel(Color.WHITE, Color.WHITE, Color.WHITE, radius));
    }
    ImageView icon(String kind, int color, int size) {
        ImageView i = new ImageView(context);
        i.setImageDrawable(new Glyph(kind, color));
        i.setLayoutParams(new LinearLayout.LayoutParams(dp(size), dp(size)));
        i.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        return i;
    }
    LinearLayout command(String tag, String glyph, String label, String subtitle,
                         int accent, boolean compact, Runnable action) {
        LinearLayout button = row();
        button.setTag(tag);
        int first = accent == RED ? 0xff77283b : accent == GREEN ? 0xff008d7c : 0xff075165;
        int second = accent == RED ? 0xff371a2a : accent == GREEN ? 0xff07464b : 0xff0a293b;
        button.setBackground(touch(first, second, accent, 11));
        button.setPadding(dp(compact ? 10 : 12), dp(8), dp(9), dp(8));
        button.addView(icon(glyph, accent, compact ? 24 : 28));
        LinearLayout labels = column();
        labels.setGravity(Gravity.CENTER_VERTICAL);
        labels.setPadding(dp(compact ? 8 : 10), 0, 0, 0);
        TextView primary = line(label, compact ? 12 : 14, TEXT, true);
        TextView secondary = line(subtitle, compact ? 9 : 10, 0xffabd2d8, false);
        labels.addView(primary);
        LinearLayout.LayoutParams sub = new LinearLayout.LayoutParams(-1, -2);
        sub.topMargin = dp(4);
        labels.addView(secondary, sub);
        button.addView(labels, new LinearLayout.LayoutParams(0, -1, 1));
        button.setFocusable(true);
        button.setClickable(true);
        button.setContentDescription(label + ". " + subtitle);
        button.setOnClickListener(v -> action.run());
        return button;
    }

    /** Consistent vector strokes rather than platform-dependent Unicode button glyphs. */
    static final class Glyph extends Drawable {
        private final String kind;
        private final Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        Glyph(String kind, int color) { this.kind = kind; p.setColor(color); }
        private void line(Canvas c, float x, float y, float xx, float yy) { c.drawLine(x,y,xx,yy,p); }
        private void path(Canvas c, float... points) {
            Path a = new Path(); a.moveTo(points[0], points[1]);
            for (int i = 2; i < points.length; i += 2) a.lineTo(points[i], points[i+1]);
            c.drawPath(a,p);
        }
        @Override public void draw(Canvas canvas) {
            canvas.save();
            canvas.translate(getBounds().left, getBounds().top);
            canvas.scale(getBounds().width()/24f, getBounds().height()/24f);
            p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(1.8f);
            p.setStrokeCap(Paint.Cap.ROUND); p.setStrokeJoin(Paint.Join.ROUND);
            switch (kind) {
                case "role":
                    canvas.drawCircle(12,7,3.2f,p);
                    canvas.drawArc(new RectF(5,12,19,24),180,180,false,p); line(canvas,5,18,19,18); break;
                case "wifi":
                    for (int r : new int[]{5,9,13}) canvas.drawArc(new RectF(12-r,20-r,12+r,20+r),225,90,false,p);
                    p.setStyle(Paint.Style.FILL); canvas.drawCircle(12,19,1.5f,p); break;
                case "camera":
                    canvas.drawRoundRect(3,6,21,20,2,2,p); path(canvas,8,6,9,3,15,3,16,6);
                    canvas.drawCircle(12,13,4,p); break;
                case "session": case "clock":
                    canvas.drawCircle(12,12,9,p); path(canvas,12,6,12,12,16,15); break;
                case "phone":
                    canvas.drawRoundRect(6,2,18,22,2,2,p); line(canvas,10,19,14,19); break;
                case "search":
                    canvas.drawCircle(10,10,6.5f,p); line(canvas,15,15,21,21); break;
                case "arm":
                    path(canvas,12,2,20,6,19,14,16,19,12,22,8,19,5,14,4,6,12,2); break;
                case "play":
                    p.setStyle(Paint.Style.FILL); Path triangle=new Path(); triangle.moveTo(6,3);
                    triangle.lineTo(21,12); triangle.lineTo(6,21); triangle.close(); canvas.drawPath(triangle,p); break;
                case "stop":
                    p.setStyle(Paint.Style.FILL); canvas.drawRoundRect(4,4,20,20,2,2,p); break;
                case "menu":
                    p.setStyle(Paint.Style.FILL); for(int y:new int[]{5,12,19}) canvas.drawCircle(12,y,1.6f,p); break;
                case "refresh":
                    canvas.drawArc(new RectF(4,4,20,20),40,300,false,p); path(canvas,15,3,20,5,21,0); break;
                case "folder":
                    path(canvas,3,19,3,5,9,5,12,8,21,8,21,19,3,19); break;
                case "battery":
                    canvas.drawRoundRect(2,7,20,17,1.5f,1.5f,p); line(canvas,22,10,22,14); break;
                case "fit":
                    path(canvas,3,8,3,3,8,3); path(canvas,16,3,21,3,21,8);
                    path(canvas,3,16,3,21,8,21); path(canvas,16,21,21,21,21,16); break;
                case "error":
                    canvas.drawCircle(12,12,9,p); line(canvas,12,6,12,13);
                    p.setStyle(Paint.Style.FILL); canvas.drawCircle(12,17,1,p); break;
                default: canvas.drawCircle(12,12,7,p);
            }
            canvas.restore();
        }
        @Override public void setAlpha(int alpha) { p.setAlpha(alpha); invalidateSelf(); }
        @Override public void setColorFilter(ColorFilter filter) { p.setColorFilter(filter); invalidateSelf(); }
        @Override public int getOpacity() { return PixelFormat.TRANSLUCENT; }
    }
}
