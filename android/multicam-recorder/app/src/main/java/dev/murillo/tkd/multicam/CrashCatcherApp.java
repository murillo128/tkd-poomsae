package dev.murillo.tkd.multicam;

import android.app.Application;

import java.io.PrintWriter;
import java.io.StringWriter;

public final class CrashCatcherApp extends Application {
    @Override
    public void onCreate() {
        super.onCreate();

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
            } catch (Throwable ignored) {
            }

            if (previous != null) {
                previous.uncaughtException(thread, throwable);
            } else {
                System.exit(10);
            }
        });
    }
}
