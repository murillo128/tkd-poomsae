package dev.murillo.tkd.multicam;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/** Private, app-owned working files. A START marker protects a recording after a crash. */
public final class WarmupStore {
    private static final Set<String> CLEANED = new HashSet<>();
    private final File file, startedMarker;
    private boolean started;

    public WarmupStore(File directory) throws IOException {
        if (!directory.isDirectory() && !directory.mkdirs()) throw new IOException("Cannot create private warm-up directory");
        initialize(directory);
        file = File.createTempFile("warmup-", ".mp4", directory);
        startedMarker = new File(file.getPath() + ".started");
    }
    public static void initialize(File directory) throws IOException {
        synchronized (CLEANED) {
            if (!CLEANED.contains(directory.getCanonicalPath())) {
                cleanupAbandoned(directory); CLEANED.add(directory.getCanonicalPath());
            }
        }
    }
    public File file() { return file; }
    public void markStarted() throws IOException {
        if (started) return;
        try (FileOutputStream out = new FileOutputStream(startedMarker)) {
            out.write("START\n".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
            out.getFD().sync();
        }
        started = true;
    }
    public boolean hasStarted() { return started; }
    public void discard() throws IOException {
        if (file.exists() && !file.delete()) throw new IOException("Cannot delete private warm-up: " + file);
        if (startedMarker.exists() && !startedMarker.delete()) throw new IOException("Cannot delete START marker: " + startedMarker);
    }
    public void discardUnlessStarted() throws IOException { if (!started) discard(); }

    /** Only call at process startup, not while another capture instance is recording. */
    public static void cleanupAbandoned(File directory) throws IOException {
        File[] files = directory.listFiles();
        if (files == null) return;
        for (File f : files) {
            if (!f.getCanonicalFile().getParentFile().equals(directory.getCanonicalFile())) continue;
            if (f.isFile() && f.getName().startsWith("warmup-") && f.getName().endsWith(".mp4")
                    && !new File(f.getPath() + ".started").exists() && !f.delete())
                throw new IOException("Cannot clean abandoned warm-up: " + f);
            if (f.isFile() && f.getName().startsWith("warmup-") && f.getName().endsWith(".mp4.started")
                    && !new File(f.getPath().substring(0, f.getPath().length()-8)).exists()) f.delete();
        }
    }
}
