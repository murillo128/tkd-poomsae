package dev.murillo.tkd.multicam;

/** Camera-handler confined generation guard for asynchronous camera/session callbacks. */
public final class SessionEpoch {
    private long value;
    public long advance() { return ++value; }
    public long current() { return value; }
    public boolean matches(long token) { return token == value; }
}
