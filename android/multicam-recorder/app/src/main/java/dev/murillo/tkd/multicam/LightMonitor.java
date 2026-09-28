package dev.murillo.tkd.multicam;

import java.util.Arrays;
import java.util.Locale;

/** Passive, bounded high-speed CaptureResult observer. No Android/camera control calls. */
public final class LightMonitor {
    public static final long WINDOW_NS = 500_000_000L;
    public static final long STALE_NS = 1_500_000_000L;
    private static final int CAPACITY = 160;

    /** Advisory thresholds, not calibrated illuminance or a guarantee of image sharpness. */
    public static final class Config {
        public final long warnExposureNs, severeExposureNs;
        public final int highIso;
        public Config(long warnExposureNs, long severeExposureNs, int highIso) {
            if (warnExposureNs <= 0 || severeExposureNs <= warnExposureNs || highIso <= 0)
                throw new IllegalArgumentException("Invalid light thresholds");
            this.warnExposureNs = warnExposureNs;
            this.severeExposureNs = severeExposureNs;
            this.highIso = highIso;
        }
        public static Config defaults() { return new Config(2_500_000L, 4_000_000L, 1600); }
    }

    public static final class Snapshot {
        public final String sessionId, reason;
        public final long sequence, exposureNs, sampleElapsedNs, sensorTimestampNs;
        public final int iso, blurBand, samples;
        public final boolean highIso, known;
        Snapshot(String sessionId, long sequence, long exposureNs, int iso, int blurBand,
                 boolean highIso, boolean known, String reason, int samples,
                 long sampleElapsedNs, long sensorTimestampNs) {
            this.sessionId = sessionId; this.sequence = sequence; this.exposureNs = exposureNs;
            this.iso = iso; this.blurBand = blurBand; this.highIso = highIso; this.known = known;
            this.reason = reason; this.samples = samples; this.sampleElapsedNs = sampleElapsedNs;
            this.sensorTimestampNs = sensorTimestampNs;
        }
        public static Snapshot unknown(String session, String reason) {
            return new Snapshot(session == null ? "" : session, 0, -1, -1, 0, false,
                    false, reason, 0, -1, -1);
        }
        public boolean warning() { return known && (blurBand > 0 || highIso); }
        public String label() {
            if (!known) return "LIGHT UNKNOWN";
            if (blurBand == 2) return highIso ? "BLUR RISK + HIGH ISO" : "BLUR RISK";
            if (blurBand == 1) return highIso ? "LOW LIGHT + HIGH ISO" : "LOW LIGHT";
            return highIso ? "HIGH ISO" : "GOOD LIGHT";
        }
        public String values() {
            String shutter = exposureNs > 0
                    ? String.format(Locale.US, "1/%.0f s", 1_000_000_000.0 / exposureNs) : "Shutter —";
            return shutter + " · ISO " + (iso > 0 ? iso : "—");
        }
        public String detail() {
            return label() + "\n" + values()
                    + (exposureNs > 0 ? String.format(Locale.US, " (%.2f ms)", exposureNs / 1e6) : "")
                    + "\n" + reason + "\nWindow samples: " + samples
                    + "\nSensor timestamp: " + sensorTimestampNs
                    + "\n\nAdvisory light/blur risk, not a lux measurement or a sharpness guarantee."
                    + "\nMedian over about 0.5 s, with hysteresis. Camera2 high-speed metadata can be batched;"
                    + " this is not a one-to-one mapping to encoded frames."
                    + "\nDefault warnings: exposure >2.5 ms; severe >4 ms; ISO >=1600."
                    + "\nAdd continuous light to reduce blur/noise. Auto exposure and recording stay unchanged.";
        }
    }

    public static final class Summary {
        public final long uniqueResults, duplicatesOrReordered, missingTimestamp, completeResults;
        public final long slowResults, severeResults, highIsoResults, minExposureNs, maxExposureNs;
        public final long minIso, maxIso;
        Summary(long[] v) {
            uniqueResults=v[0]; duplicatesOrReordered=v[1]; missingTimestamp=v[2]; completeResults=v[3];
            slowResults=v[4]; severeResults=v[5]; highIsoResults=v[6]; minExposureNs=v[7];
            maxExposureNs=v[8]; minIso=v[9]; maxIso=v[10];
        }
    }

    private final Config config;
    private final long[] receipt = new long[CAPACITY], exposure = new long[CAPACITY];
    private final int[] sensitivity = new int[CAPACITY];
    private int next, size;
    private String session = "";
    private long lastSensor = -1, lastReceipt = -1, firstReceipt = -1, lastPoll = -1, sequence;
    private long unique, duplicates, noTimestamp, complete, slow, severe, high;
    private long minExposure = Long.MAX_VALUE, maxExposure, minIso = Long.MAX_VALUE, maxIso;
    private int stableBand, candidateBand = -1;
    private boolean stableHigh, candidateHigh;
    private long candidateSince = -1;
    private Snapshot published = Snapshot.unknown("", "No high-speed data yet");

    public LightMonitor() { this(Config.defaults()); }
    public LightMonitor(Config config) { this.config = config; }

    public synchronized void reset(String id) {
        session = id == null ? "" : id;
        next=0; size=0; lastSensor=-1; lastReceipt=-1; firstReceipt=-1; lastPoll=-1; sequence=0;
        unique=0; duplicates=0; noTimestamp=0; complete=0; slow=0; severe=0; high=0;
        minExposure=Long.MAX_VALUE; maxExposure=0; minIso=Long.MAX_VALUE; maxIso=0;
        stableBand=0; stableHigh=false; candidateBand=-1; candidateSince=-1;
        published=Snapshot.unknown(session, "Waiting for high-speed metadata");
    }

    /** O(1), allocation-free. Caller converts nullable result fields to -1. */
    public synchronized void observe(long sensorNs, long elapsedNs, long exposureNs, int iso) {
        if (sensorNs < 0) { noTimestamp++; return; }
        if (sensorNs <= lastSensor) { duplicates++; return; }
        if (elapsedNs < 0 || (lastReceipt >= 0 && elapsedNs < lastReceipt)) return;
        lastSensor=sensorNs; lastReceipt=elapsedNs;
        if (firstReceipt < 0) firstReceipt=elapsedNs;
        // Conservative sanity bounds for diagnostic input; never infer absent metadata.
        long e=exposureNs > 0 && exposureNs <= 1_000_000_000L ? exposureNs : -1;
        int i=iso > 0 && iso <= 1_000_000 ? iso : -1;
        receipt[next]=elapsedNs; exposure[next]=e; sensitivity[next]=i;
        next=(next+1)%CAPACITY; size=Math.min(size+1,CAPACITY); unique++;
        if (e>0 && i>0) {
            complete++; minExposure=Math.min(minExposure,e); maxExposure=Math.max(maxExposure,e);
            minIso=Math.min(minIso,i); maxIso=Math.max(maxIso,i);
            if (e>config.warnExposureNs) slow++;
            if (e>config.severeExposureNs) severe++;
            if (i>=config.highIso) high++;
        }
    }

    /** At most 2 expensive median computations/second; safe to read from the UI thread. */
    public synchronized Snapshot snapshot(long nowNs) {
        if (lastReceipt < 0 || nowNs-lastReceipt > STALE_NS || nowNs < lastReceipt) {
            candidateBand=-1; candidateSince=-1; stableBand=0; stableHigh=false;
            published=new Snapshot(session, ++sequence, -1,-1,0,false,false,
                    lastReceipt < 0 ? "No high-speed metadata" : "High-speed metadata is stale",
                    0,lastReceipt,lastSensor);
            return published;
        }
        if (lastPoll >= 0 && nowNs-lastPoll < WINDOW_NS) return published;
        lastPoll=nowNs;
        long[] ex=new long[size]; int[] isos=new int[size];
        int n=0, ec=0, ic=0, pairs=0;
        // A brief callback pause reuses the latest complete window only up to STALE_NS.
        long windowEnd=lastReceipt;
        for(int p=0;p<size;p++) {
            int index=(next-1-p+CAPACITY)%CAPACITY;
            if(windowEnd-receipt[index]>WINDOW_NS) break;
            n++;
            if(exposure[index]>0) ex[ec++]=exposure[index];
            if(sensitivity[index]>0) isos[ic++]=sensitivity[index];
            if(exposure[index]>0 && sensitivity[index]>0) pairs++;
        }
        Arrays.sort(ex,0,ec); Arrays.sort(isos,0,ic);
        long e=ec==0?-1:ex[ec/2]; int iso=ic==0?-1:isos[ic/2];
        int newest=(next-1+CAPACITY)%CAPACITY;
        String unknownReason = n<4 || lastReceipt-firstReceipt<250_000_000L ? "Measuring high-speed exposure"
                : pairs*4<n*3 || exposure[newest]<=0 || sensitivity[newest]<=0 ? "Shutter or ISO metadata unavailable" : null;
        if(unknownReason!=null) {
            candidateBand=-1; candidateSince=-1; stableBand=0; stableHigh=false;
            published=new Snapshot(session,++sequence,e,iso,0,false,false,unknownReason,n,lastReceipt,lastSensor);
            return published;
        }
        int band=e>config.severeExposureNs?2:e>config.warnExposureNs?1:0;
        if(stableBand==2 && e>config.severeExposureNs*9/10) band=2;
        else if(stableBand>=1 && band==0 && e>config.warnExposureNs*9/10) band=1;
        boolean highIso=iso>=config.highIso || (stableHigh && iso>config.highIso*4/5);
        if(band!=candidateBand || highIso!=candidateHigh) {
            candidateBand=band; candidateHigh=highIso; candidateSince=nowNs;
        }
        boolean stable=nowNs-candidateSince>=WINDOW_NS;
        if(stable) { stableBand=band; stableHigh=highIso; }
        // Do not show an initial green state while a warning is still being established.
        boolean known=published.known || stable;
        published=new Snapshot(session,++sequence,e,iso,stableBand,stableHigh,known,
                known ? "Auto exposure · median high-speed metadata" : "Stabilizing light assessment",
                n,lastReceipt,lastSensor);
        return published;
    }

    public synchronized Summary summary() {
        return new Summary(new long[]{unique,duplicates,noTimestamp,complete,slow,severe,high,
                complete==0?-1:minExposure,complete==0?-1:maxExposure,
                complete==0?-1:minIso,complete==0?-1:maxIso});
    }
}
