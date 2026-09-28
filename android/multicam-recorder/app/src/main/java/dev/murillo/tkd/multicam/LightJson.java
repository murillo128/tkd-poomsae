package dev.murillo.tkd.multicam;

import org.json.JSONObject;

/** Optional v1 telemetry envelope, carried by existing v3 unicast sync replies. */
public final class LightJson {
    private LightJson() {}
    public static final long REMOTE_STALE_NS = 3_000_000_000L;

    public static JSONObject encode(LightMonitor.Snapshot s, long nowNs) {
        if (s == null) return null;
        try {
            long age = s.sampleElapsedNs < 0 ? 60_000 : Math.max(0, (nowNs-s.sampleElapsedNs)/1_000_000L);
            return new JSONObject().put("v",1).put("session",s.sessionId).put("seq",s.sequence)
                    .put("age_ms",Math.min(60_000,age)).put("exposure_ns",s.exposureNs)
                    .put("iso",s.iso).put("band",s.blurBand).put("high_iso",s.highIso)
                    .put("known",s.known).put("reason",s.reason).put("samples",s.samples)
                    .put("sensor_ns",s.sensorTimestampNs);
        } catch (Exception ignored) { return null; } // Diagnostics must not break control packets.
    }

    public static final class Report {
        public final LightMonitor.Snapshot sample;
        public final long receivedNs, ageAtReceiptNs;
        Report(LightMonitor.Snapshot s, long receivedNs, long ageAtReceiptNs) {
            this.sample=s; this.receivedNs=receivedNs; this.ageAtReceiptNs=ageAtReceiptNs;
        }
        public LightMonitor.Snapshot at(long nowNs) {
            if (nowNs < receivedNs || ageAtReceiptNs + nowNs-receivedNs > REMOTE_STALE_NS)
                return LightMonitor.Snapshot.unknown(sample.sessionId,"Remote light telemetry is stale");
            return sample;
        }
    }

    public static Report decode(JSONObject j, String expectedSession, Report previous, long receivedNs) {
        if (j == null) return previous; // Old app versions simply omit this field.
        try {
            if(j.getInt("v")!=1) return previous;
            String session=j.getString("session");
            long seq=j.getLong("seq"), age=j.getLong("age_ms"), e=j.getLong("exposure_ns");
            int iso=j.getInt("iso"), band=j.getInt("band"), samples=j.getInt("samples");
            boolean known=j.getBoolean("known"), high=j.getBoolean("high_iso");
            if(session.isEmpty() || session.length()>128 || seq<0 || age<0 || age>60_000
                    || e < -1 || e>1_000_000_000L || iso < -1 || iso>1_000_000 || band<0 || band>2
                    || samples<0 || samples>160 || (known && (e<=0 || iso<=0))) return previous;
            if(expectedSession!=null && !expectedSession.equals(session)) return previous;
            if(previous!=null && previous.sample.sessionId.equals(session) && seq<=previous.sample.sequence)
                return previous; // Reject delayed/duplicated packets without refreshing freshness.
            String reason=j.optString("reason","");
            if(reason.length()>200) return previous;
            // Sender and receiver elapsed clocks are NOT directly comparable. Only age is transported.
            LightMonitor.Snapshot sample=new LightMonitor.Snapshot(session,seq,e,iso,band,high,known,
                    reason,samples,-1,j.optLong("sensor_ns",-1));
            return new Report(sample,receivedNs,age*1_000_000L);
        } catch(Exception ignored) { return previous; }
    }

    public static JSONObject summary(LightMonitor.Summary s) {
        JSONObject j = new JSONObject();
        try {
            j.put("source","high-speed CaptureResult; not encoded-frame counts")
                    .put("window_ms",500).put("warn_exposure_ns",2_500_000)
                    .put("severe_exposure_ns",4_000_000).put("high_iso_threshold",1600)
                    .put("unique_results",s.uniqueResults).put("duplicates_or_reordered",s.duplicatesOrReordered)
                    .put("missing_timestamp",s.missingTimestamp).put("complete_results",s.completeResults)
                    .put("incomplete_results",s.uniqueResults-s.completeResults)
                    .put("slow_exposure_results",s.slowResults).put("severe_exposure_results",s.severeResults)
                    .put("high_iso_results",s.highIsoResults).put("min_exposure_ns",s.minExposureNs)
                    .put("max_exposure_ns",s.maxExposureNs).put("min_iso",s.minIso).put("max_iso",s.maxIso);
        } catch(Exception ignored) {}
        return j;
    }
}
