package dev.murillo.tkd.multicam;

import org.junit.Test;
import static org.junit.Assert.*;

public class LightMonitorTest {
    private long time=0, sensor=0;
    private LightMonitor.Snapshot feed(LightMonitor m,long exposure,int iso,int frames) {
        LightMonitor.Snapshot s=null;
        for(int i=0;i<frames;i++) {
            time+=8_333_333L; sensor+=8_333_333L;
            m.observe(sensor,time,exposure,iso);
            s=m.snapshot(time);
        }
        return s;
    }
    @Test public void goodLowSlowAndIsoAreIndependent() {
        LightMonitor m=new LightMonitor();m.reset("one");
        LightMonitor.Snapshot s=feed(m,2_000_000,640,200);
        assertTrue(s.known);assertEquals("GOOD LIGHT",s.label());assertEquals(2_000_000,s.exposureNs);
        s=feed(m,3_333_333,800,240);assertEquals("LOW LIGHT",s.label());assertTrue(s.warning());
        s=feed(m,6_000_000,2500,240);assertEquals(2,s.blurBand);assertTrue(s.highIso);
        s=feed(m,1_000_000,1600,240);assertEquals("HIGH ISO",s.label());
        s=feed(m,2_000_000,400,240);assertEquals("GOOD LIGHT",s.label());
    }
    @Test public void thresholdBoundariesAreIntentional() {
        LightMonitor m=new LightMonitor();m.reset("one");
        assertEquals("GOOD LIGHT",feed(m,2_500_000,1599,220).label());
        m.reset("two");assertEquals("LOW LIGHT",feed(m,4_000_000,1599,220).label());
        m.reset("three");assertEquals("HIGH ISO",feed(m,2_000_000,1600,220).label());
        m.reset("four");assertEquals(2,feed(m,4_000_001,640,220).blurBand);
    }
    @Test public void duplicatesAndOutOfOrderNeverInflateSamples() {
        LightMonitor m=new LightMonitor();m.reset("one");
        for(int i=1;i<=100;i++) {
            long t=i*8_333_333L;m.observe(t,t,2_000_000,640);
            m.observe(t,t,8_000_000,6400);m.observe(t-1,t,8_000_000,6400);
        }
        assertEquals(100,m.summary().uniqueResults);assertEquals(200,m.summary().duplicatesOrReordered);
        assertEquals(2_000_000,m.summary().maxExposureNs);
    }
    @Test public void missingPartialAndStaleCannotLookHealthy() {
        LightMonitor m=new LightMonitor();m.reset("one");
        assertFalse(m.snapshot(0).known);
        assertTrue(feed(m,2_000_000,640,200).known);
        LightMonitor.Snapshot s=feed(m,-1,640,150);assertFalse(s.known);assertEquals(-1,s.exposureNs);
        s=feed(m,2_000_000,-1,150);assertFalse(s.known);assertEquals(-1,s.iso);
        assertTrue(feed(m,2_000_000,640,200).known);
        s=m.snapshot(time+LightMonitor.STALE_NS+1);assertFalse(s.known);assertEquals(-1,s.iso);
        m.observe(-1,time+2_000_000_000L,2_000_000,400);assertEquals(1,m.summary().missingTimestamp);
    }
    @Test public void smoothingAndHysteresisAvoidFlashing() {
        LightMonitor m=new LightMonitor();m.reset("one");
        assertEquals("GOOD LIGHT",feed(m,2_000_000,400,200).label());
        assertEquals("GOOD LIGHT",feed(m,8_000_000,6400,8).label());
        assertEquals("GOOD LIGHT",feed(m,2_000_000,400,180).label());
        assertEquals("LOW LIGHT",feed(m,3_000_000,400,240).label());
        assertEquals("LOW LIGHT",feed(m,2_450_000,400,240).label());
        assertEquals("GOOD LIGHT",feed(m,2_000_000,400,240).label());
        assertEquals("HIGH ISO",feed(m,2_000_000,1800,240).label());
        assertEquals("HIGH ISO",feed(m,2_000_000,1500,240).label());
        assertEquals("GOOD LIGHT",feed(m,2_000_000,1200,240).label());
    }
    @Test public void resetClearsAllPriorSessionData() {
        LightMonitor m=new LightMonitor();m.reset("first");feed(m,8_000_000,3200,200);
        m.reset("second");LightMonitor.Snapshot s=m.snapshot(time);
        assertFalse(s.known);assertEquals("second",s.sessionId);assertEquals(0,m.summary().completeResults);
        assertEquals(-1,m.summary().maxIso);assertEquals("GOOD LIGHT",feed(m,1_000_000,200,200).label());
    }
    @Test public void hugeSessionUsesBoundedWindowAndLifetimeCounters() {
        LightMonitor m=new LightMonitor();m.reset("long");
        LightMonitor.Snapshot s=feed(m,3_000_000,1800,100_000);
        assertTrue(s.samples<=160);assertEquals(100_000,m.summary().uniqueResults);
        assertEquals(100_000,m.summary().highIsoResults);assertEquals(100_000,m.summary().slowResults);
    }
    @Test public void configurableThresholdsAndInvalidReadings() {
        LightMonitor m=new LightMonitor(new LightMonitor.Config(2_000_000,3_000_000,800));m.reset("one");
        assertEquals("LOW LIGHT + HIGH ISO",feed(m,2_500_000,1000,200).label());
        assertFalse(feed(m,Long.MAX_VALUE,Integer.MAX_VALUE,200).known);
    }
}
