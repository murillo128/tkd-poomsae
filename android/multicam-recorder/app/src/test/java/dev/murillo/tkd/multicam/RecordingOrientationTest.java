package dev.murillo.tkd.multicam;

import org.junit.Test;
import static org.junit.Assert.*;

public class RecordingOrientationTest {
    @Test public void rearSensor90UsesPhysicalOrientationNotLockedDisplay() {
        int[] physical={0,90,180,270}; int[] expected={90,180,270,0};
        for(int i=0;i<4;i++) assertEquals(expected[i],RecordingOrientation.fromPhysical(90,physical[i],false));
        assertEquals(0,RecordingOrientation.fromPhysical(90,270,false));
        assertEquals(90,RecordingOrientation.fromDisplay(90,0,false));
    }
    @Test public void frontAndOtherMountsUseTheCorrectSign() {
        int[] expected={270,180,90,0};
        for(int i=0;i<4;i++) assertEquals(expected[i],RecordingOrientation.fromPhysical(270,i*90,true));
        for(int sensor:new int[]{0,90,180,270}) for(int display:new int[]{0,90,180,270})
            assertEquals(RecordingOrientation.fromPhysical(sensor,(360-display)%360,false),
                    RecordingOrientation.fromDisplay(sensor,display,false));
    }
    @Test public void unknownAndHysteresisDoNotGuessRandomAxes() {
        assertEquals(-1,RecordingOrientation.stableQuarter(-1,-1));
        assertEquals(270,RecordingOrientation.stableQuarter(270,-1));
        assertEquals(0,RecordingOrientation.stableQuarter(-1,359));
        assertEquals(0,RecordingOrientation.stableQuarter(0,50));
        assertEquals(90,RecordingOrientation.stableQuarter(0,56));
        assertEquals(90,RecordingOrientation.stableQuarter(90,40));
        assertEquals(0,RecordingOrientation.stableQuarter(90,34));
        assertEquals(0,RecordingOrientation.stableQuarter(0,310));
    }
    @Test(expected=IllegalArgumentException.class) public void invalidRotationRejected() {
        new RecordingOrientation.Choice(45,"test",0,false);
    }
}
