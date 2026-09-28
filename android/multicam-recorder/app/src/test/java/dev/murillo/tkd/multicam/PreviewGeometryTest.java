package dev.murillo.tkd.multicam;

import org.junit.Test;
import static org.junit.Assert.*;

public class PreviewGeometryTest {
    @Test public void geometryIsIsotropicCenteredAndNotMirroredIn192Cases() {
        int cases = 0;
        for (int sensor : new int[]{0,90,180,270}) for (int display : new int[]{0,90,180,270})
        for (int[] view : new int[][]{{360,220},{392,310},{860,550},{500,280},{240,640},{17,91}})
        for (boolean fill : new boolean[]{false,true}) {
            int w = view[0], h = view[1];
            float[] m = PreviewGeometry.matrix(w,h,1920,1080,sensor,display,fill);
            double nw = sensor % 180 == 0 ? 1920 : 1080;
            double nh = sensor % 180 == 0 ? 1080 : 1920;
            // Compose with the DEFAULT TextureView stretch: test SOURCE-pixel geometry,
            // not just whether our matrix happens to have equal X/Y diagonal elements.
            double xx=m[0]*w/nw, xy=m[3]*w/nw, yx=m[1]*h/nh, yy=m[4]*h/nh;
            assertEquals("equal pixel scale",Math.hypot(xx,xy),Math.hypot(yx,yy),.003);
            assertEquals("perpendicular axes",0,xx*yx+xy*yy,.003);
            assertTrue("no reflection",xx*yy-yx*xy>0);
            assertEquals(w/2.0,m[0]*w/2+m[1]*h/2+m[2],.003);
            assertEquals(h/2.0,m[3]*w/2+m[4]*h/2+m[5],.003);
            double width=Math.abs(m[0]*w)+Math.abs(m[1]*h);
            double height=Math.abs(m[3]*w)+Math.abs(m[4]*h);
            if (fill) { assertTrue(width>=w-.003); assertTrue(height>=h-.003); }
            else { assertTrue(width<=w+.003); assertTrue(height<=h+.003); }
            assertTrue(Math.abs(width-w)<.003 || Math.abs(height-h)<.003);
            cases++;
        }
        assertEquals(192,cases);
    }
    @Test public void nativePortraitMustNotReceiveSensorRotationTwice() {
        float[] m=PreviewGeometry.matrix(1080,1920,1920,1080,90,0,false);
        assertArrayEquals(new float[]{1,0,0,0,1,0,0,0,1},m,.0001f);
    }
    @Test public void landscapeCompensatesDisplayCounterclockwise() {
        float[] m=PreviewGeometry.matrix(1920,1080,1920,1080,90,90,false);
        assertEquals(0,m[0],.0001); assertTrue(m[1]>0); assertTrue(m[3]<0);
    }
    @Test(expected=IllegalArgumentException.class) public void zeroSizeRejected() {
        PreviewGeometry.matrix(0,10,1920,1080,90,0,false);
    }
}
