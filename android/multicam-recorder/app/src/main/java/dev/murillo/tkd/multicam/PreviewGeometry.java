package dev.murillo.tkd.multicam;

/** Pure, testable TextureView geometry. All matrix inputs are VIEW coordinates. */
public final class PreviewGeometry {
    private PreviewGeometry() {}

    public static float[] matrix(int vw, int vh, int bw, int bh,
                                 int sensorDegrees, int displayDegrees, boolean fill) {
        if (vw <= 0 || vh <= 0 || bw <= 0 || bh <= 0) {
            throw new IllegalArgumentException("Positive dimensions required");
        }
        if (sensorDegrees % 90 != 0 || displayDegrees % 90 != 0) {
            throw new IllegalArgumentException("Only right-angle rotations supported");
        }
        // Camera2/SurfaceTexture already compensates the sensor's mounting rotation.
        // Its DEFAULT transform stretches that naturally-oriented image into vw x vh.
        // Undo that stretch, rotate only against the DISPLAY, then fit/fill uniformly.
        boolean sensorSwap = Math.floorMod(sensorDegrees, 180) == 90;
        double nw = sensorSwap ? bh : bw;
        double nh = sensorSwap ? bw : bh;
        int quarter = Math.floorMod(displayDegrees / 90, 4);
        double[] cos = {1, 0, -1, 0};
        double[] sin = {0, -1, 0, 1};
        double c = cos[quarter], s = sin[quarter];
        double rw = quarter % 2 == 0 ? nw : nh;
        double rh = quarter % 2 == 0 ? nh : nw;
        double k = fill ? Math.max(vw / rw, vh / rh) : Math.min(vw / rw, vh / rh);
        double a = k * c * nw / vw;
        double b = -k * s * nh / vh;
        double d = k * s * nw / vw;
        double e = k * c * nh / vh;
        return new float[] {
            (float) a, (float) b, (float) (vw / 2.0 - a * vw / 2.0 - b * vh / 2.0),
            (float) d, (float) e, (float) (vh / 2.0 - d * vw / 2.0 - e * vh / 2.0),
            0, 0, 1
        };
    }
}
