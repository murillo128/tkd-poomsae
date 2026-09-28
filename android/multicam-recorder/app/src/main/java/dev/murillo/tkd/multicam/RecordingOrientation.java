package dev.murillo.tkd.multicam;

/** Output orientation in clockwise degrees, independent of the locked UI/preview. */
public final class RecordingOrientation {
    private RecordingOrientation() {}

    public static final class Choice {
        public final int clockwiseDegrees;
        public final String source;
        public final long measurementAgeNs;
        public final boolean uncertain;
        public Choice(int clockwiseDegrees, String source, long measurementAgeNs, boolean uncertain) {
            if (clockwiseDegrees < 0 || clockwiseDegrees >= 360 || clockwiseDegrees % 90 != 0)
                throw new IllegalArgumentException("Rotation must be 0, 90, 180 or 270");
            this.clockwiseDegrees = clockwiseDegrees;
            this.source = source;
            this.measurementAgeNs = measurementAgeNs;
            this.uncertain = uncertain;
        }
    }

    /** OrientationEventListener angles are clockwise; Display angles use the opposite sign. */
    public static int fromPhysical(int sensorDegrees, int physicalClockwise, boolean front) {
        check(sensorDegrees); check(physicalClockwise);
        return Math.floorMod(sensorDegrees + (front ? -physicalClockwise : physicalClockwise), 360);
    }
    public static int fromDisplay(int sensorDegrees, int displayDegrees, boolean front) {
        check(displayDegrees);
        return fromPhysical(sensorDegrees, Math.floorMod(-displayDegrees, 360), front);
    }
    public static int stableQuarter(int previous, int rawDegrees) {
        if (rawDegrees < 0 || rawDegrees >= 360) return previous;
        if (previous >= 0) {
            int d = Math.abs(previous - rawDegrees);
            if (Math.min(d, 360 - d) < 55) return previous;
        }
        return ((rawDegrees + 45) / 90 * 90) % 360;
    }
    private static void check(int value) {
        if (value < 0 || value >= 360 || value % 90 != 0) throw new IllegalArgumentException("Not a quarter turn");
    }
}
