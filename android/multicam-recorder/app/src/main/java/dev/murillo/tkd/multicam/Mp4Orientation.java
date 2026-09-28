package dev.murillo.tkd.multicam;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Bounded ISO-BMFF header edit: only the single video track's 36-byte tkhd matrix.
 * No decoding, remuxing, sample-table rewrite, payload copying or timestamp change.
 * Operates ONLY on finalized, app-owned MP4 files before MediaStore publication.
 */
public final class Mp4Orientation {
    private Mp4Orientation() {}
    private static final int MAX_BOXES = 10000;
    private static final int ONE = 1 << 16;
    private static final int MOOV=0x6d6f6f76, TRAK=0x7472616b, MDIA=0x6d646961;
    private static final int HDLR=0x68646c72, TKHD=0x746b6864, VIDE=0x76696465, MOOF=0x6d6f6f66;

    private static final class Box {
        final long payload, end;
        final int type;
        Box(long payload, long end, int type) { this.payload=payload; this.end=end; this.type=type; }
    }
    private static List<Box> children(RandomAccessFile file, long start, long end, int[] budget) throws IOException {
        List<Box> boxes = new ArrayList<>();
        for (long pos = start; pos < end;) {
            if (end-pos < 8 || --budget[0] < 0) throw new IOException("Invalid/truncated MP4 box structure");
            file.seek(pos);
            long size = Integer.toUnsignedLong(file.readInt());
            int type = file.readInt();
            long header = 8;
            if (size == 1) {
                if (end-pos < 16) throw new IOException("Truncated extended box");
                size = file.readLong(); header=16;
            } else if (size == 0) size=end-pos;
            if (size < header || size > end-pos) throw new IOException("MP4 box outside parent bounds");
            if (type == MOOF) throw new IOException("Fragmented MP4 is unsupported");
            boxes.add(new Box(pos+header,pos+size,type));
            pos += size;
        }
        return boxes;
    }
    private static Box unique(List<Box> boxes, int type, boolean required) throws IOException {
        Box found = null;
        for (Box box : boxes) if (box.type==type) {
            if (found!=null) throw new IOException("Ambiguous duplicate MP4 box");
            found=box;
        }
        if (required && found==null) throw new IOException("Required MP4 box is missing");
        return found;
    }
    private static long matrixOffset(RandomAccessFile file) throws IOException {
        int[] budget={MAX_BOXES};
        List<Box> root=children(file,0,file.length(),budget);
        Box moov=unique(root,MOOV,true);
        long offset=-1;
        for (Box track : children(file,moov.payload,moov.end,budget)) if(track.type==TRAK) {
            List<Box> contents=children(file,track.payload,track.end,budget);
            Box mdia=unique(contents,MDIA,true);
            Box handler=unique(children(file,mdia.payload,mdia.end,budget),HDLR,true);
            if(handler.end-handler.payload<12) throw new IOException("Truncated media handler");
            file.seek(handler.payload+8);
            if(file.readInt()!=VIDE) continue;
            if(offset>=0) throw new IOException("More than one video track is unsupported");
            Box tkhd=unique(contents,TKHD,true);
            if(tkhd.end-tkhd.payload<4) throw new IOException("Truncated track header");
            file.seek(tkhd.payload);
            int version=file.readUnsignedByte();
            if(version!=0 && version!=1) throw new IOException("Unsupported tkhd version");
            offset=tkhd.payload+(version==0?40:52);
            if(tkhd.end-offset<44) throw new IOException("Truncated matrix/dimensions");
        }
        if(offset<0) throw new IOException("No video track");
        return offset;
    }
    public static int[] matrix(int clockwise) {
        switch(clockwise) {
            case 0: return new int[]{ONE,0,0,0,ONE,0,0,0,1<<30};
            case 90: return new int[]{0,ONE,0,-ONE,0,0,0,0,1<<30};
            case 180: return new int[]{-ONE,0,0,0,-ONE,0,0,0,1<<30};
            case 270: return new int[]{0,-ONE,0,ONE,0,0,0,0,1<<30};
            default: throw new IllegalArgumentException("Rotation must be 0, 90, 180 or 270");
        }
    }
    private static int rotation(byte[] bytes) throws IOException {
        ByteBuffer b=ByteBuffer.wrap(bytes);
        int[] values=new int[9]; for(int i=0;i<9;i++) values[i]=b.getInt();
        // Native MediaRecorder writes pure quarter-turn matrices. Fail closed on
        // other transforms rather than destroying unknown scaling/perspective.
        for(int angle:new int[]{0,90,180,270}) if(Arrays.equals(values,matrix(angle))) return angle;
        throw new IOException("Unsupported non-rotation video matrix");
    }
    public static int read(File path) throws IOException {
        try(RandomAccessFile file=new RandomAccessFile(path,"r")) {
            long offset=matrixOffset(file); byte[] old=new byte[36];
            file.seek(offset);file.readFully(old);return rotation(old);
        }
    }
    public static void set(File path, int clockwise) throws IOException {
        if (!path.isFile()) throw new IOException("Finalized MP4 file does not exist");
        ByteBuffer replacement=ByteBuffer.allocate(36);
        for(int value:matrix(clockwise)) replacement.putInt(value);
        // Validate all relevant bounds and the existing matrix before ANY write.
        try(RandomAccessFile file=new RandomAccessFile(path,"rw")) {
            long offset=matrixOffset(file);byte[] old=new byte[36];
            file.seek(offset);file.readFully(old);rotation(old);
            if(Arrays.equals(old,replacement.array())) return;
            try {
                file.seek(offset);file.write(replacement.array());file.getFD().sync();
                byte[] check=new byte[36];file.seek(offset);file.readFully(check);
                if(!Arrays.equals(check,replacement.array())) throw new IOException("Rotation write verification failed");
            } catch(IOException failure) {
                try {file.seek(offset);file.write(old);file.getFD().sync();}
                catch(IOException rollback) {failure.addSuppressed(rollback);}
                throw failure;
            }
        }
    }
}
