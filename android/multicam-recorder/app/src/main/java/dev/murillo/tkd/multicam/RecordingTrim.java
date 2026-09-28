package dev.murillo.tkd.multicam;

import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.os.Build;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.Arrays;

/** Lossless single-video remux: keep preceding sync + dependencies, preserve decode order. */
public final class RecordingTrim {
    private static final int MAX_SAMPLE = 16 * 1024 * 1024;
    private static final long MAX_PTS_ROUNDING_US = 25;
    private RecordingTrim() {}
    public static final class Result {
        public final long requestedFromFirstUs, discardedUs, retainedLeadUs, firstSourcePtsUs, frames, maxPtsRoundingUs;
        public final String sampleDigest;
        Result(long requested,long discarded,long lead,long first,long frames,long rounding,String digest) {
            requestedFromFirstUs=requested;discardedUs=discarded;retainedLeadUs=lead;
            firstSourcePtsUs=first;this.frames=frames;maxPtsRoundingUs=rounding;sampleDigest=digest;
        }
    }
    private static MediaFormat select(MediaExtractor ex) throws IOException {
        if(ex.getTrackCount()!=1) throw new IOException("Expected a single video-only recording");
        MediaFormat f=ex.getTrackFormat(0);
        if(!"video/avc".equals(f.getString(MediaFormat.KEY_MIME))) throw new IOException("Expected H.264 recording");
        ex.selectTrack(0);
        if(ex.getSampleTrackIndex()<0) throw new IOException("No video samples");
        return f;
    }
    private static ByteBuffer capacity(MediaExtractor ex,ByteBuffer buffer) throws IOException {
        long size=Build.VERSION.SDK_INT>=28?ex.getSampleSize():8L*1024*1024;
        if(size<=0||size>MAX_SAMPLE) throw new IOException("Unsupported sample size: "+size);
        return size>buffer.capacity()?ByteBuffer.allocateDirect((int)size):buffer;
    }
    private static int flags(MediaExtractor ex) throws IOException {
        int flags=ex.getSampleFlags();
        if((flags&~MediaExtractor.SAMPLE_FLAG_SYNC)!=0) throw new IOException("Encrypted/partial samples unsupported");
        return (flags&MediaExtractor.SAMPLE_FLAG_SYNC)!=0?MediaCodec.BUFFER_FLAG_KEY_FRAME:0;
    }
    private static void digest(MessageDigest d,ByteBuffer b,int flags) {
        d.update(ByteBuffer.allocate(8).putInt(b.remaining()).putInt(flags).array());d.update(b.asReadOnlyBuffer());
    }
    private static String hex(byte[] bytes) {
        StringBuilder out=new StringBuilder();for(byte b:bytes)out.append(String.format(java.util.Locale.US,"%02x",b&255));return out.toString();
    }
    private static void seek(MediaExtractor ex,long cut) throws IOException {
        ex.seekTo(cut,MediaExtractor.SEEK_TO_PREVIOUS_SYNC);
        if(ex.getSampleTrackIndex()<0 || ex.getSampleTime()!=cut || flags(ex)!=MediaCodec.BUFFER_FLAG_KEY_FRAME)
            throw new IOException("Cannot return to selected sync sample");
    }
    public static Result remux(File input,File output,long startFromFirstUs) throws IOException {
        if(startFromFirstUs<0||output.exists()||input.getCanonicalFile().equals(output.getCanonicalFile()))
            throw new IOException("Invalid trim paths or time");
        File partial=new File(output.getPath()+".partial");
        if(partial.exists())throw new IOException("Partial output already exists");
        MediaExtractor ex=new MediaExtractor();MediaMuxer mux=null;boolean complete=false;
        try {
            ex.setDataSource(input.getPath());MediaFormat format=select(ex);
            long first=ex.getSampleTime(),target=Math.addExact(first,startFromFirstUs);
            ex.seekTo(target,MediaExtractor.SEEK_TO_PREVIOUS_SYNC);
            long cut=ex.getSampleTime();
            if(ex.getSampleTrackIndex()<0||cut<first||cut>target||flags(ex)!=MediaCodec.BUFFER_FLAG_KEY_FRAME)
                throw new IOException("Cannot find a safe sync frame before START");
            long min=cut,max=cut,count=0;
            while(ex.getSampleTrackIndex()>=0) {
                min=Math.min(min,ex.getSampleTime());max=Math.max(max,ex.getSampleTime());count++;
                if(!ex.advance())break;
            }
            if(count>500_000)throw new IOException("Recording exceeds bounded sample limit");
            if(max<target||count<2)throw new IOException("Recording ended before any START frames were captured");
            seek(ex,cut);
            mux=new MediaMuxer(partial.getPath(),MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
            mux.setOrientationHint(Mp4Orientation.read(input));
            int track=mux.addTrack(format);mux.start();
            MediaCodec.BufferInfo info=new MediaCodec.BufferInfo();ByteBuffer b=ByteBuffer.allocateDirect(256*1024);
            MessageDigest written=MessageDigest.getInstance("SHA-256");
            long[] presentation=new long[(int)count];int sampleIndex=0;
            while(ex.getSampleTrackIndex()>=0) {
                b=capacity(ex,b);b.clear();int n=ex.readSampleData(b,0),flag=flags(ex);
                if(n<=0||n>b.capacity())throw new IOException("Invalid sample while trimming");
                b.position(0);b.limit(n);presentation[sampleIndex]=ex.getSampleTime()-min;
                // Stage only in monotonic decode order. Native MPEG4Writer subtracts
                // its first PTS and otherwise rejects later pictures with earlier PTS.
                // The unpublished staging timeline is replaced before verification.
                info.set(0,n,Math.round(sampleIndex*1_000_000.0/120),flag);sampleIndex++;
                digest(written,b,flag);mux.writeSampleData(track,b,info);
                if(!ex.advance())break;
            }
            mux.stop();mux.release();mux=null;
            CompositionTimes.restore(partial,presentation);
            byte[] expected=written.digest();long[] verified=verify(input,partial,cut,count,expected);
            if(!partial.renameTo(output))throw new IOException("Cannot commit trimmed MP4");
            complete=true;
            long lead=target-cut+verified[0];
            return new Result(startFromFirstUs,Math.max(0,startFromFirstUs-lead),Math.max(0,lead),cut,count,verified[1],hex(expected));
        } catch(IOException e){throw e;}
        catch(Exception e){throw new IOException("Lossless trim failed: "+e.getMessage(),e);}
        finally {
            ex.release();if(mux!=null){try{mux.release();}catch(Exception ignored){}}
            if(!complete&&partial.exists())partial.delete();
        }
    }
    private static long[] verify(File input,File output,long cut,long count,byte[] expected) throws Exception {
        MediaExtractor source=new MediaExtractor(),out=new MediaExtractor();
        try {
            source.setDataSource(input.getPath());select(source);seek(source,cut);
            out.setDataSource(output.getPath());select(out);
            if(flags(out)!=MediaCodec.BUFFER_FLAG_KEY_FRAME)throw new IOException("Final video must start with a sync sample");
            long first=out.getSampleTime(),frames=0,maxError=0;
            ByteBuffer b=ByteBuffer.allocateDirect(256*1024);MessageDigest actual=MessageDigest.getInstance("SHA-256");
            while(out.getSampleTrackIndex()>=0) {
                if(source.getSampleTrackIndex()<0)throw new IOException("Unexpected extra output sample");
                long error=Math.abs((out.getSampleTime()-first)-(source.getSampleTime()-cut));
                maxError=Math.max(maxError,error);
                if(error>MAX_PTS_ROUNDING_US)throw new IOException("Remux changed relative timestamps by "+error+" us");
                b=capacity(out,b);b.clear();int n=out.readSampleData(b,0);
                if(n<=0||n>b.capacity())throw new IOException("Invalid verification sample");
                b.position(0);b.limit(n);digest(actual,b,flags(out));frames++;
                source.advance();if(!out.advance())break;
            }
            if(frames!=count||source.getSampleTrackIndex()>=0||!Arrays.equals(expected,actual.digest()))
                throw new IOException("Remux changed compressed samples or sample order");
            return new long[]{first,maxError};
        } finally {source.release();out.release();}
    }
}
