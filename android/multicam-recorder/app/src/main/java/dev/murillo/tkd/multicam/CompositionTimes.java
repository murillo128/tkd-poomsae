package dev.murillo.tkd.multicam;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;

/** Restore presentation times after a monotonic native-mux staging pass.
 * Android MPEG4Writer rejects samples whose PTS precedes its first sample. We let
 * it mux payloads in decoder order, then write an explicit signed CTTS table.
 * Only a private, unpublished partial MP4 is touched. Chunk offsets/payloads stay put.
 */
public final class CompositionTimes {
    private CompositionTimes() {}
    private static final int MAX_MOOV=32*1024*1024, MAX_SAMPLES=500_000;
    private static final class Box {
        final int start, body, end; final String type;
        Box(int start,int body,int end,String type){this.start=start;this.body=body;this.end=end;this.type=type;}
    }
    private static int i32(byte[] b,int p){return ByteBuffer.wrap(b,p,4).getInt();}
    private static long u32(byte[] b,int p){return Integer.toUnsignedLong(i32(b,p));}
    private static void put32(byte[] b,int p,long value)throws IOException {
        if(value<0||value>0xffffffffL)throw new IOException("MP4 duration overflow");
        ByteBuffer.wrap(b,p,4).putInt((int)value);
    }
    private static List<Box> boxes(byte[] b,int start,int end)throws IOException {
        List<Box> out=new ArrayList<>();
        for(int pos=start;pos<end;){
            if(end-pos<8||out.size()>10_000)throw new IOException("Invalid MP4 children");
            long size=u32(b,pos);int header=8;
            if(size==1){if(end-pos<16)throw new IOException("Invalid extended atom");size=ByteBuffer.wrap(b,pos+8,8).getLong();header=16;}
            else if(size==0)size=end-pos;
            if(size<header||size>end-pos)throw new IOException("Atom exceeds parent");
            out.add(new Box(pos,pos+header,pos+(int)size,new String(b,pos+4,4,java.nio.charset.StandardCharsets.US_ASCII)));
            pos+=(int)size;
        }
        return out;
    }
    private static Box only(List<Box> boxes,String type)throws IOException {
        Box found=null;for(Box b:boxes)if(type.equals(b.type)){if(found!=null)throw new IOException("Duplicate "+type);found=b;}
        if(found==null)throw new IOException("Missing "+type);return found;
    }
    private static byte[] raw(byte[] b,Box box){return Arrays.copyOfRange(b,box.start,box.end);}
    private static byte[] atom(String name,byte[] content)throws IOException {
        ByteArrayOutputStream out=new ByteArrayOutputStream(content.length+8);DataOutputStream d=new DataOutputStream(out);
        d.writeInt(content.length+8);d.writeBytes(name);d.write(content);return out.toByteArray();
    }
    private static long ticks(long us,long scale)throws IOException {
        if(us<0||us>Long.MAX_VALUE/scale)throw new IOException("Invalid presentation time");
        return (us*scale+500_000L)/1_000_000L;
    }
    private static long timeScale(byte[] b,Box box)throws IOException {
        if(box.end-box.body<24)throw new IOException("Truncated time header");
        int version=b[box.body]&255;
        if(version!=0&&version!=1)throw new IOException("Unsupported time header");
        int p=box.body+(version==0?12:20);
        if(p+4>box.end)throw new IOException("Truncated timescale");
        long scale=u32(b,p);if(scale<=0)throw new IOException("Zero timescale");return scale;
    }
    private static byte[] duration(byte[] b,Box box,long value,boolean track)throws IOException {
        byte[] out=raw(b,box);int body=box.body-box.start;int version=out[body]&255;
        if(version!=0&&version!=1)throw new IOException("Unsupported duration version");
        int p=body+(track?(version==0?20:28):(version==0?16:24));
        int bytes=version==0?4:8;
        if(p+bytes>out.length)throw new IOException("Truncated duration");
        if(version==0)put32(out,p,value);else ByteBuffer.wrap(out,p,8).putLong(value);
        return out;
    }
    public static void restore(File path,long[] presentationUs)throws IOException {
        if(presentationUs.length<2||presentationUs.length>MAX_SAMPLES)throw new IOException("Unsupported sample count");
        long max=0;for(long p:presentationUs){if(p<0)throw new IOException("Negative normalized PTS");max=Math.max(max,p);}
        long[] sorted=presentationUs.clone();Arrays.sort(sorted);
        long[] delta=new long[sorted.length-1];int n=0;
        for(int j=1;j<sorted.length;j++)if(sorted[j]>sorted[j-1])delta[n++]=sorted[j]-sorted[j-1];
        if(n==0)throw new IOException("No presentation duration");Arrays.sort(delta,0,n);
        long endUs=Math.addExact(max,delta[n/2]);
        try(RandomAccessFile f=new RandomAccessFile(path,"rw")){
            long oldOffset=-1,oldSize=0,length=f.length();int roots=0;
            for(long pos=0;pos<length;){
                if(length-pos<8||++roots>10_000)throw new IOException("Invalid MP4 roots");
                f.seek(pos);long size=Integer.toUnsignedLong(f.readInt());int type=f.readInt(),header=8;
                if(size==1){if(length-pos<16)throw new IOException("Invalid extended root");size=f.readLong();header=16;}
                else if(size==0)size=length-pos;
                if(size<header||size>length-pos)throw new IOException("Invalid root size");
                if(type==0x6d6f6f66)throw new IOException("Fragmented MP4 unsupported");
                if(type==0x6d6f6f76){if(oldOffset>=0||size>MAX_MOOV)throw new IOException("Invalid moov");oldOffset=pos;oldSize=size;}
                pos+=size;
            }
            if(oldOffset<0)throw new IOException("No moov");
            byte[] b=new byte[(int)oldSize];f.seek(oldOffset);f.readFully(b);
            Box moov=only(boxes(b,0,b.length),"moov");List<Box> mc=boxes(b,moov.body,moov.end);
            Box mvhd=only(mc,"mvhd"),trak=only(mc,"trak");
            List<Box> tc=boxes(b,trak.body,trak.end);Box tkhd=only(tc,"tkhd"),mdia=only(tc,"mdia");
            List<Box> dc=boxes(b,mdia.body,mdia.end);Box mdhd=only(dc,"mdhd"),hdlr=only(dc,"hdlr"),minf=only(dc,"minf");
            if(hdlr.end-hdlr.body<12||i32(b,hdlr.body+8)!=0x76696465)throw new IOException("Not single video");
            List<Box> ic=boxes(b,minf.body,minf.end);Box stbl=only(ic,"stbl");
            List<Box> sc=boxes(b,stbl.body,stbl.end);Box stts=only(sc,"stts"),stsz=only(sc,"stsz");
            if(stsz.end-stsz.body<12||u32(b,stsz.body+8)!=presentationUs.length)throw new IOException("Sample count mismatch");
            long mediaScale=timeScale(b,mdhd),movieScale=timeScale(b,mvhd);
            if(stts.end-stts.body<8)throw new IOException("Truncated STTS");
            long entries=u32(b,stts.body+4);
            if(entries>(stts.end-stts.body-8)/8)throw new IOException("Truncated STTS entries");
            int[] offsets=new int[presentationUs.length];int index=0;long dts=0;
            for(int j=0;j<entries;j++){
                int p=stts.body+8+j*8;long count=u32(b,p),step=u32(b,p+4);
                if(count<=0||count>offsets.length-index)throw new IOException("Invalid STTS sample run");
                for(int k=0;k<count;k++){
                    long offset=ticks(presentationUs[index],mediaScale)-dts;
                    if(offset<Integer.MIN_VALUE||offset>Integer.MAX_VALUE)throw new IOException("CTTS offset overflow");
                    offsets[index++]=(int)offset;dts=Math.addExact(dts,step);
                }
            }
            if(index!=offsets.length)throw new IOException("STTS count mismatch");
            ByteArrayOutputStream runs=new ByteArrayOutputStream();DataOutputStream rd=new DataOutputStream(runs);int runCount=0;
            for(int j=0;j<offsets.length;){int k=j+1;while(k<offsets.length&&offsets[k]==offsets[j])k++;
                rd.writeInt(k-j);rd.writeInt(offsets[j]);runCount++;j=k;}
            ByteArrayOutputStream cp=new ByteArrayOutputStream();DataOutputStream cd=new DataOutputStream(cp);
            cd.writeInt(0x01000000);cd.writeInt(runCount);cd.write(runs.toByteArray());
            byte[] ctts=atom("ctts",cp.toByteArray());
            ByteArrayOutputStream sb=new ByteArrayOutputStream();
            for(Box box:sc)if(!box.type.equals("ctts"))sb.write(raw(b,box));sb.write(ctts);
            byte[] newStbl=atom("stbl",sb.toByteArray());
            ByteArrayOutputStream ib=new ByteArrayOutputStream();for(Box box:ic)ib.write(box==stbl?newStbl:raw(b,box));
            byte[] newMinf=atom("minf",ib.toByteArray());
            ByteArrayOutputStream db=new ByteArrayOutputStream();
            for(Box box:dc)db.write(box==minf?newMinf:box==mdhd?duration(b,box,Math.max(dts,ticks(endUs,mediaScale)),false):raw(b,box));
            byte[] newMdia=atom("mdia",db.toByteArray());
            ByteArrayOutputStream tb=new ByteArrayOutputStream();
            for(Box box:tc)if(!box.type.equals("edts"))tb.write(box==mdia?newMdia:box==tkhd?duration(b,box,ticks(endUs,movieScale),true):raw(b,box));
            byte[] newTrak=atom("trak",tb.toByteArray());
            ByteArrayOutputStream mb=new ByteArrayOutputStream();
            for(Box box:mc)mb.write(box==trak?newTrak:box==mvhd?duration(b,box,ticks(endUs,movieScale),false):raw(b,box));
            byte[] replacement=atom("moov",mb.toByteArray());
            if(replacement.length>MAX_MOOV)throw new IOException("New moov exceeds bound");
            // Chunk offsets and payload bytes stay in place. A private partial is
            // published only after the caller reopens and verifies all samples/PTS.
            f.seek(length);f.write(replacement);f.getFD().sync();
            f.seek(oldOffset+4);f.writeInt(0x66726565);f.getFD().sync();
        }catch(ArithmeticException e){throw new IOException("MP4 timestamp overflow",e);}
    }
}
