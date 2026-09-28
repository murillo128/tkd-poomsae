package dev.murillo.tkd.multicam;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.util.Arrays;
import org.junit.Test;
import static org.junit.Assert.*;

public class Mp4OrientationTest {
    private byte[] join(byte[]... arrays) throws IOException {
        ByteArrayOutputStream out=new ByteArrayOutputStream();
        for(byte[] a:arrays) out.write(a);return out.toByteArray();
    }
    private byte[] box(String name,boolean extended,byte[]... payload) throws IOException {
        byte[] content=join(payload);
        ByteArrayOutputStream out=new ByteArrayOutputStream();DataOutputStream d=new DataOutputStream(out);
        d.writeInt(extended?1:content.length+8);d.writeBytes(name);
        if(extended)d.writeLong(content.length+16L);d.write(content);return out.toByteArray();
    }
    private byte[] track(int version,boolean extended,int rotation,String kind) throws IOException {
        ByteBuffer tk=ByteBuffer.allocate(version==0?84:96);tk.put((byte)version);
        tk.position(version==0?40:52);
        for(int i:Mp4Orientation.matrix(rotation)) tk.putInt(i);
        tk.putInt(1920<<16);tk.putInt(1080<<16);
        ByteBuffer h=ByteBuffer.allocate(12);h.position(8);h.put(kind.getBytes("US-ASCII"));
        return box("trak",extended,box("tkhd",extended,tk.array()),box("mdia",extended,box("hdlr",extended,h.array())));
    }
    private File file(byte[] data) throws IOException {
        File f=File.createTempFile("rotation-test", ".mp4");f.deleteOnExit();Files.write(f.toPath(),data);return f;
    }
    @Test public void v0v1AndExtendedHeadersPreserveEverythingExceptMatrix() throws Exception {
        for(int version:new int[]{0,1}) for(boolean extended:new boolean[]{false,true}) {
            byte[] payload=new byte[4096];for(int i=0;i<payload.length;i++)payload[i]=(byte)(i*31);
            byte[] prefix=box("ftyp",false,new byte[16]);
            byte[] video=track(version,extended,90,"vide");
            byte[] fixture=join(prefix,box("mdat",false,payload),box("moov",extended,
                    track(version,extended,0,"soun"),video));
            File f=file(fixture);
            // Find matrix by its distinctive 90-degree bytes in this deterministic fixture.
            byte[] old=ByteBuffer.allocate(36).putInt(0).putInt(65536).putInt(0).putInt(-65536)
                    .putInt(0).putInt(0).putInt(0).putInt(0).putInt(1<<30).array();
            int offset=-1;
            for(int i=0;i<=fixture.length-36;i++)if(Arrays.equals(old,Arrays.copyOfRange(fixture,i,i+36)))offset=i;
            assertTrue(offset>0);assertEquals(90,Mp4Orientation.read(f));
            for(int angle:new int[]{0,180,270,90}) {
                Mp4Orientation.set(f,angle);assertEquals(angle,Mp4Orientation.read(f));
                byte[] after=Files.readAllBytes(f.toPath());assertEquals(fixture.length,after.length);
                for(int i=0;i<after.length;i++) if(i<offset||i>=offset+36)assertEquals("byte "+i,fixture[i],after[i]);
            }
        }
    }
    private void rejectsUnchanged(byte[] bytes) throws Exception {
        File f=file(bytes);
        try {Mp4Orientation.set(f,0);fail("Expected rejection");}catch(IOException expected){}
        assertArrayEquals(bytes,Files.readAllBytes(f.toPath()));
    }
    @Test public void malformedUnsupportedAndMultipleVideoFilesAreNeverModified() throws Exception {
        rejectsUnchanged(new byte[]{1,2,3});
        rejectsUnchanged(ByteBuffer.allocate(8).putInt(7).putInt(0x6d6f6f76).array());
        rejectsUnchanged(ByteBuffer.allocate(16).putInt(1).putInt(0x6d6f6f76).putLong(Long.MAX_VALUE).array());
        rejectsUnchanged(box("moov",false,track(0,false,90,"vide"),track(0,false,90,"vide")));
        rejectsUnchanged(box("moov",false,track(0,false,0,"soun")));
        rejectsUnchanged(join(box("moov",false,track(0,false,90,"vide")),box("moof",false,new byte[16])));
        byte[] track=track(0,false,90,"vide");
        // tkhd starts at 8 and its payload/version byte at 16.
        track[16]=2;
        rejectsUnchanged(box("moov",false,track));
        rejectsUnchanged(box("moov",false,box("trak",false,box("tkhd",false,new byte[5]),
                box("mdia",false,box("hdlr",false,ByteBuffer.allocate(12).putInt(0).putInt(0).putInt(0x76696465).array())))));
    }
    @Test public void unchangedAngleIsByteIdenticalAndUnknownTransformRejected() throws Exception {
        byte[] bytes=box("moov",false,track(0,false,90,"vide"));
        File f=file(bytes);Mp4Orientation.set(f,90);assertArrayEquals(bytes,Files.readAllBytes(f.toPath()));
        // moov(8)+trak(8)+tkhd(8)+payload offset40 =64.
        ByteBuffer.wrap(bytes).putInt(64,123);rejectsUnchanged(bytes);
    }
    @Test(expected=IllegalArgumentException.class) public void rejectNonRightAngleBeforeWriting() throws Exception {
        Mp4Orientation.set(file(box("moov",false,track(0,false,90,"vide"))),13);
    }
}
