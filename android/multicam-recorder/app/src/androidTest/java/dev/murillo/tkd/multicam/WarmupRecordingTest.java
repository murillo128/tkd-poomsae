package dev.murillo.tkd.multicam;

import android.content.Context;
import android.content.Intent;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaMuxer;
import android.os.Handler;
import android.view.TextureView;
import androidx.test.core.app.ActivityScenario;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.*;
import java.lang.reflect.*;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.*;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class WarmupRecordingTest {
    private File fixture(String name) throws Exception {
        Context c=ApplicationProvider.getApplicationContext();File dir=new File(c.getCacheDir(),"trim-test");dir.mkdirs();
        File seed=new File(dir,"seed-"+UUID.randomUUID()+".mp4");
        try(InputStream in=InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("rotation-fixture.mp4");
            OutputStream out=new FileOutputStream(seed)) {
            byte[] bytes=new byte[8192];int n;while((n=in.read(bytes))!=-1)out.write(bytes,0,n);
        }
        File f=new File(dir,name+"-"+UUID.randomUUID()+".mp4");
        MediaExtractor ex=new MediaExtractor();MediaMuxer mux=null;
        try {
            ex.setDataSource(seed.getPath());ex.selectTrack(0);
            android.media.MediaFormat format=ex.getTrackFormat(0);format.setInteger(android.media.MediaFormat.KEY_FRAME_RATE,120);
            List<byte[]> samples=new ArrayList<>();List<Integer> flags=new ArrayList<>();
            ByteBuffer b=ByteBuffer.allocate(65536);
            while(ex.getSampleTrackIndex()>=0) {
                b.clear();int n=ex.readSampleData(b,0);byte[] data=new byte[n];b.position(0);b.get(data);
                samples.add(data);flags.add(ex.getSampleFlags()&1);if(!ex.advance())break;
            }
            assertEquals(3,samples.size());assertEquals(Integer.valueOf(1),flags.get(0));
            mux=new MediaMuxer(f.getPath(),0);int track=mux.addTrack(format);mux.start();
            MediaCodec.BufferInfo info=new MediaCodec.BufferInfo();
            // Repeat a tiny synthetic closed GOP with 120fps PTS; no private footage.
            for(int i=0;i<240;i++) {
                byte[] data=samples.get(i%3);info.set(0,data.length,Math.round(i*1_000_000.0/120),flags.get(i%3));
                mux.writeSampleData(track,ByteBuffer.wrap(data),info);
            }
            mux.stop();mux.release();mux=null;
        }finally {ex.release();if(mux!=null)mux.release();seed.delete();}
        return f;
    }
    private String sampleHash(ByteBuffer b) throws Exception {
        MessageDigest d=MessageDigest.getInstance("SHA-256");d.update(b);return Base64.getEncoder().encodeToString(d.digest());
    }
    private List<String> hashes(File file,long seek) throws Exception {
        MediaExtractor ex=new MediaExtractor();List<String> result=new ArrayList<>();
        try {ex.setDataSource(file.getPath());ex.selectTrack(0);ex.seekTo(seek,MediaExtractor.SEEK_TO_PREVIOUS_SYNC);
            ByteBuffer b=ByteBuffer.allocate(512*1024);
            while(ex.getSampleTrackIndex()>=0) {b.clear();int n=ex.readSampleData(b,0);b.position(0);b.limit(n);result.add(sampleHash(b));if(!ex.advance())break;}
            return result;
        }finally{ex.release();}
    }
    private void decodeAll(File file) throws Exception {
        MediaExtractor ex=new MediaExtractor();MediaCodec decoder=null;
        try {
            ex.setDataSource(file.getPath());ex.selectTrack(0);
            decoder=MediaCodec.createDecoderByType("video/avc");decoder.configure(ex.getTrackFormat(0),null,null,0);decoder.start();
            MediaCodec.BufferInfo info=new MediaCodec.BufferInfo();boolean inputEnded=false,outputEnded=false;int frames=0;
            long deadline=android.os.SystemClock.elapsedRealtime()+15000;
            while(!outputEnded && android.os.SystemClock.elapsedRealtime()<deadline) {
                if(!inputEnded) {
                    int i=decoder.dequeueInputBuffer(1000);
                    if(i>=0) {
                        ByteBuffer b=decoder.getInputBuffer(i);int size=ex.readSampleData(b,0);
                        if(size<0) {decoder.queueInputBuffer(i,0,0,0,MediaCodec.BUFFER_FLAG_END_OF_STREAM);inputEnded=true;}
                        else {decoder.queueInputBuffer(i,0,size,ex.getSampleTime(),0);ex.advance();}
                    }
                }
                int out=decoder.dequeueOutputBuffer(info,1000);
                if(out>=0) {if(info.size>0)frames++;outputEnded=(info.flags&MediaCodec.BUFFER_FLAG_END_OF_STREAM)!=0;decoder.releaseOutputBuffer(out,false);}
            }
            assertTrue("decoder reaches EOS without corrupt start",outputEnded);assertTrue(frames>0);
        }finally {if(decoder!=null){try{decoder.stop();}catch(Exception ignored){}decoder.release();}ex.release();}
    }
    @Test public void remuxKeepsPreviousSyncAndPayloadsAt120fpsWithRotation() throws Exception {
        File source=fixture("source");String originalHash=hashes(source,0).toString();
        Mp4Orientation.set(source,90);
        for(long requested:new long[]{0,500000,735000,1750000}) {
            File out=new File(source.getParentFile(),"final-"+UUID.randomUUID()+".mp4");
            RecordingTrim.Result r=RecordingTrim.remux(source,out,requested);
            assertTrue(r.firstSourcePtsUs<=requested);assertTrue(r.retainedLeadUs>=0);
            assertTrue(r.retainedLeadUs<510000);assertTrue(r.maxPtsRoundingUs<=25);
            assertEquals(hashes(source,r.firstSourcePtsUs),hashes(out,0));
            assertEquals(90,Mp4Orientation.read(out));
            if(requested==735000) {assertEquals(725000,r.discardedUs,25);assertEquals(10000,r.retainedLeadUs,25);assertEquals(153,r.frames);}
            decodeAll(out);assertTrue(out.delete());
        }
        assertEquals(originalHash,hashes(source,0).toString());assertTrue(source.delete());
    }
    @Test public void bPicturePtsBeforeSyncArePreservedNotDropped() throws Exception {
        File plain=fixture("plain"),reordered=new File(plain.getParentFile(),"reordered-"+UUID.randomUUID()+".mp4");
        MediaExtractor ex=new MediaExtractor();MediaMuxer mux=null;
        try {
            ex.setDataSource(plain.getPath());ex.selectTrack(0);
            mux=new MediaMuxer(reordered.getPath(),0);int track=mux.addTrack(ex.getTrackFormat(0));mux.start();
            ByteBuffer b=ByteBuffer.allocate(512*1024);MediaCodec.BufferInfo info=new MediaCodec.BufferInfo();int index=0;
            while(ex.getSampleTrackIndex()>=0) {
                b.clear();int n=ex.readSampleData(b,0);long pts=ex.getSampleTime();
                // Reordered timestamps exercise the S21-style leading-PTS case;
                // samples themselves are synthetic P pictures, not user footage.
                if(index>=61&&index<=62)pts=483333+(index-61)*8333;
                b.position(0);b.limit(n);info.set(0,n,pts,(ex.getSampleFlags()&1)!=0?1:0);mux.writeSampleData(track,b,info);
                index++;if(!ex.advance())break;
            }
            mux.stop();mux.release();mux=null;
        }finally {ex.release();if(mux!=null)mux.release();}
        File out=new File(plain.getParentFile(),"final-"+UUID.randomUUID()+".mp4");
        RecordingTrim.Result r=RecordingTrim.remux(reordered,out,513000);
        assertEquals(hashes(reordered,r.firstSourcePtsUs),hashes(out,0));assertTrue(r.frames>=180);
        assertTrue(r.maxPtsRoundingUs<=25);decodeAll(out);
        assertTrue(plain.delete());assertTrue(reordered.delete());assertTrue(out.delete());
    }
    @Test public void invalidTrimRetainsSourceAndDoesNotLeaveFinalOrPartial() throws Exception {
        File source=fixture("short"),out=new File(source.getParentFile(),"failed-"+UUID.randomUUID()+".mp4");
        List<String> before=hashes(source,0);
        try{RecordingTrim.remux(source,out,5000000);fail("No frames after START");}catch(IOException expected){}
        assertFalse(out.exists());assertFalse(new File(out+".partial").exists());assertEquals(before,hashes(source,0));
        assertTrue(source.delete());
    }
    private Field field(String name) throws Exception {Field f=CameraEngine.class.getDeclaredField(name);f.setAccessible(true);return f;}
    @Test public void engineStopWhileArmedDiscardsInsteadOfReportingSaved() throws Exception {
        Context context=ApplicationProvider.getApplicationContext();
        Intent intent=new Intent(context,MainActivity.class).putExtra("ui_test",true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        CountDownLatch done=new CountDownLatch(1);List<String> events=Collections.synchronizedList(new ArrayList<>());
        CameraEngine[] engine={null};WarmupStore[] store={null};File[] finalFile={null};
        try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(intent)) {
            scenario.onActivity(a -> {
                engine[0]=new CameraEngine(a,(TextureView)a.getWindow().getDecorView().findViewWithTag("camera_preview"),new CameraEngine.Listener(){
                    public void onCameraStatus(String s){}
                    public void onCameraReady(String s){events.add("ready");}
                    public void onCameraStarted(long t){events.add("started");}
                    public void onCameraStopped(String p,String j){events.add("saved");done.countDown();}
                    public void onCameraError(String s){events.add("error:"+s);done.countDown();}
                    public void onCameraDiscarded(String s){events.add("discarded");done.countDown();}
                });
                try {
                    Handler h=(Handler)field("cameraHandler").get(engine[0]);
                    h.post(() -> {try{
                        store[0]=new WarmupStore(new File(a.getFilesDir(),"test-cancel-"+UUID.randomUUID()));
                        finalFile[0]=new File(a.getCacheDir(),"not-created-"+UUID.randomUUID()+".mp4");
                        field("warmup").set(engine[0],store[0]);field("videoFile").set(engine[0],finalFile[0]);
                        field("state").set(engine[0],CameraEngine.State.READY);
                        engine[0].stop();
                    }catch(Exception e){events.add("setup:"+e);done.countDown();}});
                }catch(Exception e){throw new AssertionError(e);}
            });
            assertTrue(done.await(5,TimeUnit.SECONDS));assertEquals(Collections.singletonList("discarded"),events);
            assertFalse(store[0].file().exists());assertFalse(finalFile[0].exists());assertEquals(CameraEngine.State.IDLE,engine[0].getState());
            engine[0].shutdown();
        }
    }
}
