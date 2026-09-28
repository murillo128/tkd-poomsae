package dev.murillo.tkd.multicam;

import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.media.MediaExtractor;
import android.media.MediaMetadataRetriever;
import android.os.SystemClock;
import android.view.View;
import android.widget.TextView;
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
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class RecordingBehaviorTest {
    private Field field(Class<?> c,String name) throws Exception {
        Field f=c.getDeclaredField(name);f.setAccessible(true);return f;
    }
    @Test public void startHasNoCountdownEvenWithoutLocalCamera() throws Exception {
        Context context=ApplicationProvider.getApplicationContext();
        Intent intent=new Intent(context,MainActivity.class).putExtra("ui_test",true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(intent)) {
            scenario.onActivity(a -> a.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT));
            SystemClock.sleep(700);
            scenario.onActivity(a -> {
                try {
                    Field phase=field(MainActivity.class,"phase");
                    @SuppressWarnings({"rawtypes","unchecked"}) Object ready=Enum.valueOf((Class)phase.getType(),"READY");
                    phase.set(a,ready);
                    field(MainActivity.class,"localEnabled").setBoolean(a,false);
                    field(MainActivity.class,"role").set(a,NetworkCoordinator.Role.CONTROLLER);
                    field(MainActivity.class,"sessionId").set(a,"test-immediate");
                    Method render=MainActivity.class.getDeclaredMethod("render");render.setAccessible(true);render.invoke(a);
                    long before=SystemClock.elapsedRealtimeNanos();
                    View button=a.getWindow().getDecorView().findViewWithTag("action_start");
                    assertFalse(button.getContentDescription().toString().contains("+3"));
                    assertTrue(button.performClick());
                    long after=SystemClock.elapsedRealtimeNanos();
                    long target=field(MainActivity.class,"scheduledNs").getLong(a);
                    assertTrue("target must be the click instant, never click+3s",target>=before&&target<=after);
                } catch(Exception e) {throw new AssertionError(e);}
            });
            InstrumentationRegistry.getInstrumentation().waitForIdleSync();
            scenario.onActivity(a -> {
                try {assertEquals("RECORDING",field(MainActivity.class,"phase").get(a).toString());}
                catch(Exception e){throw new AssertionError(e);}
            });
        }
    }
    @Test public void physicalOrientationOverridesPortraitUiLock() throws Exception {
        Context context=ApplicationProvider.getApplicationContext();
        Intent intent=new Intent(context,MainActivity.class).putExtra("ui_test",true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(intent)) {
            scenario.onActivity(a -> a.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT));
            SystemClock.sleep(600);
            scenario.onActivity(a -> {
                try {
                    DeviceOrientation tracker=(DeviceOrientation)field(MainActivity.class,"deviceOrientation").get(a);
                    tracker.close(); // deterministic injection, not simulated camera capture
                    field(DeviceOrientation.class,"quarter").setInt(tracker,270);
                    field(DeviceOrientation.class,"unknown").setBoolean(tracker,false);
                    field(DeviceOrientation.class,"measuredNs").setLong(tracker,SystemClock.elapsedRealtimeNanos());
                    RecordingOrientation.Choice choice=tracker.read(90);
                    assertEquals(0,choice.clockwiseDegrees);assertEquals("physical_sensor",choice.source);
                    assertEquals(0,a.getWindowManager().getDefaultDisplay().getRotation());
                } catch(Exception e){throw new AssertionError(e);}
            });
        }
    }
    private String rotation(File f) throws Exception {
        MediaMetadataRetriever reader=new MediaMetadataRetriever();
        try {reader.setDataSource(f.getPath());return reader.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);}
        finally {reader.release();}
    }
    private List<String> samples(File f) throws Exception {
        MediaExtractor ex=new MediaExtractor();List<String> samples=new ArrayList<>();
        try {
            ex.setDataSource(f.getPath());ex.selectTrack(0);ByteBuffer data=ByteBuffer.allocate(1024*128);
            while(ex.getSampleTime()>=0) {
                data.clear();int n=ex.readSampleData(data,0);assertTrue(n>0);
                byte[] bytes=new byte[n];data.position(0);data.get(bytes);
                samples.add(ex.getSampleTime()+":"+Base64.getEncoder().encodeToString(MessageDigest.getInstance("SHA-256").digest(bytes)));
                if(!ex.advance())break;
            }
            return samples;
        } finally {ex.release();}
    }
    @Test public void androidReadsNewRotationWithIdenticalH264SamplesAndPts() throws Exception {
        Context context=ApplicationProvider.getApplicationContext();
        File fixture=new File(context.getCacheDir(),"rotation.mp4");
        try(InputStream in=InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("rotation-fixture.mp4");
            OutputStream out=new FileOutputStream(fixture)) {
            byte[] b=new byte[4096];int n;while((n=in.read(b))!=-1)out.write(b,0,n);
        }
        List<String> original=samples(fixture);assertEquals(3,original.size());
        long length=fixture.length();
        for(int degree:new int[]{90,0,180,270}) {
            Mp4Orientation.set(fixture,degree);assertEquals(Integer.toString(degree),rotation(fixture));
            assertEquals(original,samples(fixture));assertEquals(length,fixture.length());
        }
        assertTrue(fixture.delete());
    }
}
