package dev.murillo.tkd.multicam;

import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.view.View;
import android.widget.TextView;
import androidx.test.core.app.ActivityScenario;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.File;
import java.io.FileOutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import static org.junit.Assert.*;
import static androidx.test.espresso.Espresso.onView;
import static androidx.test.espresso.Espresso.pressBack;
import static androidx.test.espresso.matcher.RootMatchers.isDialog;
import static androidx.test.espresso.matcher.ViewMatchers.withId;
import static androidx.test.espresso.matcher.ViewMatchers.isDisplayed;
import static androidx.test.espresso.assertion.ViewAssertions.matches;

@RunWith(AndroidJUnit4.class)
public class LightTelemetryTest {
    private LightMonitor.Snapshot sample(long seq, int band, boolean high, long now) {
        return new LightMonitor.Snapshot("test-session",seq,band==2?6_000_000:band==1?3_000_000:2_000_000,
                high?2000:640,band,high,true,"SYNTHETIC LIGHT TEST",60,now,12345678);
    }
    @Test public void wireRoundTripRejectsStaleInvalidAndOtherSessionData() throws Exception {
        long sender=10_000_000_000L, receiver=1_000_000_000_000L;
        JSONObject j=LightJson.encode(sample(7,1,true,sender),sender+200_000_000L);
        LightJson.Report r=LightJson.decode(j,"test-session",null,receiver);
        assertNotNull(r);assertEquals(2000,r.sample.iso);assertEquals(3_000_000,r.sample.exposureNs);
        assertTrue(r.at(receiver+2_000_000_000L).known);
        assertFalse(r.at(receiver+3_000_000_000L).known);
        assertSame(r,LightJson.decode(j,"test-session",r,receiver+1_000_000_000L));
        assertEquals(receiver,r.receivedNs);
        assertNull(LightJson.decode(j,"new-session",null,receiver));
        assertNull(LightJson.decode(new JSONObject("{\"v\":99}"),null,null,receiver));
        assertNull(LightJson.decode(null,null,null,receiver));
        assertNull(LightJson.decode(new JSONObject(j.toString()).put("iso",0),null,null,receiver));
        assertNull(LightJson.decode(new JSONObject(j.toString()).put("age_ms",-1),null,null,receiver));
        assertNull(LightJson.decode(new JSONObject(j.toString()).put("band",8),null,null,receiver));
        assertSame(r,LightJson.decode(LightJson.encode(sample(6,0,false,sender),sender),null,r,receiver+1));
        LightMonitor m=new LightMonitor();m.reset("x");m.observe(10,20,6_000_000,2400);
        JSONObject summary=LightJson.summary(m.summary());
        assertEquals(1,summary.getLong("severe_exposure_results"));
        assertEquals(1,summary.getLong("high_iso_results"));
    }

    @Test public void lightWarningOverlayAndDetailsDoNotResizeControls() throws Exception {
        InstrumentationRegistry.getInstrumentation().getUiAutomation();
        Context c=ApplicationProvider.getApplicationContext();
        int[] orientations={ActivityInfo.SCREEN_ORIENTATION_PORTRAIT,ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE};
        for(int orientation:orientations) {
            Intent intent=new Intent(c,MainActivity.class).putExtra("ui_test",true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(intent)) {
                scenario.onActivity(a -> a.setRequestedOrientation(orientation));
                SystemClock.sleep(1200);
                int[] height={0};
                scenario.onActivity(a -> height[0]=a.getWindow().getDecorView().findViewWithTag("action_start").getHeight());
                for(int band:new int[]{0,1,2}) {
                    LightMonitor.Snapshot snapshot=sample(10+band,band,band==2,SystemClock.elapsedRealtimeNanos());
                    scenario.onActivity(a -> {
                        try {
                            Field phase=MainActivity.class.getDeclaredField("phase");phase.setAccessible(true);
                            @SuppressWarnings({"unchecked","rawtypes"})
                            Object ready=Enum.valueOf((Class)phase.getType(),"READY");
                            phase.set(a,ready);
                            Field light=MainActivity.class.getDeclaredField("lightSnapshot");light.setAccessible(true);light.set(a,snapshot);
                            Method refresh=MainActivity.class.getDeclaredMethod("updateLightIndicator");refresh.setAccessible(true);refresh.invoke(a);
                            TextView indicator=a.getWindow().getDecorView().findViewWithTag("light_indicator");
                            assertEquals(View.VISIBLE,indicator.getVisibility());
                            assertTrue(indicator.getText().toString().contains(snapshot.label()));
                            assertTrue(indicator.getText().toString().contains("ISO "+snapshot.iso));
                            assertEquals(height[0],a.getWindow().getDecorView().findViewWithTag("action_start").getHeight());
                        } catch(ReflectiveOperationException e) { throw new AssertionError(e); }
                    });
                    SystemClock.sleep(350);
                    capture("light-"+(orientation==ActivityInfo.SCREEN_ORIENTATION_PORTRAIT?"portrait":"landscape")+"-"+band+".png");
                }
                scenario.onActivity(a -> a.getWindow().getDecorView().findViewWithTag("light_indicator").performClick());
                onView(withId(android.R.id.button3)).inRoot(isDialog()).check(matches(isDisplayed()));
                capture("light-dialog-"+orientation+".png");
                pressBack();
            }
        }
    }
    private void capture(String name) throws Exception {
        Bitmap b=InstrumentationRegistry.getInstrumentation().getUiAutomation().takeScreenshot();
        assertNotNull(b);
        File dir=new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getExternalFilesDir(null),"ui-proof");
        assertTrue(dir.isDirectory()||dir.mkdirs());
        try(FileOutputStream out=new FileOutputStream(new File(dir,name))) {b.compress(Bitmap.CompressFormat.PNG,100,out);}
        b.recycle();
    }
}
