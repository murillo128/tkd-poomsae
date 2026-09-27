package dev.murillo.tkd.multicam;

import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.os.SystemClock;
import android.view.Surface;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.TextView;

import androidx.test.core.app.ActivityScenario;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.File;
import java.io.FileOutputStream;
import java.util.concurrent.atomic.AtomicBoolean;
import static org.junit.Assert.*;
import static androidx.test.espresso.Espresso.onView;
import static androidx.test.espresso.Espresso.pressBack;
import static androidx.test.espresso.matcher.RootMatchers.isDialog;
import static androidx.test.espresso.matcher.ViewMatchers.withId;
import static androidx.test.espresso.matcher.ViewMatchers.isDisplayed;
import static androidx.test.espresso.assertion.ViewAssertions.matches;

/** UI and synthetic SurfaceTexture checks; does NOT pretend to test the S21 camera HAL. */
@RunWith(AndroidJUnit4.class)
public class DashboardUiTest {
    @Test public void nativeLayoutsDialogsAndRoundPreviewInAllFourOrientations() throws Exception {
        // Establish the screenshot connection before opening any transient windows.
        InstrumentationRegistry.getInstrumentation().getUiAutomation();
        int[] orientations={ActivityInfo.SCREEN_ORIENTATION_PORTRAIT,ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE,
                ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE,ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT};
        int[] rotations={Surface.ROTATION_0,Surface.ROTATION_90,Surface.ROTATION_270,Surface.ROTATION_180};
        String[] names={"portrait","landscape","reverse-landscape","reverse-portrait"};
        Context context=ApplicationProvider.getApplicationContext();
        for(int index=0; index<orientations.length; index++) {
            Intent intent=new Intent(context,MainActivity.class).putExtra("ui_test",true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(intent)) {
                int requested=orientations[index], expected=rotations[index];
                scenario.onActivity(a -> a.setRequestedOrientation(requested));
                AtomicBoolean ready=new AtomicBoolean(false);
                for(int attempt=0;attempt<50&&!ready.get();attempt++) {
                    SystemClock.sleep(200);
                    scenario.onActivity(a -> ready.set(a.getDisplay().getRotation()==expected
                            && ((CameraPreview) a.getWindow().getDecorView().findViewWithTag("camera_preview")).isAvailable()));
                }
                assertTrue("orientation/surface ready: "+names[index],ready.get());
                InstrumentationRegistry.getInstrumentation().waitForIdleSync();
                saveScreenshot(names[index]+"-layout.png").recycle();
                scenario.onActivity(a -> {
                    View root=a.getWindow().getDecorView();
                    for(String tag:new String[]{"brand_title","status_role","status_network","status_camera","status_session",
                            "action_discover","action_arm","action_start","action_stop","preview_fit"}) {
                        View v=root.findViewWithTag(tag);
                        assertNotNull(tag,v);
                        assertTrue(tag+" has width",v.getWidth()>0);
                        assertTrue(tag+" has height",v.getHeight()>0);
                        assertTrue(tag+" participates in layout",v.isShown());
                    }
                    root.findViewWithTag("status_network").performClick();
                });
                // Espresso synchronizes with the real dialog window; a first immediate
                // UiAutomation accessibility-root read can be null while attaching.
                onView(withId(android.R.id.button3)).inRoot(isDialog()).check(matches(isDisplayed()));
                saveScreenshot(names[index]+"-dialog.png").recycle();
                pressBack();
                InstrumentationRegistry.getInstrumentation().waitForIdleSync();

                for(boolean fill:new boolean[]{false,true}) {
                    Rect[] area={new Rect()};
                    scenario.onActivity(a -> {
                        FrameLayout frame=a.getWindow().getDecorView().findViewWithTag("preview_frame");
                        CameraPreview preview=frame.findViewWithTag("camera_preview");
                        frame.getChildAt(1).setVisibility(View.GONE);
                        ((TextView) frame.getChildAt(2)).setText("SYNTHETIC GEOMETRY TEST");
                        preview.setFill(fill);
                        SurfaceTexture st=preview.getSurfaceTexture();
                        assertNotNull(st);
                        // Simulate the producer's naturally-oriented output. The camera HAL
                        // already applies sensor mounting rotation; the UI must not add it twice.
                        int nw=preview.getSensorDegrees()%180==0?1920:1080;
                        int nh=preview.getSensorDegrees()%180==0?1080:1920;
                        st.setDefaultBufferSize(nw,nh);
                        Surface surface=new Surface(st);
                        Canvas c=null;
                        try {
                            c=surface.lockCanvas(null);
                            c.drawColor(0xff122331);
                            Paint paint=new Paint(); paint.setColor(0xff537889); paint.setStrokeWidth(3);
                            for(int x=0;x<nw;x+=120)c.drawLine(x,0,x,nh,paint);
                            for(int y=0;y<nh;y+=120)c.drawLine(0,y,nw,y,paint);
                            paint.setColor(Color.GREEN);
                            c.drawCircle(nw/2f,nh/2f,Math.min(nw,nh)*.17f,paint);
                            paint.setColor(Color.WHITE); paint.setTextSize(50);
                            c.drawText("NATURAL SOURCE / NOT CAMERA",24,75,paint);
                        } catch(Exception e) { throw new AssertionError(e); }
                        finally { if(c!=null)surface.unlockCanvasAndPost(c); surface.release(); }
                        preview.refreshTransform();
                        assertTrue(preview.getGlobalVisibleRect(area[0]));
                    });
                    SystemClock.sleep(800);
                    Bitmap bitmap=saveScreenshot(names[index]+(fill?"-fill.png":"-fit.png"));
                    Rect r=area[0];
                    int minX=Integer.MAX_VALUE,minY=Integer.MAX_VALUE,maxX=-1,maxY=-1;
                    for(int y=Math.max(0,r.top);y<Math.min(bitmap.getHeight(),r.bottom);y++)
                        for(int x=Math.max(0,r.left);x<Math.min(bitmap.getWidth(),r.right);x++) {
                            int pixel=bitmap.getPixel(x,y);
                            if(Color.green(pixel)>245&&Color.red(pixel)<10&&Color.blue(pixel)<10) {
                                minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
                            }
                        }
                    assertTrue("green diagnostic circle visible",maxX-minX>20&&maxY-minY>20);
                    assertEquals("rendered source circle remains round: "+names[index]+" fill="+fill,
                            maxX-minX,maxY-minY,3.0);
                    bitmap.recycle();
                }
            }
        }
    }

    private Bitmap saveScreenshot(String name) throws Exception {
        Bitmap bitmap=InstrumentationRegistry.getInstrumentation().getUiAutomation().takeScreenshot();
        assertNotNull(bitmap);
        File directory=new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getExternalFilesDir(null),"ui-proof");
        assertTrue(directory.isDirectory()||directory.mkdirs());
        try(FileOutputStream out=new FileOutputStream(new File(directory,name))) {
            assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG,100,out));
        }
        return bitmap;
    }
}
