package dev.murillo.tkd.multicam;

import android.content.Context;
import android.view.View;
import android.widget.TextView;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class LightLayoutTest {
    @Test public void repeatedLightOverlayWidthDoesNotRequestAnotherLayout() {
        InstrumentationRegistry.getInstrumentation().runOnMainSync(() -> {
            Context context = ApplicationProvider.getApplicationContext();
            TextView text = new StudioUi(context).text("LOW LIGHT", 10, StudioUi.AMBER, false);
            text.setMaxWidth(160);
            text.measure(View.MeasureSpec.makeMeasureSpec(300, View.MeasureSpec.AT_MOST),
                    View.MeasureSpec.makeMeasureSpec(80, View.MeasureSpec.EXACTLY));
            text.layout(0, 0, text.getMeasuredWidth(), text.getMeasuredHeight());
            assertFalse(text.isLayoutRequested());
            text.setMaxWidth(160);
            assertFalse("same width must not cause a parent-layout feedback loop",text.isLayoutRequested());
            text.setMaxWidth(120);
            assertTrue("changed width must still remeasure",text.isLayoutRequested());
            assertEquals(120,text.getMaxWidth());
            text.setMaxEms(8);
            text.setMaxWidth(120);
            assertEquals("switching from ems to pixels must still work",120,text.getMaxWidth());
        });
    }
}
