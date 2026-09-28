package dev.murillo.tkd.multicam;

import java.io.File;
import java.nio.file.Files;
import org.junit.Test;
import static org.junit.Assert.*;

public class WarmupStoreTest {
    @Test public void armWithoutStartLeavesNoRecordingOrMarker() throws Exception {
        File dir=Files.createTempDirectory("warmup-test").toFile();
        WarmupStore warmup=new WarmupStore(dir);File file=warmup.file();
        Files.write(file.toPath(),new byte[]{1,2,3});
        assertFalse(warmup.hasStarted());warmup.discardUnlessStarted();
        assertFalse(file.exists());assertEquals(0,dir.list().length);assertTrue(dir.delete());
    }
    @Test public void startedRecoverySurvivesCleanupUntilSuccessfulSave() throws Exception {
        File dir=Files.createTempDirectory("warmup-test").toFile();
        WarmupStore warmup=new WarmupStore(dir);File file=warmup.file();
        warmup.markStarted();warmup.markStarted();assertTrue(warmup.hasStarted());
        warmup.discardUnlessStarted();WarmupStore.cleanupAbandoned(dir);
        assertTrue(file.exists());assertTrue(new File(file+".started").exists());
        warmup.discard();assertEquals(0,dir.list().length);assertTrue(dir.delete());
    }
    @Test public void processRestartDeletesOnlyAbandonedOwnedWarmup() throws Exception {
        File dir=Files.createTempDirectory("warmup-test").toFile();
        File privateUser=new File(dir,"not-ours.mp4");Files.write(privateUser.toPath(),new byte[]{42});
        File abandoned=new File(dir,"warmup-abandoned.mp4");Files.write(abandoned.toPath(),new byte[]{8});
        File folder=new File(dir,"warmup-directory.mp4");assertTrue(folder.mkdir());
        File orphan=new File(dir,"warmup-orphan.mp4.started");Files.write(orphan.toPath(),new byte[]{9});
        WarmupStore.cleanupAbandoned(dir);
        assertFalse(abandoned.exists());assertFalse(orphan.exists());
        assertArrayEquals(new byte[]{42},Files.readAllBytes(privateUser.toPath()));assertTrue(folder.exists());
        assertTrue(privateUser.delete());assertTrue(folder.delete());assertTrue(dir.delete());
    }
    @Test public void laterInstancesDoNotDeleteAnotherLiveTemporary() throws Exception {
        File dir=Files.createTempDirectory("warmup-test").toFile();
        WarmupStore first=new WarmupStore(dir),second=new WarmupStore(dir);
        assertTrue(first.file().exists());assertTrue(second.file().exists());
        first.discard();second.discard();assertTrue(dir.delete());
    }
    @Test public void oldCallbacksAndTimersCannotMatchNewSession() {
        SessionEpoch epoch=new SessionEpoch();long arm=epoch.advance();assertTrue(epoch.matches(arm));
        epoch.advance();assertFalse(epoch.matches(arm));
        long next=epoch.advance();assertTrue(epoch.matches(next));assertFalse(epoch.matches(arm));
    }
}
