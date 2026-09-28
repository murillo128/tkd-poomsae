#!/usr/bin/env python3
"""APK 20 approved storage/lifecycle changes; guard original capture/encoder controls."""
import hashlib
import re
from pathlib import Path
base=Path(__file__).resolve().parents[1]/'app/src/main/java/dev/murillo/tkd/multicam'
source=(base/'CameraEngine.java').read_text()
pattern=r'(?:mediaRecorder\s*\.\s*(?:setVideoSource|setOutputFormat|setVideoEncoder|setVideoSize|setVideoFrameRate|setVideoEncodingBitRate|prepare|getSurface)|builder\s*\.\s*(?:set|addTarget)|cameraDevice\s*\.\s*createCaptureRequest|highSpeedSession\s*\.\s*(?:createHighSpeedRequestList|setRepeatingBurst)|outputs\s*\.\s*add)\s*\([^;]*?\);'
normalized='\n'.join(re.sub(r'\s+','',v) for v in re.findall(pattern,source))
assert hashlib.sha256(normalized.encode()).hexdigest()=='74f3bb33783b0bea5d004f3a11a71c65241789f7a9b30439eae45a231adda8cb', 'Camera2/encoder setup changed'
for literal in ['String CAMERA_ID = "0"','int WIDTH = 1920','int HEIGHT = 1080','int FPS = 120']:
    assert literal in source, literal
assert 'setCaptureRate' not in source and 'CONTROL_AE_MODE_OFF' not in source
burst=source.index('highSpeedSession.setRepeatingBurst(')
assert source.index('mediaRecorder.start();',burst)>burst
assert 'mediaRecorder.setOutputFile(warmup.file().getAbsolutePath())' in source
assert source.index('if (!keepRecording)')<source.index('RecordingTrim.remux')<source.index('String galleryPath = publishVideoToGallery()')
assert 'if (!isArming(token)) { session.close(); return; }' in source
assert 'if (!isArming(token)) { camera.close(); return; }' in source
assert 'ARM_LIMIT_MS = 180_000L' in source
print('PASS original Camera2/MediaRecorder configuration, continuous 120fps order, and discard-before-publish path')
for name,expected in {'CameraPreview.java': 'b5bd72d41789188e17c2f6a13cba9b716edb3b160b000405f88b9150f8715497', 'PreviewGeometry.java': '9921f98963d0a913108a44c80e35194fc75ab1c2add54bd4f187a8014f7e27cd', 'LightMonitor.java': '175f8a39820e789646de24af866edb553ff8a6c2faf4458d3d9696a6a856de47', 'LightJson.java': '0b5e1167199e7ba95f6af5c433d37deb203ad4ea04fad76baf509b062b627bfc', 'Mp4Orientation.java': 'b7ba5c6a178b1da7bffe7cd4d62e47f2d916cd8502223457f918e7f7b9f7ebb2', 'RecordingOrientation.java': 'b68b449aab1fcf74259f42046b64ef24b1a0416d2b19970e392afb82f52359c6', 'DeviceOrientation.java': 'baf294f47b31d9d13e2577df6e59c6316411870ca0d2a919104a52f1c976a971'}.items():
    assert hashlib.sha256((base/name).read_bytes()).hexdigest()==expected, name+' changed'
    print('PASS unchanged '+name)
main=(base/'MainActivity.java').read_text()
assert 'network.startAll(target, 0)' in main
assert 'scheduledNs = SystemClock.elapsedRealtimeNanos();' in main
assert 'START +3S' not in main
assert 'onCameraDiscarded' in main and 'network.sendDiscarded()' in main
print('PASS immediate START, explicit cancellation UI, remote discard notification')
