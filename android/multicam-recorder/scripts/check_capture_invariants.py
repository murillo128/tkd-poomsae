#!/usr/bin/env python3
"""Allow passive light-observer additions only; preserve all existing capture/control source."""
import hashlib
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = root / 'app/src/main/java/dev/murillo/tkd/multicam'
expected = {
    'CameraEngine.java': ('cfab1b12fa73841404441fb59952ed06b4befa23', 4),
    'NetworkCoordinator.java': ('dca3d6940092cb908f48d1c6a0d6f222697942ab', 6),
}
for name, (sha, count) in expected.items():
    source = (base / name).read_text()
    pattern = re.compile(r'^    // LIGHT_OBSERVER_BEGIN\n.*?^    // LIGHT_OBSERVER_END\n', re.M | re.S)
    additions = pattern.findall(source)
    assert len(additions) == count, f'{name}: unexpected observer block count'
    if name == 'CameraEngine.java':
        # Observers may only read results, accumulate bounded stats, expose a snapshot,
        # and append JSON metadata. No camera-request or recorder configuration here.
        assert not re.search(r'CaptureRequest|setRepeating|createCapture|setVideo|setCaptureRate|\.start\(|\.stop\(|\.set\(', ''.join(additions))
    original = pattern.sub('', source).encode()
    actual = hashlib.sha1(b'blob ' + str(len(original)).encode() + b'\0' + original).hexdigest()
    assert actual == sha, f'{name}: capture/transport code changed beyond observer additions: {actual}'
    print(f'PASS {name}: baseline preserved after removing {count} passive observer blocks')
# Preview source is deliberately outside this feature's scope.
for name, sha in [('CameraPreview.java', '110f93ddc71650b8cac497ef695db35deae6bc5d'),
                  ('PreviewGeometry.java', '14f82a3783ec24348872f8f6238600d1e123c22b')]:
    data=(base/name).read_bytes()
    actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert actual==sha, f'{name}: preview changed'
    print(f'PASS {name}: unchanged')
