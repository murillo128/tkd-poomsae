import type { Landmark, Quality } from '../../contracts/types'
import { bodyEdges, handEdges, footEdges } from '../src/sceneGeometry'
import type { GeometryData } from '../src/geometryApi'
const observed: Quality = { state: 'observed', score: .95, source_ids: ['native-front'] }
export const fixture: GeometryData = {
  revision: 'r1', unit: 'm', reasons: [], truncated: false,
  calibration: { id: 'cal', kind: 'calibration', schema_version: '1.0.0', provenance: { producer: 'offline synthetic', config_digest: '0'.repeat(64) },
    scale: 'metric', world_unit: 'm', ground_z: 0, ground_status: 'resolved', quality: observed,
    cameras: [{ camera_id: 'front', source_id: 'front', intrinsics: { fx: 600, fy: 600, cx: 320, cy: 240 },
      world_to_camera: [[1,0,0,0],[0,0,-1,1],[0,1,0,2],[0,0,0,1]], quality: observed }] },
  actions: [{ id: 'left-arm-action', interval: { start: 0, end: 1 }, step_id: 'step', tracks: ['left_arm'], category: 'arm' }],
  samples: [0, .5, 1].map((time, index) => ({ id: `motion/samples/${index}`, global_seconds: time, root_xyz_world: [time*.2,0,.9], root_orientation: null, quality: observed,
    segments: [{ segment: 'head', parent: 'world', orientation: { wxyz: [1,0,0,0] }, quality: observed }],
    landmarks: [...new Set([...bodyEdges.flat(), ...handEdges.flat(), ...footEdges.flat()])].map(name => ({ name, xyz_world: position(name, time),
      quality: name === 'right_index_tip' ? { state: 'unknown', score: .1 } : name.startsWith('left_thumb') ? { state: 'interpolated', score: .6 } : name === 'right_ankle' ? { state: 'observed', score: .2 } : observed })),
  })),
}
function position(name: Landmark, time: number): [number,number,number] | null {
  if (name === 'right_pinky_tip') return null
  const side = name.startsWith('left') ? -1 : 1
  const shift = time * .2
  if (name === 'pelvis') return [shift,0,.9]
  if (name === 'spine') return [shift,0,1.2]
  if (name === 'neck') return [shift,0,1.5]
  if (/head|nose|eye|ear/.test(name)) return [shift + (/left/.test(name) ? -.06 : /right/.test(name) ? .06 : 0), name === 'nose' ? -.1 : 0, 1.68]
  if (/shoulder/.test(name)) return [shift+side*.23,0,1.45]
  if (/elbow/.test(name)) return [shift+side*.42,0,1.2]
  if (/wrist/.test(name)) return [shift+side*.55,-.06,1.02]
  if (/hip/.test(name)) return [shift+side*.12,0,.88]
  if (/knee/.test(name)) return [shift+side*.17,0,.45]
  if (/ankle/.test(name)) return [shift+side*.2,0,.1]
  if (/heel/.test(name)) return [shift+side*.2,.08,.025]
  if (/forefoot/.test(name)) return [shift+side*.2,-.16,.025]
  if (/foot_outer/.test(name)) return [shift+side*.26,-.12,.025]
  const finger = ['thumb','index','middle','ring','pinky'].findIndex(value => name.includes(value))
  const part = name.split('_').at(-1)!
  const joint = part === 'tip' ? 4 : part === 'dip' || part === 'ip' ? 3 : part === 'pip' ? 2 : part === 'mcp' ? 1 : 0
  return [shift+side*(.55+joint*.022), -.06+(finger-2)*.025, 1.02-joint*.02]
}
