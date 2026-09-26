import { describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import { buildScene, calibratedFrustum, defaultLayers, disposeGroup, handEdges, footEdges, sampleAt, worldTuple, worldVector } from './sceneGeometry'
import type { GeometryData } from './geometryApi'
import type { Landmark } from '../../contracts/types'
const q = { state: 'observed' as const, score: .9 }
export function fixture(): GeometryData {
  const names = [...new Set([...handEdges.flat(), ...footEdges.flat(), 'pelvis', 'spine', 'neck', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'])] as Landmark[]
  return { revision: 'revision', unit: 'm', reasons: [], truncated: false, actions: [], calibration: {
    id: 'cal', kind: 'calibration', schema_version: '1.0.0', provenance: { producer: 'fixture', config_digest: '0' }, scale: 'metric', world_unit: 'm', ground_z: 0, ground_status: 'resolved', quality: q,
    cameras: [{ camera_id: 'front', source_id: 'front', intrinsics: { fx: 100, fy: 100, cx: 50, cy: 50 }, world_to_camera: [[1,0,0,-1],[0,1,0,-2],[0,0,1,-3],[0,0,0,1]], quality: q }],
  }, samples: [0, .5, 1].map(time => ({ id: `motion/samples/${time * 2}`, global_seconds: time, root_xyz_world: [time,0,.8], root_orientation: null, quality: q,
    segments: [{ segment: 'head', parent: 'world', orientation: { wxyz: [1,0,0,0] }, quality: q }],
    landmarks: names.map((name, i) => ({ name, xyz_world: [time + i * .01,0,1] as [number,number,number], quality: q })) })) }
}
describe('canonical scene', () => {
  it('preserves Z-up coordinates and handedness; inverts world-to-camera positions', () => {
    expect(worldTuple(worldVector([1,2,3]))).toEqual([1,2,3])
    expect(worldVector([1,0,0]).cross(worldVector([0,1,0])).toArray()).toEqual([0,0,1])
    const result = calibratedFrustum(fixture().calibration!.cameras[0], [100, 100])
    expect(result.center.toArray()).toEqual([1,2,3])
    expect(result.corners[0].toArray()).toEqual([.8,1.8,3.4])
  })
  it('renders both complete hands and detailed feet without mutating the artifact', () => {
    const data = fixture(), original = JSON.stringify(data)
    const group = buildScene(data, 0, null, defaultLayers)
    expect(handEdges).toHaveLength(40); expect(footEdges).toHaveLength(10)
    expect(group.children.filter(object => object.userData.layer === 'Hands' && object instanceof THREE.Line)).toHaveLength(40)
    expect(group.children.filter(object => object.userData.layer === 'Feet' && object instanceof THREE.Line)).toHaveLength(10)
    expect(JSON.stringify(data)).toBe(original)
    disposeGroup(group)
  })
  it('omits null/masked joints and attached edges; marks uncertain geometry; never fills time gaps', () => {
    const data = fixture()
    const wrist = data.samples[0].landmarks.find(point => point.name === 'left_wrist')!
    wrist.xyz_world = null
    data.samples[0].landmarks.find(point => point.name === 'right_wrist')!.quality = { state: 'unknown', score: .1 }
    const group = buildScene(data, 0, { kind: 'track', id: 'left_arm', tracks: ['left_arm'] }, defaultLayers)
    expect(group.children.some(object => object.userData.id?.endsWith('/left_wrist'))).toBe(false)
    expect((group.children.find(object => object.name === 'right_wrist' && object instanceof THREE.Mesh) as THREE.Mesh).material).toMatchObject({ wireframe: true })
    expect(sampleAt(data.samples, .25)).toBeUndefined()
    disposeGroup(group)
  })
  it('respects every layer and disposes geometry/materials', () => {
    const group = buildScene(fixture(), 0, null, { ...defaultLayers, Hands: false, Feet: false, Cameras: false })
    expect(group.children.some(object => ['Hands','Feet','Cameras'].includes(object.userData.layer))).toBe(false)
    const mesh = group.children.find(object => object instanceof THREE.Mesh) as THREE.Mesh
    const dispose = vi.spyOn(mesh.geometry, 'dispose')
    disposeGroup(group)
    expect(dispose).toHaveBeenCalledOnce(); expect(group.children).toHaveLength(0)
  })
})
