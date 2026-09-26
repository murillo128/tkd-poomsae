import * as THREE from 'three'
import type { CameraCalibration, Landmark, Quality, Track } from '../../contracts/types'
import type { GeometryData, Sample } from './geometryApi'
import type { Selection } from './playback'

// Identity boundary: Three supports Z-up. No axis swap/reflection or stored mutation.
export function worldVector(xyz: readonly number[]) { return new THREE.Vector3(xyz[0], xyz[1], xyz[2]) }
export function worldTuple(point: THREE.Vector3): [number, number, number] { return [point.x, point.y, point.z] }
export const layerNames = ['Body', 'Hands', 'Feet', 'Head orientation', 'Ground', 'Root trajectory', 'Selected trajectories', 'Cameras'] as const
export type Layer = typeof layerNames[number]
export type Layers = Record<Layer, boolean>
export const defaultLayers = Object.fromEntries(layerNames.map(name => [name, true])) as Layers
export const bodyEdges: [Landmark, Landmark][] = [
  ['pelvis', 'spine'], ['spine', 'neck'], ['neck', 'head'], ['head', 'nose'],
  ['nose', 'left_eye'], ['nose', 'right_eye'], ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
  ['left_shoulder', 'right_shoulder'], ['left_hip', 'right_hip'],
  ...(['left', 'right'] as const).flatMap(side => [
    ['neck', `${side}_shoulder`], [`${side}_shoulder`, `${side}_elbow`], [`${side}_elbow`, `${side}_wrist`],
    ['pelvis', `${side}_hip`], [`${side}_hip`, `${side}_knee`], [`${side}_knee`, `${side}_ankle`],
  ] as [Landmark, Landmark][]),
]
export const handEdges: [Landmark, Landmark][] = (['left', 'right'] as const).flatMap(side =>
  ['thumb', 'index', 'middle', 'ring', 'pinky'].flatMap(finger => {
    const parts = finger === 'thumb' ? ['cmc', 'mcp', 'ip', 'tip'] : ['mcp', 'pip', 'dip', 'tip']
    const chain = [`${side}_wrist`, ...parts.map(part => `${side}_${finger}_${part}`)] as Landmark[]
    return chain.slice(1).map((point, i) => [chain[i], point] as [Landmark, Landmark])
  }))
export const footEdges: [Landmark, Landmark][] = (['left', 'right'] as const).flatMap(side => [
  [`${side}_ankle`, `${side}_heel`], [`${side}_heel`, `${side}_forefoot`],
  [`${side}_forefoot`, `${side}_foot_outer`], [`${side}_foot_outer`, `${side}_heel`],
  [`${side}_ankle`, `${side}_forefoot`],
] as [Landmark, Landmark][])
export function trackFor(name: string): Track {
  if (/eye|ear|head|nose/.test(name)) return 'head'
  if (name.startsWith('left_')) return /hip|knee|ankle|heel|foot/.test(name) ? 'left_leg' : 'left_arm'
  if (name.startsWith('right_')) return /hip|knee|ankle|heel|foot/.test(name) ? 'right_leg' : 'right_arm'
  return 'body_root'
}
export function evidenceColor(q: Quality): number {
  if (q.state === 'unknown') return 0xa7aab1
  if (q.state === 'interpolated') return 0xcb94ff
  if (q.state === 'inferred') return 0xffb45e
  if (q.score == null || q.score < 0.5) return 0xff7373
  return 0x73d8ef
}
export function calibratedFrustum(camera: CameraCalibration, size: [number, number]): { center: THREE.Vector3; corners: THREE.Vector3[] } {
  const m = new THREE.Matrix4().set(...camera.world_to_camera.flat() as Parameters<THREE.Matrix4['set']>).invert()
  const { fx, fy, cx, cy } = camera.intrinsics
  // Pinhole frustum from registered oriented image bounds; distortion is not applied.
  return { center: new THREE.Vector3().applyMatrix4(m), corners: [[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]].map(([u, v]) =>
    new THREE.Vector3((u - cx) / fx * .4, (v - cy) / fy * .4, .4).applyMatrix4(m)) }
}
export function sampleAt(samples: Sample[], seconds: number): Sample | undefined {
  // Show only a native instant: never silently hold or fabricate a pose across a gap.
  return samples.find(sample => Math.abs(sample.global_seconds - seconds) < 1e-6)
}
export function disposeGroup(group: THREE.Object3D) {
  group.traverse(object => {
    const renderable = object as THREE.Mesh
    renderable.geometry?.dispose()
    if (renderable.material) (Array.isArray(renderable.material) ? renderable.material : [renderable.material]).forEach(material => material.dispose())
  })
  group.clear()
}
export function buildScene(data: GeometryData, seconds: number, selection: Selection | null, layers: Layers) {
  const group = new THREE.Group(), sample = sampleAt(data.samples, seconds)
  const activeAction = data.actions.find(action => action.id === selection?.id)
  const selectedName = selection?.id.split('/').at(-1)
  const highlighted = (name: string) => name === selectedName || Boolean(selection?.tracks.includes(trackFor(name)))
  function line(points: THREE.Vector3[], quality: Quality, name: string, layer: Layer) {
    if (!layers[layer]) return
    const geometry = new THREE.BufferGeometry().setFromPoints(points)
    const color = highlighted(name) ? 0xffff55 : evidenceColor(quality)
    const material = quality.state === 'observed' && quality.score != null && quality.score >= .5
      ? new THREE.LineBasicMaterial({ color }) : new THREE.LineDashedMaterial({ color, dashSize: .04, gapSize: .025 })
    const object = new THREE.Line(geometry, material)
    object.computeLineDistances(); object.name = name; object.userData.layer = layer; group.add(object)
  }
  if (sample) {
    const points = new Map(sample.landmarks.map(point => [point.name, point]))
    for (const [edges, layer] of [[bodyEdges, 'Body'], [handEdges, 'Hands'], [footEdges, 'Feet']] as const) {
      const names = new Set(edges.flat())
      for (const name of names) {
        const point = points.get(name)
        if (!point?.xyz_world || sample.missing_mask?.[name] || !layers[layer]) continue
        const sphere = new THREE.Mesh(new THREE.SphereGeometry(layer === 'Hands' ? .012 : .023, 8, 6),
          new THREE.MeshBasicMaterial({ color: highlighted(name) ? 0xffff55 : evidenceColor(point.quality), wireframe: point.quality.state === 'unknown' }))
        sphere.position.copy(worldVector(point.xyz_world)); sphere.name = name
        sphere.userData = { id: `${sample.id}/${name}`, track: trackFor(name), time: sample.global_seconds, layer, quality: point.quality }
        group.add(sphere)
      }
      for (const [a, b] of edges) {
        const first = points.get(a), second = points.get(b)
        if (!first?.xyz_world || !second?.xyz_world || sample.missing_mask?.[a] || sample.missing_mask?.[b]) continue
        const quality = evidenceColor(first.quality) === 0x73d8ef ? second.quality : first.quality
        line([worldVector(first.xyz_world), worldVector(second.xyz_world)], quality, a, layer)
      }
    }
    const head = points.get('head')
    const orientation = sample.segments?.find(segment => segment.segment === 'head' && segment.parent === 'world')
    if (head?.xyz_world && !sample.missing_mask?.head && orientation?.orientation && layers['Head orientation']) {
      const [w, x, y, z] = orientation.orientation.wxyz
      const q = new THREE.Quaternion(x, y, z, w)
      const origin = worldVector(head.xyz_world)
      for (const direction of [new THREE.Vector3(.2, 0, 0), new THREE.Vector3(0, .2, 0), new THREE.Vector3(0, 0, .2)])
        line([origin, origin.clone().add(direction.applyQuaternion(q))], orientation.quality, 'head', 'Head orientation')
    }
  }
  if (layers.Ground && data.calibration?.ground_status === 'resolved' && data.calibration.ground_z != null) {
    const grid = new THREE.GridHelper(6, 12, 0x7e99b0, 0x3a5062)
    grid.rotation.x = Math.PI / 2; grid.position.z = data.calibration.ground_z; grid.userData.layer = 'Ground'; group.add(grid)
  }
  // Separate native segments; nulls and unknown samples break paths.
  function trajectory(name: string, root: boolean, layer: Layer) {
    let previous: { point: THREE.Vector3; sample: Sample; quality: Quality } | null = null
    for (const row of data.samples) {
      const landmark = row.landmarks.find(point => point.name === name)
      const xyz = root ? row.root_xyz_world : landmark?.xyz_world
      const quality = root ? row.quality : landmark?.quality
      if (!xyz || !quality || quality.state === 'unknown' || row.missing_mask?.[name] || (activeAction && (row.global_seconds < activeAction.interval.start || row.global_seconds > activeAction.interval.end))) { previous = null; continue }
      const point = worldVector(xyz)
      if (previous) line([previous.point, point], quality, name, layer)
      previous = { point, sample: row, quality }
    }
  }
  trajectory('pelvis', true, 'Root trajectory')
  if (selection && (selection.kind === 'track' || activeAction || data.samples.some(row => row.landmarks.some(point => point.name === selectedName)))) for (const name of new Set(data.samples.flatMap(row => row.landmarks.map(point => point.name))))
    if (selection.kind === 'entity' && data.samples.some(row => row.landmarks.some(point => point.name === selectedName))
      ? name === selectedName : selection.tracks.includes(trackFor(name))) trajectory(name, false, 'Selected trajectories')
  if (layers.Cameras) for (const camera of (data.calibration?.cameras ?? []).slice(0, 16)) {
    const size = data.cameraSizes?.[camera.camera_id]
    const { center, corners } = calibratedFrustum(camera, size ?? [0, 0])
    if (size) for (const corner of corners) line([center, corner], camera.quality, camera.camera_id, 'Cameras')
    if (size) line([...corners, corners[0]], camera.quality, camera.camera_id, 'Cameras')
    const marker = new THREE.Mesh(new THREE.BoxGeometry(.07, .07, .07), new THREE.MeshBasicMaterial({ color: evidenceColor(camera.quality) }))
    marker.position.copy(center); marker.name = camera.camera_id; marker.userData.layer = 'Cameras'; group.add(marker)
  }
  const axes = new THREE.AxesHelper(.5); group.add(axes)
  return group
}
