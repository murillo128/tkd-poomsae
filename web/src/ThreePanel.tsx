import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { ProjectSnapshot } from './projectApi'
import type { PlaybackAction, PlaybackState } from './playback'
import { readGeometry, type GeometryData } from './geometryApi'
import { bodyEdges, handEdges, footEdges, buildScene, defaultLayers, disposeGroup, layerNames, sampleAt } from './sceneGeometry'

export function ThreePanel({ project, playback, dispatch, onData, windowSeconds = 1 }: {
  windowSeconds?: number;
  project: ProjectSnapshot | null; playback: PlaybackState; dispatch: (action: PlaybackAction) => void
  onData: (data: GeometryData | null) => void
}) {
  const [data, setData] = useState<GeometryData | null>(null)
  const [message, setMessage] = useState('Open a project to inspect 3D geometry.')
  const [rendererError, setRendererError] = useState<string | null>(null)
  const [layers, setLayers] = useState({ ...defaultLayers })
  const [retry, setRetry] = useState(0)
  const host = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<{ scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: OrbitControls } | null>(null)
  const groupRef = useRef<THREE.Group | null>(null)
  const selectionRef = useRef({ dispatch, playback })
  selectionRef.current = { dispatch, playback }
  const epoch = useRef(0)
  const [rendererReady, setRendererReady] = useState(0)

  const windowSecond = playback.playing ? Math.floor(playback.cursorSeconds) : playback.cursorSeconds
  useEffect(() => {
    setData(null); onData(null)
    if (!project) { setMessage('Open a project to inspect 3D geometry.'); return }
    const controller = new AbortController(), request = ++epoch.current
    setMessage('Loading bounded native geometry…')
    // Debounce seeks/playback; all requests share one abortable revision-bound batch.
    const timer = window.setTimeout(() => {
      readGeometry(project.detail.id, windowSecond, controller.signal, windowSeconds + (playback.playing ? 1 : 0)).then(result => {
        if (controller.signal.aborted || epoch.current !== request) return
        setData(result); onData(result); setMessage(result.reasons.join('; '))
      }).catch(error => {
        if (!controller.signal.aborted && epoch.current === request) setMessage(`3D unavailable: ${String(error)}`)
      })
    }, 80)
    return () => { window.clearTimeout(timer); controller.abort(); epoch.current++ }
  }, [project?.detail.id, project?.revision, windowSecond, windowSeconds, playback.playing, retry, onData])

  useEffect(() => {
    const element = host.current
    setRendererError(null)
    if (!element || !project) return
    if (typeof WebGL2RenderingContext === 'undefined') { setRendererError('3D rendering unavailable: WebGL2 is not supported.'); return }
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true }) }
    catch { setRendererError('3D rendering unavailable: WebGL could not initialize.'); return }
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0x101d29)
    const camera = new THREE.PerspectiveCamera(45, 1, .01, 1000)
    camera.up.set(0, 0, 1); camera.position.set(3, -4, 2.8)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 0, .9); controls.update()
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); element.append(renderer.domElement)
    renderer.domElement.setAttribute('aria-label', '3D world viewport')
    sceneRef.current = { scene, camera, renderer, controls }
    const render = () => renderer.render(scene, camera)
    const resize = () => {
      const width = element.clientWidth, height = element.clientHeight
      camera.aspect = width / height; camera.updateProjectionMatrix(); renderer.setSize(width, height); render()
    }
    const observer = new ResizeObserver(resize); observer.observe(element); resize()
    controls.addEventListener('change', render)
    let down = { x: 0, y: 0 }
    const pointerDown = (event: PointerEvent) => { down = { x: event.clientX, y: event.clientY } }
    const pick = (event: PointerEvent) => {
      if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > 4) return
      const rect = renderer.domElement.getBoundingClientRect()
      const ray = new THREE.Raycaster(); ray.setFromCamera(new THREE.Vector2(
        (event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1), camera)
      const hit = ray.intersectObjects(scene.children, true).find(item => item.object.userData.id)
      if (!hit) return
      const { id, track, time } = hit.object.userData
      selectionRef.current.dispatch({ type: 'select', selection: { kind: 'entity', id, tracks: [track] } })
      selectionRef.current.dispatch({ type: 'seek', seconds: time })
    }
    renderer.domElement.addEventListener('pointerdown', pointerDown)
    renderer.domElement.addEventListener('pointerup', pick)
    setRendererReady(value => value + 1)
    return () => {
      observer.disconnect(); controls.removeEventListener('change', render); controls.dispose()
      renderer.domElement.removeEventListener('pointerdown', pointerDown); renderer.domElement.removeEventListener('pointerup', pick)
      if (groupRef.current) { disposeGroup(groupRef.current); groupRef.current = null }
      renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove(); scene.clear(); sceneRef.current = null
    }
  }, [project?.detail.id])

  useEffect(() => {
    const view = sceneRef.current
    if (!view) return
    if (groupRef.current) { view.scene.remove(groupRef.current); disposeGroup(groupRef.current); groupRef.current = null }
    if (data) {
      groupRef.current = buildScene({ ...data, samples: data.samples.filter(row => Math.abs(row.global_seconds - playback.cursorSeconds) <= windowSeconds) }, playback.cursorSeconds, playback.selection, layers)
      view.scene.add(groupRef.current)
    }
    view.renderer.render(view.scene, view.camera)
  }, [data, playback.cursorSeconds, playback.selection, layers, rendererReady, windowSeconds])

  const sample = data && sampleAt(data.samples, playback.cursorSeconds)
  const expected = new Set([...bodyEdges.flat(), ...handEdges.flat(), ...footEdges.flat()])
  const missing = sample ? [...expected].filter(name => !sample.landmarks.find(point => point.name === name)?.xyz_world || sample.missing_mask?.[name]) : []
  function pickLandmark(name: string) {
    const object = groupRef.current?.children.find(child => child.name === name && child.userData.id)
    if (object) dispatch({ type: 'select', selection: { kind: 'entity', id: object.userData.id, tracks: [object.userData.track] } })
  }
  return <section className="panel three-d"><h2>3D reconstruction</h2>
    <div className="layer-controls" aria-label="3D layers">{layerNames.map(layer => <label key={layer}>
      <input type="checkbox" checked={layers[layer]} onChange={event => setLayers(previous => ({ ...previous, [layer]: event.target.checked }))} />{layer}
    </label>)}</div>
    <div className="three-viewport" ref={host} />
    <p>XY ground · Z up. X red, Y green, Z blue. Units: {data?.unit ?? 'unknown'}.</p>
    <p>Pinhole camera frusta use registered image bounds; lens distortion is not shown.</p>
    <p>Drag to orbit · right drag to pan · scroll to zoom. <button onClick={() => { sceneRef.current?.controls.reset() }}>Reset view</button></p>
    <p className="evidence-legend">Cyan observed · red low/unspecified score · purple interpolated · orange inferred · gray wire unknown. Dashed edges carry uncertain evidence.</p>
    {rendererError && <p role="status">{rendererError}</p>}
    {message && <p role="status">{message} <button onClick={() => setRetry(value => value + 1)}>Retry 3D</button></p>}
    {data && <>
      <p>Cursor {playback.cursorSeconds.toFixed(3)} s · {sample ? `native sample ${sample.global_seconds.toFixed(3)} s` : 'No native sample at this cursor; pose unavailable.'}</p>
      {data.truncated && <p role="status">Display truncated at the geometry page or 16-camera limit; paths show only returned native samples.</p>}
      {data.calibration?.ground_status !== 'resolved' && <p>Ground plane unavailable: calibrated ground unresolved.</p>}
      <label>Native sample <select aria-label="3D native sample" value={sample?.id ?? ''} onChange={event => {
        const row = data.samples.find(value => value.id === event.target.value)
        if (row) dispatch({ type: 'seek', seconds: row.global_seconds })
      }}><option value="" disabled>Select native instant</option>{data.samples.map(row => <option key={row.id} value={row.id}>{row.global_seconds.toFixed(6)} s</option>)}</select></label>
      {sample && <label>Pick landmark <select aria-label="Pick 3D landmark" value="" onChange={event => pickLandmark(event.target.value)}>
        <option value="">Select landmark</option>{sample.landmarks.filter(point => point.xyz_world && !sample.missing_mask?.[point.name]).map(point => <option key={point.name} value={point.name}>{point.name}</option>)}</select></label>}
      <details><summary>Missing geometry ({missing.length})</summary>{missing.map(name => { const point = sample?.landmarks.find(value => value.name === name); return <p key={name}>{name}: unavailable coordinates · evidence {point?.quality.state ?? 'unknown'}; source IDs {point?.quality.source_ids?.join(', ') || 'unavailable'}</p> })}
        {!sample && <p>No pose invented between native instants.</p>}
        {sample && !sample.root_xyz_world && <p>Root trajectory unavailable at this instant: no root coordinates.</p>}
        {sample && sample.quality.state === 'unknown' && <p>Root path gaps: native sample evidence is unknown.</p>}
        {sample && !sample.segments?.some(segment => segment.segment === 'head' && segment.parent === 'world' && segment.orientation) && <p>Head orientation unavailable: no world-parent frame.</p>}
      </details>
      {data.actions.length > 0 && <div className="scene-actions" aria-label="3D actions">{data.actions.map(action => <button key={action.id} aria-pressed={playback.selection?.id === action.id} onClick={() => {
        dispatch({ type: 'select', selection: { kind: 'entity', id: action.id, tracks: action.tracks } }); dispatch({ type: 'seek', seconds: action.interval.start })
      }}>{action.id}</button>)}</div>}
    </>}
  </section>
}
