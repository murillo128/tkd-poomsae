import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { SyncControls } from './SyncControls'
import { GroundView } from './GroundView'
import { Timeline } from './Timeline'
import { CameraPanel } from './CameraPanel'
import { canStep, initialPlayback, playbackReducer } from './playback'
import { API_ROOT, listProjects, readProject, type ProjectSnapshot } from './projectApi'

import { ThreePanel } from './ThreePanel'
import { trackFor } from './sceneGeometry'
import { GeometryInspector, selectedLandmarkName } from './GeometryInspector'
import type { GeometryData } from './geometryApi'

type LoadState<T> = { status: 'loading' } | { status: 'error'; message: string } | { status: 'ready'; value: T }
const stages = ['ingest', 'sync', 'calibration', 'observations', 'attachment', 'reconstruction', 'ground', 'parsing']

function formatTime(seconds: number): string { return `${seconds.toFixed(3)} s` }

export function App() {
  const [playback, dispatch] = useReducer(playbackReducer, initialPlayback)
  const [catalog, setCatalog] = useState<LoadState<string[]>>({ status: 'loading' })
  const [catalogAttempt, setCatalogAttempt] = useState(0)
  const [projectId, setProjectId] = useState<string | null>(null)
  const [project, setProject] = useState<LoadState<ProjectSnapshot> | null>(null)
  const [projectAttempt, setProjectAttempt] = useState(0)
  const [refreshError, setRefreshError] = useState('')
  const [seekDraft, setSeekDraft] = useState<string | null>(null)
  const [seekError, setSeekError] = useState(false)
  const [geometry, setGeometry] = useState<GeometryData | null>(null)
  const onGeometry = useCallback((data: GeometryData | null) => {
    setGeometry(data)
    dispatch({ type: 'reconstructionSamples', times: data?.samples.map(sample => sample.global_seconds) ?? [] })
  }, [])
  const requestEpoch = useRef(0)
  const loadedProject = useRef<string | null>(null)
  const [windowSeconds, setWindowSeconds] = useState(1)
  const refreshArtifacts = useCallback(() => {
    dispatch({ type: 'play', playing: false })
    setGeometry(null)
    setProjectAttempt(n => n + 1)
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    setCatalog({ status: 'loading' })
    listProjects(controller.signal).then(result => {
      if (!controller.signal.aborted) setCatalog({ status: 'ready', value: result.projects })
    }).catch(error => {
      if (!controller.signal.aborted) setCatalog({ status: 'error', message: String(error) })
    })
    return () => controller.abort()
  }, [catalogAttempt])

  useEffect(() => {
    if (!projectId) { setProject(null); return }
    const controller = new AbortController()
    const epoch = ++requestEpoch.current
    setRefreshError('')
    function failure(message: string) {
      if (loadedProject.current === projectId) setRefreshError(message)
      else setProject({ status: 'error', message })
    }
    if (loadedProject.current !== projectId) setProject({ status: 'loading' })
    readProject(projectId, controller.signal).then(snapshot => {
      if (controller.signal.aborted || epoch !== requestEpoch.current) return
      // The paired metadata reads must describe the same stage revision.
      for (const [name, stage] of Object.entries(snapshot.capabilities.stages)) {
        const detailStage = snapshot.detail.state.stages[name]
        if (detailStage && (detailStage.key !== stage.status.key || detailStage.status !== stage.status.status)) {
          failure('Project changed during metadata read. Retry to load its current revision.')
          return
        }
      }
      setProject({ status: 'ready', value: snapshot })
      if (loadedProject.current !== projectId) dispatch({ type: 'project', id: projectId,
        cameras: snapshot.detail.sources.map(id => ({ id, offsetSeconds: Number.NaN, frames: [] })) })
      loadedProject.current = projectId
    }).catch(error => {
      if (!controller.signal.aborted && epoch === requestEpoch.current) {
        failure(String(error))
      }
    })
    return () => { controller.abort(); requestEpoch.current++ }
  }, [projectId, projectAttempt])

  useEffect(() => {
    if (!playback.playing) return
    let previous = performance.now()
    const timer = window.setInterval(() => {
      const now = performance.now()
      // UI ticks move the desired cursor; only a video renderer can report a delivered frame.
      dispatch({ type: 'tick', elapsedSeconds: (now - previous) / 1000 })
      previous = now
    }, 50)
    return () => window.clearInterval(timer)
  }, [playback.playing])

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (project?.status !== 'ready' || event.altKey || event.ctrlKey || event.metaKey) return
      const target = event.target as HTMLElement | null
      if (target?.closest('input, select, textarea, button, [contenteditable="true"]')) return
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        if (!canStep(playback)) return
        event.preventDefault()
        dispatch({ type: 'step', direction: event.key === 'ArrowLeft' ? -1 : 1 })
      } else if (event.code === 'Space') {
        event.preventDefault()
        dispatch({ type: 'play', playing: !playback.playing })
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [playback, project?.status])

  const snapshot = project?.status === 'ready' ? { ...project.value, revision: `${project.value.revision}/${projectAttempt}` } : null
  const capabilities = snapshot?.capabilities.stages
  const hasProject = Boolean(snapshot)
  const stepAvailable = hasProject && canStep(playback)
  const selectedCamera = playback.cameras.find(camera => camera.id === playback.selectedCameraId)
  const selectionLabel = selectedLandmarkName(playback.selection) ?? playback.selection?.id ?? 'None'
  function openProject(id: string) {
    setProjectId(id)
    loadedProject.current = null
    setGeometry(null)
    setProject(null)
    setSeekDraft(null)
    setSeekError(false)
    dispatch({ type: 'project', id })
  }
  function commitSeek(discardInvalid = false): boolean {
    if (seekDraft === null) return true
    const value = seekDraft.trim()
    const seconds = Number(value)
    if (value && Number.isFinite(seconds) && seconds >= 0) {
      dispatch({ type: 'seek', seconds })
      setSeekDraft(null)
      setSeekError(false)
      return true
    }
    setSeekError(true)
    if (discardInvalid) setSeekDraft(null)
    return false
  }

  return <main className="workspace">
    <header className="masthead">
      <div><p className="eyebrow">Local motion inspection</p><h1>TKD Poomsae</h1></div>
      <span className="service-badge">Local service · {new URL(API_ROOT).host}</span>
    </header>
    <section className="project-bar" aria-label="Project browser">
      <div><h2>Projects</h2><p>Open a registered local project to inspect its current capabilities.</p></div>
      <div className="project-actions">
        {catalog.status === 'loading' && <span role="status">Loading projects…</span>}
        {catalog.status === 'error' && <span role="alert">Service unavailable: {catalog.message}</span>}
        {catalog.status === 'ready' && (catalog.value.length === 0
          ? <span>No registered projects.</span>
          : <label>Project <select value={projectId ?? ''} onChange={event => openProject(event.target.value)}>
              <option value="" disabled>Select project</option>
              {catalog.value.map(id => <option key={id} value={id}>{id}</option>)}
            </select></label>)}
        <button type="button" onClick={() => setCatalogAttempt(value => value + 1)}>Retry projects</button>
      </div>
    </section>
    {projectId && project?.status === 'loading' && <p role="status" className="notice">Loading {projectId} metadata…</p>}
    {projectId && project?.status === 'error' && <div role="alert" className="notice error">
      {project.message} <button type="button" onClick={() => setProjectAttempt(value => value + 1)}>Retry project</button>
    </div>}
    {refreshError && <p role="alert">Project status refresh failed: {refreshError} <button onClick={refreshArtifacts}>Retry status refresh</button></p>}
    {snapshot && <section className="project-summary" aria-label="Project capabilities">
      <div><p className="eyebrow">Current project</p><h2>{snapshot.detail.id}</h2>
        <p>{snapshot.detail.sources.length} camera source{snapshot.detail.sources.length === 1 ? '' : 's'}</p></div>
      <div className="capability-list">{stages.map(name => {
        const stage = capabilities?.[name]
        return <div key={name} className="capability"><strong>{name}</strong><span>{stage?.status.status ?? 'unavailable'}</span>
          <small>{stage?.reason || stage?.status.diagnostics?.join('; ') || (stage?.available ? 'Producer available' : 'No capability reported')}</small>
          {stage && <small>Schema {stage.schema_version} · Software {stage.software_revision} · Model {stage.model_revision ?? 'none'}</small>}
          {stage?.status.key && <small title={stage.status.key}>Artifact {stage.status.key.slice(0, 12)}…</small>}</div>
      })}</div>
      <button type="button" onClick={() => setProjectAttempt(value => value + 1)}>Refresh status</button>
    </section>}
    <section className="transport" aria-label="Shared playback controls">
      <div className="transport-buttons">
        <button type="button" disabled={!stepAvailable} onClick={() => dispatch({ type: 'step', direction: -1 })} aria-label="Previous native frame">◀ Frame</button>
        <button type="button" disabled={!hasProject} onClick={() => dispatch({ type: 'play', playing: !playback.playing })}>{playback.playing ? 'Pause' : 'Play'}</button>
        <button type="button" disabled={!stepAvailable} onClick={() => dispatch({ type: 'step', direction: 1 })} aria-label="Next native frame">Frame ▶</button>
        <label>Speed <select disabled={!hasProject} value={playback.speed} onChange={event => dispatch({ type: 'speed', speed: Number(event.target.value) })}>
          {[0.25, 0.5, 1, 2].map(speed => <option key={speed} value={speed}>{speed}×</option>)}
        </select></label>
      </div>
      <label className="seek">Global time <input type="text" inputMode="decimal" disabled={!hasProject}
        value={seekDraft ?? playback.cursorSeconds.toFixed(3)} aria-invalid={seekError}
        onFocus={() => { setSeekDraft(playback.cursorSeconds.toFixed(3)); setSeekError(false) }}
        onChange={event => { setSeekDraft(event.target.value); setSeekError(false) }}
        onBlur={() => commitSeek(true)}
        onKeyDown={event => {
          if (event.key === 'Enter') event.currentTarget.blur()
          if (event.key === 'Escape') { setSeekDraft(null); setSeekError(false) }
        }} /> seconds</label>
      {seekError && <span role="alert" className="seek-error">Enter a non-negative time in seconds.</span>}
      <div className="time-readout"><span>Desired cursor <strong>{formatTime(playback.cursorSeconds)}</strong></span><span>Delivered video frame <strong>{playback.deliveredFrameSeconds === null ? 'Unavailable' : formatTime(playback.deliveredFrameSeconds)}</strong></span></div>
      <div className="step-options"><label>Step source <select disabled={!hasProject} value={playback.stepMode} onChange={event => dispatch({ type: 'mode', mode: event.target.value as 'camera' | 'reconstruction' })}>
        <option value="camera">Selected camera PTS</option><option value="reconstruction">Reconstruction samples</option>
      </select></label>{playback.stepMode === 'camera' && <label>Camera <select disabled={!hasProject} value={playback.selectedCameraId ?? ''} onChange={event => dispatch({ type: 'camera', id: event.target.value })}>
        {playback.cameras.map(camera => <option key={camera.id} value={camera.id}>{camera.id}</option>)}
      </select></label>}</div>
      {hasProject && !stepAvailable && <p className="hint">Frame stepping unavailable: {playback.stepMode === 'camera' ? `${selectedCamera?.id ?? 'selected camera'} PTS` : 'reconstruction sample grid'} is not exposed by the local service. Keyboard: Space plays or pauses; ← and → step when samples are available.</p>}
    </section>
    {snapshot && <section className="panel" aria-label="Development controls"><h2>Development controls</h2>
      <label>Local trajectory radius (seconds) <select value={windowSeconds} onChange={e => setWindowSeconds(Number(e.target.value))}>
        {[0.5, 1, 2, 5, 10].map(value => <option key={value} value={value}>{value}</option>)}
      </select></label><p>3D and dynamic ground trajectories use the same bounded window around the global cursor. Each panel exposes its analytical layers.</p>
      {snapshot.detail.sources.map(camera => <SyncControls key={`${snapshot.detail.id}/${camera}`} project={snapshot.detail.id} camera={camera} revision={snapshot.revision} onChanged={refreshArtifacts} />)}
    </section>}
    <div className="inspection-grid" aria-label="Inspection layout">
      <section className="panel cameras"><h2>Camera views</h2><div className="camera-grid">{snapshot?.detail.sources.map(id => <CameraPanel key={`${snapshot.detail.id}/${snapshot.revision}/${id}`} project={snapshot.detail.id} camera={id}
        seconds={playback.cursorSeconds} selected={playback.selectedCameraId === id} playing={playback.playing} speed={playback.speed}
        onClock={camera => dispatch({ type: 'cameraClock', camera })}
        onDelivered={(camera, seconds) => { if (camera === playback.selectedCameraId) dispatch({ type: 'frameDelivered', seconds }) }}
        onSelect={(id, description) => dispatch({ type: 'select', selection: { kind: 'entity', id, tracks: [trackFor(id.split('/').at(-1)!)], description } })} />) ?? <p>Open a project to view cameras.</p>}</div></section>
      <ThreePanel project={snapshot} playback={playback} dispatch={dispatch} onData={onGeometry} windowSeconds={windowSeconds} />
      <GroundView windowSeconds={windowSeconds} projectId={snapshot?.detail.id ?? null} revision={snapshot?.revision} seconds={playback.cursorSeconds} playing={playback.playing} selection={playback.selection} dispatch={dispatch} />
      <Timeline onChanged={refreshArtifacts} key={projectId ?? 'no-project'} projectId={snapshot?.detail.id ?? null} revision={snapshot?.revision} seconds={playback.cursorSeconds} selection={playback.selection} dispatch={dispatch} />
      <section className="panel inspector"><h2>Inspector</h2><p>Selection: <strong>{selectionLabel}</strong></p><p>Participating tracks: {playback.selection?.tracks.join(', ') || 'None'}</p><p>{playback.selection?.description ?? ''}</p><GeometryInspector project={snapshot?.detail.id ?? null} data={geometry} selection={playback.selection} cursorSeconds={playback.cursorSeconds} revision={snapshot?.revision} onEntity={id => {
        dispatch({ type: 'play', playing: false }); dispatch({ type: 'select', selection: { kind: 'entity', id, tracks: playback.selection?.tracks ?? [] } })
      }} onCamera={camera => {
        dispatch({ type: 'play', playing: false }); dispatch({ type: 'camera', id: camera })
      }} /></section>
    </div>
  </main>
}
