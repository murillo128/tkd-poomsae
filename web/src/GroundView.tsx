import { useEffect, useState } from 'react'
import type { Dispatch } from 'react'
import { useGroundSnapshot } from './useGroundSnapshot'
import { readGroundSnapshot, readGroundSummary } from './groundApi'
import type { GroundFrame, GroundSummary, SnapshotResponse, Geometry, XY } from './groundApi'
import type { PlaybackAction, Selection } from './playback'
import type { Track } from '../../contracts/types'

const layerNames = ['Footprints', 'Foot axes', 'Root path', 'Contacts', 'Pivots', 'Measurements'] as const
type Layer = typeof layerNames[number]
const footTrack = (foot: string): Track => foot === 'left' ? 'left_leg' : 'right_leg'
export function GroundView({ projectId, revision, seconds, playing, selection, dispatch, windowSeconds = 1 }: {
  windowSeconds?: number
  projectId: string | null; revision: string | undefined; seconds: number; playing: boolean; selection: Selection | null; dispatch: Dispatch<PlaybackAction>
}) {
  const [mode, setMode] = useState<'dynamic' | 'summary'>('dynamic')
  const [layers, setLayers] = useState<Set<Layer>>(new Set(layerNames))
  const [data, setData] = useState<{ id: string; revision?: string; response: SnapshotResponse; summary: GroundSummary } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    setData(null); setError(null)
    if (!projectId) return
    const controller = new AbortController()
    async function load() {
      const response = await readGroundSnapshot(projectId!, 0, controller.signal)
      if (!response.ground_view) throw new Error(response.reason || 'Ground geometry unavailable')
      const summary = await readGroundSummary(projectId!, response.ground_view, response.revision, controller.signal)
      if (!controller.signal.aborted) setData({ id: projectId!, revision, response, summary })
    }
    load().catch(reason => { if (!controller.signal.aborted) setError(String(reason)) })
    return () => controller.abort()
  }, [projectId, revision, attempt])
  const active = data?.id === projectId && data.revision === revision ? data : null
  const { current, error: snapshotError } = useGroundSnapshot(active?.id ?? null, active?.response.revision ?? null, seconds)
  const displayed = current && (playing || current.seconds === seconds) ? current : null
  function choose(id: string, time: number, tracks: Track[]) {
    dispatch({ type: 'play', playing: false })
    dispatch({ type: 'seek', seconds: time })
    dispatch({ type: 'select', selection: { kind: 'entity', id, tracks } })
  }
  function highlighted(id: string, tracks: Track[]) {
    return selection?.id === id || Boolean(selection && (selection.kind === 'track' || !groundIds.has(selection.id)) && selection.tracks.some(track => tracks.includes(track)))
  }
  const meta = active?.response.ground_view
  const bounds = meta?.scene_bounds
  const summary = active?.summary
  const groundIds = new Set([...(summary?.placements.map(e => e.footprint.id) ?? []), ...(summary?.rotations.map(e => e.pivot.id) ?? []), ...(summary?.frames.flatMap(f => [f.id, f.contact_event?.id ?? '']) ?? [])])
  const snapshot = displayed?.response.snapshot ?? null
  const near = (start: number, end: number) => mode === 'summary' || (start <= seconds + windowSeconds && end >= seconds - windowSeconds)
  const visible = (layer: Layer) => layers.has(layer)
  // A fixed viewBox uses authoritative execution-wide bounds with equal XY scale.
  const extent = bounds ? Math.max(bounds.maximum_xy[0] - bounds.minimum_xy[0], bounds.maximum_xy[1] - bounds.minimum_xy[1], 0.001) : 1
  const pad = extent * .12
  const point = (xy: XY): XY => [xy[0], -xy[1]]
  const points = (xy: XY[]) => xy.map(p => point(p).join(',')).join(' ')
  const viewBox = bounds ? `${bounds.minimum_xy[0] - pad} ${-bounds.maximum_xy[1] - pad} ${bounds.maximum_xy[0] - bounds.minimum_xy[0] + pad * 2} ${bounds.maximum_xy[1] - bounds.minimum_xy[1] + pad * 2}` : '0 0 1 1'
  function geometry(value: Geometry) {
    return <>
      {visible('Footprints') && value.polygon && <polygon points={points(value.polygon)} className="foot-shape" />}
      {visible('Footprints') && value.supported_points.map((xy, i) => <circle key={i} cx={point(xy)[0]} cy={point(xy)[1]} r={extent * .008} />)}
      {visible('Foot axes') && value.axis_start && value.axis_end && <line x1={point(value.axis_start)[0]} y1={point(value.axis_start)[1]} x2={point(value.axis_end)[0]} y2={point(value.axis_end)[1]} markerEnd="url(#ground-axis-arrow)" />}
    </>
  }
  function selectable(id: string, time: number, tracks: Track[], label: string) {
    return { role: 'button', tabIndex: 0, 'aria-label': label, 'aria-pressed': highlighted(id, tracks), onClick: () => choose(id, time, tracks), onKeyDown: (event: React.KeyboardEvent) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(id, time, tracks) } } }
  }
  const rootRuns = new Map<number, GroundFrame[]>()
  summary?.frames.filter(frame => frame.path_run !== null && frame.root?.xy_ground && near(frame.sampled_seconds, frame.sampled_seconds)).forEach(frame => {
    const run = rootRuns.get(frame.path_run!) ?? []; run.push(frame); rootRuns.set(frame.path_run!, run)
  })
  return <section className="panel ground" aria-label="Ground view"><h2>Ground view</h2>
    <div className="ground-controls"><label>View <select value={mode} onChange={event => setMode(event.target.value as typeof mode)}><option value="dynamic">Dynamic</option><option value="summary">Summary</option></select></label>
      {layerNames.map(layer => <label key={layer}><input type="checkbox" checked={visible(layer)} onChange={() => setLayers(previous => { const next = new Set(previous); if (next.has(layer)) next.delete(layer); else next.add(layer); return next })} />{layer}</label>)}
    </div>
    {!projectId ? <p>Open a project to inspect ground geometry.</p> : error || snapshotError ? <p role="alert">Ground view unavailable: {error || snapshotError} <button onClick={() => setAttempt(n => n + 1)}>Retry ground view</button></p> : !active ? <p role="status">Loading ground geometry…</p> : !bounds ? <p>Ground view unavailable: no projected geometry.</p> : <>
      <p>{meta?.participant_id} · {meta?.world_unit === 'm' ? 'Metric ground coordinates (m)' : 'Metric scale unresolved · arbitrary world units'} · +X right, +Y up</p>
      <p className="ground-legend">L = left · R = right · circles: measured landmarks · dashed shape: approximate sole · dashed paths: pivot trajectories</p>
      {mode === 'dynamic' && <p role="status">{!displayed ? 'Loading current native snapshot…' : displayed.response.available ? `${snapshot?.status === 'native_snapshot' ? 'Preceding native snapshot' : 'Native sample'} at ${snapshot?.sampled_seconds.toFixed(3)} s · requested ${displayed.seconds.toFixed(3)} s · cursor ${seconds.toFixed(3)} s` : `Current geometry unavailable: ${displayed.response.reason}`}</p>}
      <svg className="ground-scene" aria-label="Top-down ground XY scene" viewBox={viewBox} preserveAspectRatio="xMidYMid meet">
        <defs><marker id="ground-axis-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" /></marker></defs>
        {visible('Root path') && [...rootRuns].map(([run, frames]) => <polyline key={run} points={points(frames.map(f => f.root!.xy_ground!))} className="root-path" />)}
        {visible('Root path') && [...rootRuns.values()].flat().map(frame => <circle key={frame.id} {...selectable(frame.root!.id, frame.sampled_seconds, ['body_root'], `Root path at ${frame.sampled_seconds} seconds`)} className="root-point" cx={point(frame.root!.xy_ground!)[0]} cy={point(frame.root!.xy_ground!)[1]} r={extent * .008} />)}
        {(visible('Footprints') || visible('Foot axes')) && summary?.placements.filter(e => near(e.footprint.interval.start, e.footprint.interval.end)).map(event => <g key={event.footprint.id} {...selectable(event.footprint.id, event.event_seconds, [footTrack(event.footprint.foot)], `${event.footprint.foot} ${event.kind} placement at ${event.event_seconds} seconds`)} className={`ground-foot ${event.footprint.foot} ${event.geometry.approximated ? 'approximate' : ''}`}>
          {geometry(event.geometry)}{event.footprint.xy_ground && <text x={point(event.footprint.xy_ground)[0]} y={point(event.footprint.xy_ground)[1]} fontSize={extent * .028}>{event.footprint.foot === 'left' ? 'L' : 'R'}</text>}
        </g>)}
        {mode === 'dynamic' && snapshot?.feet.map(foot => <g key={foot.foot} aria-pressed={highlighted(foot.event_id ?? foot.foot, [footTrack(foot.foot)])} className={`ground-foot current-foot ${foot.foot} ${foot.geometry.approximated ? 'approximate' : ''}`} aria-label={`Current ${foot.foot} foot: ${foot.status}`}>{geometry(foot.geometry)}</g>)}
        {mode === 'dynamic' && snapshot?.root?.xy_ground && <circle className="current-root" cx={point(snapshot.root.xy_ground)[0]} cy={point(snapshot.root.xy_ground)[1]} r={extent * .015} />}
        {visible('Pivots') && summary?.rotations.filter(e => near(e.pivot.interval.start, e.pivot.interval.end)).map(event => <g key={event.pivot.id} {...selectable(event.pivot.id, event.pivot.interval.start, [footTrack(event.pivot.foot)], `${event.pivot.foot} pivot at ${event.pivot.interval.start} seconds`)} className={`pivot-path ${event.pivot.foot}`}>
          {event.translations.map(path => <polyline key={path.landmark} points={points(path.positions)}><title>{path.landmark} · {event.classification} · region {event.pivot.region}</title></polyline>)}
        </g>)}
      </svg>
      {visible('Contacts') && <div className="ground-events" aria-label="Contact and support events">{summary?.frames.filter(f => near(f.sampled_seconds, f.sampled_seconds) && f.contact_event).map(frame => <button key={frame.id} aria-pressed={highlighted(frame.contact_event!.id, ['left_leg', 'right_leg'])} onClick={() => choose(frame.contact_event!.id, frame.sampled_seconds, ['left_leg', 'right_leg'])}>{frame.sampled_seconds.toFixed(3)} s · support {frame.contact_event!.sample.support} · L {frame.contact_event!.sample.left.state} / R {frame.contact_event!.sample.right.state}</button>)}</div>}
      {visible('Measurements') && <div className="ground-measurements">{summary?.relations.filter(r => mode === 'summary' || summary.placements.some(p => [r.first_id, r.second_id].includes(p.footprint.id) && near(p.footprint.interval.start, p.footprint.interval.end))).map(relation => <p key={`${relation.first_id}/${relation.second_id}`}>{relation.measurements.map(m => `${m.name}: ${m.value === null || m.quality.state === 'unknown' || (meta?.world_unit !== 'm' && m.unit === 'm') ? 'unknown' : `${m.value.toPrecision(3)} ${m.unit}`} (${m.quality.state})`).join(' · ')}</p>)}</div>}
      {summary?.placements.filter(e => near(e.footprint.interval.start, e.footprint.interval.end) && e.geometry.kind === 'unavailable').map(e => <p key={e.footprint.id}>{e.footprint.foot} {e.kind} placement geometry unavailable at {e.event_seconds.toFixed(3)} s</p>)}
      {summary?.placements.filter(e => e.footprint.id === selection?.id).map(e => <p key={e.footprint.id}>Placement confidence: {e.footprint.quality.state} · uncertainty {e.footprint.quality.uncertainty ?? 'unknown'} {meta?.world_unit === 'm' ? 'm' : 'arbitrary units'}</p>)}
      {summary?.rotations.filter(e => e.pivot.id === selection?.id).map(e => <p key={e.pivot.id}>Pivot region: {e.pivot.region} · {e.classification} · confidence {e.pivot.quality.state} · rotation {e.pivot.rotation_rad ?? 'unknown'} rad</p>)}
      {mode === 'dynamic' && snapshot?.root?.quality.state === 'unknown' && <p>Root geometry unknown.</p>}
      <p>Selected entity: {selection?.id ?? 'None'}. Native evidence retains unknown contact and missing geometry; sole boundaries are approximate when supplied.</p>
      {mode === 'dynamic' && snapshot?.feet.map(f => <p key={`${f.foot}-uncertainty`}>{f.foot} foot: {f.status} · contact {f.contact.state} ({f.contact.quality.state}) · position uncertainty {f.position_uncertainty ?? 'unknown'} {meta?.world_unit === 'm' ? 'm' : 'arbitrary units'} · angle uncertainty {f.angle_uncertainty ?? 'unknown'} rad</p>)}
      {mode === 'dynamic' && snapshot?.feet.filter(f => f.geometry.kind === 'unavailable' || f.status === 'unavailable').map(f => <p key={f.foot}>{f.foot} geometry unknown: {f.reasons.join(', ')}</p>)}
    </>}
  </section>
}
