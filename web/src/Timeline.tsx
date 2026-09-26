import { useEffect, useRef, useState, type Dispatch } from 'react'
import type { Interval, Track } from '../../contracts/types'
import type { PlaybackAction, Selection } from './playback'
import { editTimeline, readTimeline, type EditBase, type EditOperation, type TimelineData } from './timelineApi'
export const trackNames: Track[] = ['left_arm', 'right_arm', 'left_leg', 'right_leg', 'body_root', 'head']
type Entity = { id: string; kind: 'step' | 'action' | 'phase' | 'keyframe' | 'stance'; time: number; interval?: Interval; tracks: Track[]; label: string; detail: unknown }
type Draft = { entity: Entity; start: string; end: string }
function entities(data: TimelineData): Entity[] {
  const tracks = (id: string) => data.actions.find(a => a.id === id)?.tracks ?? []
  return [
    ...data.steps.map(s => ({ id: s.id, kind: 'step' as const, time: s.interval.start, interval: s.interval, tracks: [...new Set(data.actions.filter(a => s.action_ids.includes(a.id)).flatMap(a => a.tracks))], label: `Step ${s.id}`, detail: s })),
    ...data.actions.map(a => ({ id: a.id, kind: 'action' as const, time: a.interval.start, interval: a.interval, tracks: a.tracks, label: `${a.category} ${a.id} · ${a.quality?.state ?? 'unknown'}`, detail: a })),
    ...data.phases.map(p => ({ id: p.id, kind: 'phase' as const, time: p.interval.start, interval: p.interval, tracks: tracks(p.action_id), label: `Phase ${p.name} ${p.id}`, detail: p })),
    ...data.keyframes.map(k => ({ id: k.id, kind: 'keyframe' as const, time: k.global_seconds, tracks: k.track ? [k.track] : tracks(k.action_id), label: `Keyframe ${k.event} ${k.id}`, detail: k })),
    ...data.stances.map(s => ({ id: s.id, kind: 'stance' as const, time: s.interval.start, interval: s.interval, tracks: ['left_leg', 'right_leg'] as Track[], label: `Stance ${s.label} ${s.id} · ${s.quality.state}`, detail: s })),
  ]
}
export function Timeline({ projectId, revision, seconds, selection, dispatch }: { projectId: string | null; revision?: string; seconds: number; selection: Selection | null; dispatch: Dispatch<PlaybackAction> }) {
  const [data, setData] = useState<TimelineData | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState(0)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [queue, setQueue] = useState<EditOperation[]>([])
  const [base, setBase] = useState<EditBase | null>(null)
  const [author, setAuthor] = useState('')
  const [reason, setReason] = useState('')
  const [busy, setBusy] = useState(false)
  const write = useRef<AbortController | null>(null)
  useEffect(() => () => write.current?.abort(), [])
  useEffect(() => {
    const controller = new AbortController()
    setData(null); setError('')
    if (!projectId) return
    setLoading(true)
    readTimeline(projectId, controller.signal).then(value => {
      if (!controller.signal.aborted) { setData(value); setLoading(false) }
    }).catch(e => { if (!controller.signal.aborted) { setError(String(e)); setLoading(false) } })
    return () => controller.abort()
  }, [projectId, revision, attempt])
  const all = data ? entities(data) : []
  const selected = all.find(e => e.id === selection?.id)
  const bounds = data?.bounds ?? { start: 0, end: Math.max(30, seconds) }
  const duration = Math.max(0.001, bounds.end - bounds.start)
  const span = duration / zoom
  const start = bounds.start + pan * (duration - span)
  const end = start + span
  const position = (time: number) => (time - start) / span * 100
  function select(entity: Entity) {
    dispatch({ type: 'play', playing: false })
    dispatch({ type: 'seek', seconds: entity.time })
    dispatch({ type: 'select', selection: { kind: 'entity', id: entity.id, tracks: entity.tracks } })
    if (entity.time < start || entity.time > end) setPan(duration === span ? 0 : Math.max(0, Math.min(1, (entity.time - bounds.start - span / 2) / (duration - span))))
  }
  function navigate(kind: Entity['kind'], direction: -1 | 1) {
    const sorted = all.filter(e => e.kind === kind).sort((a, b) => a.time - b.time || a.id.localeCompare(b.id))
    const index = sorted.findIndex(e => e.id === selection?.id)
    const target = index >= 0 ? sorted[index + direction] : direction > 0 ? sorted.find(e => e.time > seconds) : [...sorted].reverse().find(e => e.time < seconds)
    if (target) select(target)
  }
  function beginDraft() {
    if (!selected || selected.kind === 'stance' || !data?.semantic) return
    setBase(previous => previous ?? { expected_revision: data.semantic!.effective_edit_revision, automatic_revision: data.semantic!.artifact_revision })
    setDraft({ entity: selected, start: String(selected.time), end: String(selected.interval?.end ?? '') })
  }
  function stageDraft() {
    if (!draft) return
    const a = Number(draft.start), b = Number(draft.end)
    if (!draft.start.trim() || !Number.isFinite(a) || a < 0 || (draft.entity.interval && (!draft.end.trim() || !Number.isFinite(b) || b <= a))) {
      setError('Enter finite non-negative times; interval end must follow start. Draft retained.'); return
    }
    if (queue.length >= 64) { setError('Apply this batch before adding more edits (maximum 64).'); return }
    const operation: EditOperation = draft.entity.interval ? { kind: 'boundary', target_id: draft.entity.id, interval: { start: a, end: b } } : { kind: 'keyframe_move', target_id: draft.entity.id, global_seconds: a }
    setQueue(previous => [...previous.filter(o => o.target_id !== operation.target_id), operation]); setDraft(null); setError('')
  }
  async function command(command: 'apply' | 'undo' | 'reset') {
    if (!projectId || !data?.semantic || busy || !author.trim() || !reason.trim()) return
    const controller = new AbortController(); write.current = controller; setBusy(true); setError('')
    try {
      await editTimeline(projectId, command === 'apply' && base ? base : { expected_revision: data.semantic.effective_edit_revision, automatic_revision: data.semantic.artifact_revision }, command, command === 'apply' ? queue : [], author, reason, controller.signal)
      if (controller.signal.aborted) return
      setQueue([]); setBase(null); setDraft(null); setAttempt(value => value + 1)
    } catch (e) { if (!controller.signal.aborted) setError(String(e)) }
    finally { if (!controller.signal.aborted) setBusy(false) }
  }
  function laneRows(items: Entity[]) {
    // Pack disjoint entities of the same kind; overlapping boundaries keep
    // independent rows. Off-window entities must not leave empty lanes.
    const rows: Entity[][] = []
    const visible = items.filter(e => (e.interval?.end ?? e.time) >= start && e.time <= end)
      .sort((a, b) => a.kind.localeCompare(b.kind) || a.time - b.time || a.id.localeCompare(b.id))
    for (const entity of visible) {
      const row = rows.find(row => row[0].kind === entity.kind &&
        (row.at(-1)!.interval?.end ?? row.at(-1)!.time + span * .04) < entity.time)
      if (row) row.push(entity)
      else rows.push([entity])
    }
    return rows.map(row => <div key={row[0].id} className="timeline-row">{row.map(entityButton)}</div>)
  }
  function entityButton(entity: Entity) {
    const visibleStart = Math.max(start, entity.time), visibleEnd = Math.min(end, entity.interval?.end ?? entity.time)
    if (visibleEnd < start || entity.time > end) return null
    return <button type="button" key={entity.id} className={`timeline-entity ${entity.kind}`} aria-label={entity.label} aria-pressed={selection?.id === entity.id}
      data-entity-id={entity.id} data-start={entity.time} data-end={entity.interval?.end ?? entity.time}
      style={{ marginLeft: `${Math.max(0, position(visibleStart))}%`, width: entity.interval ? `${Math.max(.5, position(visibleEnd) - position(visibleStart))}%` : undefined }}
      onClick={() => select(entity)} title={`${entity.label}: ${entity.time}–${entity.interval?.end ?? entity.time} s`}>{entity.kind === 'keyframe' ? '◆' : entity.label}</button>
  }
  return <section className="panel timeline" aria-label="Semantic timeline" tabIndex={0} onKeyDown={event => {
    if ((event.target as HTMLElement).closest('input, textarea, select')) return
    const kind = event.key.toLowerCase() === 's' ? 'step' : event.key.toLowerCase() === 'a' ? 'action' : event.key.toLowerCase() === 'k' ? 'keyframe' : null
    if (kind && !event.ctrlKey && !event.metaKey && !event.altKey) { event.preventDefault(); navigate(kind, event.shiftKey ? -1 : 1) }
  }}>
    <h2>Timeline</h2><p>Global cursor: <strong>{seconds.toFixed(3)} s</strong></p>
    {loading && <p role="status">Loading timeline…</p>}
    {error && <p role="alert">{error}</p>}
    {projectId && <button type="button" disabled={busy} onClick={() => setAttempt(v => v + 1)}>Reload timeline</button>}
    <p>{data?.semantic?.available ? `${data.semantic.origin === 'manual' ? 'Manual edits over preserved automatic artifact' : 'Original automatic output'} · edit revision ${data.semantic.effective_edit_revision} · automatic artifact ${data.semantic.artifact_revision}` : 'Parsing unavailable — raw tracks and global time remain usable.'}</p>
    <p>Dense motion: {data?.motion?.available ? `available (${data.motion.arrays.length} array descriptors)` : 'unavailable'}. Event keyframes do not replace dense motion between events.</p>
    <div className="timeline-controls">
      <label>Zoom <input type="range" min="1" max="20" step=".5" value={zoom} onChange={e => setZoom(Number(e.target.value))} /></label>
      <label>Pan <input type="range" min="0" max="1" step=".01" value={pan} onChange={e => setPan(Number(e.target.value))} /></label>
      {(['step', 'action', 'keyframe'] as const).map(kind => <span key={kind}><button type="button" disabled={!all.some(e => e.kind === kind)} onClick={() => navigate(kind, -1)}>Previous {kind}</button> <button type="button" disabled={!all.some(e => e.kind === kind)} onClick={() => navigate(kind, 1)}>Next {kind}</button></span>)}
    </div>
    <p>Keyboard within timeline: S steps, A actions, K keyframes; Shift goes backward. Times are global seconds.</p>
    <label className="timeline-scrub">Timeline time <input type="range" min={start} max={end} step=".001" value={Math.max(start, Math.min(end, seconds))} disabled={!projectId} onChange={e => { dispatch({ type: 'play', playing: false }); dispatch({ type: 'seek', seconds: Number(e.target.value) }) }} /></label>
    <div className="timeline-ruler"><span>{start.toFixed(3)} s</span><span>{end.toFixed(3)} s</span></div>
    <div className="timeline-lanes">
      <div className="timeline-lane" aria-label="SequenceSteps">{laneRows(all.filter(e => e.kind === 'step'))}</div>
      {trackNames.map(track => <div className="timeline-track" key={track} data-track={track} data-highlighted={selection?.tracks.includes(track) ?? false}>
        <button type="button" aria-label={`Track lane ${track.replaceAll('_', ' ')}`} disabled={!projectId} aria-pressed={selection?.tracks.includes(track) ?? false} onClick={() => dispatch({ type: 'select', selection: { kind: 'track', id: track, tracks: [track] } })}>{track.replaceAll('_', ' ')}</button>
        <div className="timeline-lane">{laneRows(all.filter(e => e.kind !== 'step' && e.tracks.includes(track)))}</div>
      </div>)}
      {seconds >= start && seconds <= end && <div className="timeline-cursor" style={{ left: `${position(seconds)}%` }} aria-hidden="true" />}
    </div>
    <div className="track-list" aria-label="Shared track selection">{trackNames.map(track => <button type="button" key={track} disabled={!projectId} aria-pressed={selection?.tracks.includes(track) ?? false} onClick={() => dispatch({ type: 'select', selection: { kind: 'track', id: track, tracks: [track] } })}>{track.replaceAll('_', ' ')}</button>)}</div>
    {selected && <div className="timeline-inspector"><h3>{selected.label}</h3><pre>{JSON.stringify(selected.detail, null, 2)}</pre>
      <button type="button" disabled={busy || Boolean(draft) || selected.kind === 'stance' || !data?.semantic?.available} onClick={beginDraft}>Draft timing edit</button></div>}
    <fieldset className="timeline-editor" disabled={busy}><legend>Auditable timing edits</legend>
      <label>Author <input value={author} onChange={e => setAuthor(e.target.value)} /></label><label>Reason <input value={reason} onChange={e => setReason(e.target.value)} /></label>
      {draft && <div><p>Draft for {draft.entity.id} · expected revision {base?.expected_revision}</p>
        <label>{draft.entity.interval ? 'Boundary start' : 'Keyframe time'} <input value={draft.start} onChange={e => setDraft({ ...draft, start: e.target.value })} /></label>
        {draft.entity.interval && <label>Boundary end <input value={draft.end} onChange={e => setDraft({ ...draft, end: e.target.value })} /></label>}
        <button type="button" onClick={stageDraft}>Queue timing edit</button></div>}
      {queue.length > 0 && <pre aria-label="Queued edits">{JSON.stringify(queue, null, 2)}</pre>}
      <p>Queue related boundaries together. The service validates the complete batch and preserves overlaps.</p>
      <button type="button" disabled={!data?.semantic?.available || !queue.length || Boolean(draft) || !author.trim() || !reason.trim()} onClick={() => void command('apply')}>Apply edits</button>{' '}
      <button type="button" disabled={!base} onClick={() => { setQueue([]); setDraft(null); setBase(null); setError('') }}>Cancel drafts</button>{' '}
      <button type="button" disabled={Boolean(base) || !author.trim() || !reason.trim() || !data?.semantic?.effective_edit_revision} onClick={() => void command('undo')}>Undo edit</button>{' '}
      <button type="button" disabled={Boolean(base) || !author.trim() || !reason.trim() || data?.semantic?.origin !== 'manual'} onClick={() => void command('reset')}>Reset to automatic</button>
    </fieldset>
  </section>
}
