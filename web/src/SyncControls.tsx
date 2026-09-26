import { useEffect, useRef, useState } from 'react'
import { mapTime, type TimeMapping } from './cameraApi'
import { API_ROOT } from './projectApi'
import { inspectionPath } from './geometryApi'

export function SyncControls({ project, camera, revision, onChanged }: {
  project: string; camera: string; revision: string; onChanged: () => void
}) {
  const [mapping, setMapping] = useState<TimeMapping | null>(null)
  const [draft, setDraft] = useState<string | null>(null)
  const [base, setBase] = useState<number | null>(null)
  const [author, setAuthor] = useState('')
  const [reason, setReason] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const write = useRef<AbortController | null>(null)
  useEffect(() => () => write.current?.abort(), [])
  useEffect(() => {
    const controller = new AbortController()
    setMapping(null)
    mapTime(project, camera, 0, controller.signal).then(value => {
      if (!value.offset?.quality || !Number.isFinite(value.effective_offset_seconds) || !Number.isInteger(value.sync_revision)) throw new Error('Synchronization edit capability unavailable')
      if (!controller.signal.aborted) setMapping(value)
    }).catch(e => { if (!controller.signal.aborted) setError(`Synchronization unavailable: ${String(e)}`) })
    return () => controller.abort()
  }, [project, camera, revision, attempt])
  async function save() {
    const seconds = Number(draft)
    if (busy || !mapping || base === null || !draft?.trim() || !Number.isFinite(seconds) || !author.trim() || !reason.trim()) {
      setError('Enter a finite offset, author and reason. Draft retained.'); return
    }
    const controller = new AbortController(); write.current = controller
    setBusy(true); setError('')
    try {
      const response = await fetch(`${API_ROOT}${inspectionPath(project)}/sync-offset`, {
        method: 'POST', signal: controller.signal,
        headers: { 'Content-Type': 'application/json', 'x-tkd-local-request': '1' },
        body: JSON.stringify({ expected_revision: base, camera, offset_seconds: seconds, author, reason, source: 'web-sync-controls' }),
      })
      if (!response.ok) {
        if (response.status === 409) onChanged()
        const body = await response.json().catch(() => null)
        throw new Error(`${response.status === 409 ? 'Stale revision / conflict' : 'Save rejected'} (${response.status}): ${JSON.stringify(body?.detail ?? 'local service error')}. Draft retained.`)
      }
      if (controller.signal.aborted) return
      setDraft(null); setBase(null); onChanged()
    } catch (e) { if (!controller.signal.aborted) setError(String(e)) }
    finally { if (!controller.signal.aborted) setBusy(false) }
  }
  return <fieldset className="sync-editor" disabled={busy}><legend>{camera} synchronization</legend>
    <p>Automatic estimate {mapping?.offset.automatic_seconds ?? 'unavailable'} s · effective {mapping?.effective_offset_seconds ?? 'unavailable'} s · confidence {mapping?.offset.quality.score ?? 'unknown'} ({mapping?.offset.quality.state ?? 'unknown'})</p>
    <p>Automatic synchronization is preserved. Saving invalidates incompatible 3D, ground and parser artifacts; inspection never reruns models.</p>
    {Boolean(mapping?.manual_revision) && <details><summary>Manual edit provenance</summary><pre>{JSON.stringify(mapping?.manual_revision, null, 2)}</pre></details>}
    {error && <p role="alert">{error}</p>}
    <label>{camera} offset seconds <input value={draft ?? String(mapping?.effective_offset_seconds ?? '')} disabled={!mapping || !Number.isInteger(mapping.sync_revision)} onChange={e => {
      setDraft(e.target.value); setBase(previous => previous ?? mapping!.sync_revision)
    }} /></label>
    <label>{camera} sync author <input value={author} onChange={e => setAuthor(e.target.value)} /></label>
    <label>{camera} sync reason <input value={reason} onChange={e => setReason(e.target.value)} /></label>
    <button type="button" disabled={!mapping || base === null || !author.trim() || !reason.trim()} onClick={() => void save()}>Save {camera} offset</button>
    <button type="button" onClick={() => setAttempt(n => n + 1)}>Reload {camera} synchronization</button>
    <button type="button" disabled={draft === null} onClick={() => { setDraft(null); setBase(null); setError('') }}>Discard {camera} offset draft</button>
  </fieldset>
}
