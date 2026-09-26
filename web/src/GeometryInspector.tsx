import { useEffect, useState } from 'react'
import { readEntity, type EntityDetail, type GeometryData, type SourceEvidence } from './geometryApi'
import { inspectionPath } from './geometryApi'
import { readJson } from './projectApi'
import { exactImage, nearestFrame } from './cameraApi'
import type { Selection } from './playback'
import { sampleAt } from './sceneGeometry'

// A joint selection follows its physical landmark across native reconstruction instants.
export function selectedLandmarkName(selection: Selection | null): string | null {
  return selection?.kind === 'entity' ? /\/samples\/\d+\/([^/]+)$/.exec(selection.id)?.[1] ?? null : null
}

function EvidenceFrame({ project, evidence, cursorSeconds, onCamera }: {
  project: string; evidence: SourceEvidence; cursorSeconds: number; onCamera?: (camera: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [image, setImage] = useState<string | null>(null)
  const [error, setError] = useState('')
  const frame = evidence.frame
  useEffect(() => {
    setImage(null); setError('')
    if (!open) return
    const controller = new AbortController()
    let url: string | null = null
    async function load() {
      const native = (await nearestFrame(project, frame.camera_id, frame.source_seconds, controller.signal)).frame
      if (native.source_id !== frame.source_id || native.pts !== frame.pts ||
          native.time_base_num !== frame.time_base_num || native.time_base_den !== frame.time_base_den) {
        throw new Error('Contributing frame identity changed')
      }
      url = await exactImage(project, frame.camera_id, native, controller.signal)
      if (controller.signal.aborted) { URL.revokeObjectURL(url); return }
      setImage(url)
    }
    void load().catch(e => { if (!controller.signal.aborted) setError(String(e)) })
    return () => { controller.abort(); if (url) URL.revokeObjectURL(url) }
  }, [project, evidence, open])
  const gap = frame.global_seconds === null || !Number.isFinite(frame.global_seconds) ? null : frame.global_seconds - cursorSeconds
  return <div className="source-evidence">
    <button type="button" onClick={() => { setOpen(true); onCamera?.(frame.camera_id) }}>Inspect {frame.camera_id} PTS {frame.pts}</button>
    <p>{evidence.observation_id} · source {frame.source_seconds} s · global {frame.global_seconds ?? 'unavailable'} s · sample-time mismatch {gap === null ? 'unavailable' : `${gap.toFixed(6)} s`}. Global cursor preserved.</p>
    {open && !image && !error && <p role="status">Loading contributing frame…</p>}
    {image && <img className="evidence-image" src={image} alt={`Contributing ${frame.camera_id} PTS ${frame.pts}`} />}
    {error && <p role="alert">Contributing frame unavailable: {error}</p>}
    <details><summary>Native scores and generating provenance</summary><pre>{JSON.stringify(evidence, null, 2)}</pre></details>
  </div>
}

export function GeometryInspector({ project, data, selection, cursorSeconds, revision, onCamera, onEntity }: {
  project: string | null; data: GeometryData | null; selection: Selection | null; cursorSeconds: number
  revision?: string; onCamera?: (camera: string) => void; onEntity?: (id: string) => void
}) {
  const [result, setResult] = useState<{ key: string; detail?: EntityDetail; error?: string } | null>(null)
  const [attempt, setAttempt] = useState(0)
  const landmark = selectedLandmarkName(selection)
  const sample = data && sampleAt(data.samples, cursorSeconds)
  const point = sample?.landmarks.find(value => value.name === landmark)
  const landmarkAvailable = Boolean(point?.xyz_world && !sample?.missing_mask?.[landmark!])
  const entityId = selection?.kind === 'entity'
    ? landmark ? landmarkAvailable ? `${sample!.id}/${landmark}` : null : selection.id
    : null
  const requestKey = project && entityId ? JSON.stringify([project, revision, landmark ? data?.revision : null, entityId, attempt]) : null
  useEffect(() => {
    setResult(null)
    if (!project || !entityId || !requestKey) return
    const controller = new AbortController()
    async function load() {
      // Semantic/ground/camera inspection must remain usable without a 3D product.
      const meta = await readJson<{ revision: string }>(inspectionPath(project!), controller.signal)
      const detail = await readEntity(project!, entityId!, meta.revision, controller.signal)
      if (!controller.signal.aborted) setResult({ key: requestKey!, detail })
    }
    void load().catch(error => { if (!controller.signal.aborted) setResult({ key: requestKey, error: String(error) }) })
    return () => controller.abort()
  }, [project, entityId, requestKey])
  const current = result?.key === requestKey ? result : null
  const detail = current?.detail
  const unavailable = landmark && !landmarkAvailable
    ? !data ? 'Current native geometry is unavailable.'
      : !sample ? `No native sample at ${cursorSeconds.toFixed(3)} s; ${landmark} unavailable.`
        : `${landmark} unavailable at ${cursorSeconds.toFixed(3)} s: landmark coordinates are missing or masked.`
    : null
  const entity = detail?.entity as { interval?: { start: number; end: number } } | undefined
  return <>
    <p>Derived quality and uncertainty describe evidence state. Raw detector scores are model outputs. Interpolation, inference and manual edits are distinct; reprojection is internal consistency, not measured model accuracy.</p>
    {detail && <>
      <p>Coordinates: {detail.unit === 'm' ? 'metres (m)' : detail.unit === 'px' ? 'pixels (px)' : 'arbitrary world units · metric scale unresolved'}{entity?.interval && ` · duration ${(entity.interval.end - entity.interval.start).toFixed(6)} s`}</p>
      <pre className="geometry-values">{JSON.stringify(detail.entity, null, 2)}</pre>
      <p>Origin {detail.origin ?? 'automatic'} · edit revision {detail.effective_edit_revision ?? 0} · artifact {detail.artifact_revision ?? 'unavailable'}</p>
      <details><summary>Generating model, configuration and revisions</summary><pre>{JSON.stringify({ provenance: detail.provenance, revisions: detail.generating_revisions }, null, 2)}</pre></details>
      {Boolean(detail.physical_evidence?.length) && <details><summary>Contributing physical samples</summary>{detail.physical_evidence!.map(sample => <div key={sample.id}>
        <button type="button" disabled={!onEntity} onClick={() => onEntity?.(sample.id)}>Inspect physical sample {sample.global_seconds} s</button>
        <p>{sample.id} · sample-time mismatch {(sample.global_seconds - cursorSeconds).toFixed(6)} s · {JSON.stringify(sample.quality)}</p>
      </div>)}</details>}
      {detail.source_evidence_reason && <p role="status">{detail.source_evidence_reason}</p>}
      <details><summary>Contributing camera evidence</summary><pre className="camera-evidence-values">{JSON.stringify(detail.source_evidence, null, 2)}</pre>{detail.source_evidence.map((item, index) => <EvidenceFrame key={`${requestKey}/${index}`} project={project!} evidence={item} cursorSeconds={cursorSeconds} onCamera={onCamera} />)}</details>
    </>}
    {requestKey && !current && <p role="status">Loading selected evidence…</p>}
    {unavailable && <p role="status">Selected geometry unavailable: {unavailable}</p>}
    {current?.error && <p role="status">Selected evidence unavailable: {current.error} <button onClick={() => setAttempt(n => n + 1)}>Retry selected evidence</button></p>}
  </>
}
