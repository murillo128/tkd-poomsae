import { useEffect, useState } from 'react'
import { readEntity, type GeometryData } from './geometryApi'
import type { Selection } from './playback'
import { sampleAt } from './sceneGeometry'

// Joint entity IDs identify a native instant; their landmark suffix identifies
// the physical selection across seeks. Other entity IDs remain fixed selections.
export function selectedLandmarkName(selection: Selection | null): string | null {
  return selection?.kind === 'entity' ? /\/samples\/\d+\/([^/]+)$/.exec(selection.id)?.[1] ?? null : null
}

export function GeometryInspector({ project, data, selection, cursorSeconds }: {
  project: string | null; data: GeometryData | null; selection: Selection | null; cursorSeconds: number
}) {
  const [result, setResult] = useState<{
    key: string; detail?: Awaited<ReturnType<typeof readEntity>>; error?: string
  } | null>(null)
  const landmark = selectedLandmarkName(selection)
  const sample = data && sampleAt(data.samples, cursorSeconds)
  const point = sample?.landmarks.find(value => value.name === landmark)
  const landmarkAvailable = Boolean(point?.xyz_world && !sample?.missing_mask?.[landmark!])
  const entityId = selection?.kind === 'entity'
    ? landmark ? landmarkAvailable ? `${sample!.id}/${landmark}` : null : selection.id
    : null
  const revision = data?.revision
  const requestKey = project && revision && entityId ? JSON.stringify([project, revision, entityId]) : null

  useEffect(() => {
    setResult(null)
    if (!project || !revision || !entityId || !requestKey) return
    const controller = new AbortController()
    readEntity(project, entityId, revision, controller.signal).then(detail => {
      if (!controller.signal.aborted) setResult({ key: requestKey, detail })
    }).catch(error => {
      if (!controller.signal.aborted) setResult({ key: requestKey, error: String(error) })
    })
    return () => controller.abort()
  }, [project, revision, entityId, requestKey])

  // Bind displayed values as well as responses to the current native target so
  // the previous pose/evidence cannot flash during a seek or a project change.
  const current = result?.key === requestKey ? result : null
  const detail = current?.detail
  const unavailable = landmark && !landmarkAvailable
    ? !data ? 'Current native geometry is unavailable.'
      : !sample ? `No native sample at ${cursorSeconds.toFixed(3)} s; ${landmark} unavailable.`
        : `${landmark} unavailable at ${cursorSeconds.toFixed(3)} s: landmark coordinates are missing or masked.`
    : null
  return <>{detail && <><pre className="geometry-values">{JSON.stringify(detail.entity, null, 2)}</pre><p>{detail.source_evidence_reason}</p><details><summary>Contributing camera evidence</summary><pre className="geometry-values">{JSON.stringify(detail.source_evidence, null, 2)}</pre></details></>}
    {unavailable && <p role="status">Selected geometry unavailable: {unavailable}</p>}
    {current?.error && <p role="status">Selected geometry unavailable: {current.error}</p>}</>
}
