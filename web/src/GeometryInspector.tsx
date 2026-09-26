import { useEffect, useState } from 'react'
import { readEntity, type GeometryData } from './geometryApi'
import type { Selection } from './playback'
export function GeometryInspector({ project, data, selection }: { project: string | null; data: GeometryData | null; selection: Selection | null }) {
  const [detail, setDetail] = useState<Awaited<ReturnType<typeof readEntity>> | null>(null)
  const [message, setMessage] = useState('')
  useEffect(() => {
    setDetail(null); setMessage('')
    if (!project || !data || selection?.kind !== 'entity') return
    const controller = new AbortController()
    readEntity(project, selection.id, data.revision, controller.signal).then(value => {
      if (!controller.signal.aborted) setDetail(value)
    }).catch(error => { if (!controller.signal.aborted) setMessage(String(error)) })
    return () => controller.abort()
  }, [project, data?.revision, selection?.id, selection?.kind])
  return <>{detail && <><pre className="geometry-values">{JSON.stringify(detail.entity, null, 2)}</pre><p>{detail.source_evidence_reason}</p><details><summary>Contributing camera evidence</summary><pre className="geometry-values">{JSON.stringify(detail.source_evidence, null, 2)}</pre></details></>}
    {message && <p role="status">Selected geometry unavailable: {message}</p>}</>
}
