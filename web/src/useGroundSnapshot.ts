import { useEffect, useRef, useState } from 'react'
import { readGroundSnapshot, type SnapshotResponse } from './groundApi'

// One request at a time. Cursor ticks replace the next desired time, rather than
// cancelling every request before a slower local service can deliver geometry.
export function useGroundSnapshot(projectId: string | null, revision: string | null, seconds: number) {
  const [current, setCurrent] = useState<{ projectId: string; revision: string; seconds: number; response: SnapshotResponse } | null>(null)
  const [error, setError] = useState<{ projectId: string; revision: string; message: string } | null>(null)
  const latestSeconds = useRef(seconds)
  const requestLatest = useRef<(() => void) | null>(null)

  useEffect(() => {
    latestSeconds.current = seconds
    requestLatest.current?.()
  }, [seconds])

  useEffect(() => {
    setCurrent(null)
    setError(null)
    if (!projectId || !revision) return
    const controller = new AbortController()
    let inFlight = false
    let requestedSeconds: number | null = null
    function request() {
      if (controller.signal.aborted || inFlight || requestedSeconds === latestSeconds.current) return
      const target = latestSeconds.current
      requestedSeconds = target
      inFlight = true
      readGroundSnapshot(projectId!, target, controller.signal, revision!).then(response => {
        if (controller.signal.aborted) return
        if (response.revision !== revision) throw new Error('Ground revision changed')
        setError(null)
        setCurrent({ projectId: projectId!, revision: revision!, seconds: target, response })
      }).catch(reason => {
        if (!controller.signal.aborted) setError({ projectId: projectId!, revision: revision!, message: String(reason) })
      }).finally(() => {
        inFlight = false
        if (!controller.signal.aborted && latestSeconds.current !== target) request()
      })
    }
    requestLatest.current = request
    request()
    return () => {
      controller.abort()
      requestLatest.current = null
    }
  }, [projectId, revision])

  return {
    current: current?.projectId === projectId && current.revision === revision ? current : null,
    error: error?.projectId === projectId && error.revision === revision ? error.message : null,
  }
}
