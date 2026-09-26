import { useEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import type { Landmark2D, Observation } from '../../contracts/types'
import type { CameraClock } from './playback'
import { exactImage, mapTime, mediaPath, nearestFrame, observationsAt, readFramePage, readMedia, type MediaMetadata, type NativeFrame, type TimeMapping } from './cameraApi'

interface Props {
  project: string; camera: string; seconds: number; playing: boolean; speed: number; selected: boolean
  onClock: (clock: CameraClock) => void
  onDelivered: (camera: string, seconds: number | null) => void
  onSelect: (id: string, description: string) => void
}
interface Delivery { frame: NativeFrame; requested: number; image: string | null; observations: Observation[]; observationStatus: string }
export function CameraPanel({ project, camera, seconds, playing, speed, selected, onClock, onDelivered, onSelect }: Props) {
  const [overlays, setOverlays] = useState(true)
  const [metadata, setMetadata] = useState<MediaMetadata | null>(null)
  const [mapping, setMapping] = useState<TimeMapping | null>(null)
  const [delivery, setDelivery] = useState<Delivery | null>(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [status, setStatus] = useState('Video frame unavailable — loading source')
  const video = useRef<HTMLVideoElement>(null)
  const callbacks = useRef({ onClock, onDelivered })
  callbacks.current = { onClock, onDelivered }
  const cursor = useRef(seconds)
  cursor.current = seconds
  const epoch = useRef(0)
  const imageUrl = useRef<string | null>(null)
  function clearImage() {
    if (imageUrl.current) URL.revokeObjectURL(imageUrl.current)
    imageUrl.current = null
  }
  useEffect(() => {
    const controller = new AbortController()
    readMedia(project, camera, controller.signal).then(value => {
      if (value.camera_id !== camera || value.first_frame.source_sha256 !== value.source_sha256) throw new Error('Source identity changed')
      if (!controller.signal.aborted) setMetadata(value)
    }).catch(error => { if (!controller.signal.aborted) setStatus(`Video frame unavailable — ${error}`) })
    return () => { controller.abort(); clearImage() }
  }, [project, camera])

  // A paused cursor uses service-decoded frames. Browser seek completion alone cannot prove identity.
  useEffect(() => {
    if (!metadata) return
    const controller = new AbortController()
    const generation = ++epoch.current
    setDelivery(null)
    setImageLoaded(false)
    callbacks.current.onDelivered(camera, null)
    clearImage()
    setStatus('Video frame unavailable — mapping requested time')
    async function load() {
      const mapped = await mapTime(project, camera, seconds, controller.signal)
      if (mapped.camera !== camera || mapped.source_id !== metadata!.source_id || mapped.nearest.source_sha256 !== metadata!.source_sha256) throw new Error('Time mapping source changed')
      if (controller.signal.aborted) return
      setMapping(mapped)
      const page = await readFramePage(project, camera, mapped.nearest.ordinal, controller.signal)
      if (page.source_sha256 !== metadata!.source_sha256) throw new Error('Frame index source changed')
      if (controller.signal.aborted) return
      callbacks.current.onClock({ id: camera, offsetSeconds: mapped.effective_offset_seconds,
        frames: page.frames.map(frame => ({ pts: frame.pts, timeBaseNum: frame.time_base_num, timeBaseDen: frame.time_base_den })) })
      if (mapped.interpolation === 'outside_coverage') { setStatus('Unavailable — outside camera coverage'); return }
      if (playing) {
        setStatus(!metadata!.browser_playback || metadata!.first_frame.source_seconds !== 0
          ? 'Browser playback unsupported for this source timeline. Pause for exact inspection.'
          : !video.current?.requestVideoFrameCallback
            ? 'Presented frame metadata unavailable; pause for exact inspection. Overlays hidden.'
            : 'Playback uses presented frames; skipped frames are not temporal evidence.')
        return
      }
      const url = await exactImage(project, camera, mapped.nearest, controller.signal)
      if (controller.signal.aborted || generation !== epoch.current) { URL.revokeObjectURL(url); return }
      imageUrl.current = url
      const current: Delivery = { frame: mapped.nearest, requested: seconds, image: url, observations: [], observationStatus: 'Loading observations' }
      setDelivery(current)
      setStatus('Exact source frame')
      // Delivered time becomes observable only once the image has loaded.
      await loadObservations(current, mapped, controller, generation)
    }
    async function loadObservations(current: Delivery, mapped: TimeMapping, control: AbortController, generation: number) {
      try {
        const rows = await observationsAt(project, current.frame, mapped.effective_offset_seconds, mapped.revision, control.signal)
        if (!control.signal.aborted && generation === epoch.current) setDelivery({ ...current, observations: rows, observationStatus: rows.length ? 'Frame-matched observations' : 'Missing observations' })
      } catch (error) {
        if (!control.signal.aborted && generation === epoch.current) setDelivery({ ...current, observationStatus: `Observations unavailable — ${error}` })
      }
    }
    void load().catch(error => { if (!controller.signal.aborted) { setMapping(null); callbacks.current.onClock({ id: camera, offsetSeconds: Number.NaN, frames: [] }); setStatus(`Video frame unavailable — excluded or unmapped view: ${error}`) } })
    return () => { controller.abort(); epoch.current++ }
    // Playback ticks are synchronized locally below; map only on entering/leaving playback or paused seeking.
  }, [project, camera, metadata, playing, playing ? null : seconds])

  const offset = mapping?.effective_offset_seconds
  const inCoverage = metadata !== null && offset !== undefined && seconds - offset >= metadata.first_frame.source_seconds && seconds - offset <= metadata.last_frame.source_seconds
  useEffect(() => {
    if (!inCoverage) { setDelivery(null); callbacks.current.onDelivered(camera, null) }
  }, [inCoverage, camera])
  const browserPlayable = metadata?.browser_playback && metadata.first_frame.source_seconds === 0
  useEffect(() => {
    const element = video.current
    if (!element || !playing || !inCoverage || offset === undefined) { element?.pause(); return }
    const desired = seconds - offset
    element.playbackRate = speed
    if (Math.abs(element.currentTime - desired) > 0.08) {
      epoch.current++
      setDelivery(null)
      callbacks.current.onDelivered(camera, null)
      element.currentTime = desired
    }
    void element.play().catch(error => setStatus(`Playback unavailable — ${error}`))
  }, [seconds, playing, speed, offset, inCoverage])

  useEffect(() => {
    const element = video.current
    if (!element || !playing || !mapping || !metadata || !inCoverage) return
    if (!element.requestVideoFrameCallback) {
      setStatus('Presented frame metadata unavailable; pause for exact inspection. Overlays hidden.')
      return
    }
    const controller = new AbortController()
    let callbackId = 0
    let busy = false
    let presentation = 0
    const presented = (_now: number, info: VideoFrameCallbackMetadata) => {
      const serial = ++presentation
      const generation = epoch.current
      const requested = cursor.current
      // Remove the previous SVG before the compositor presents this new video frame.
      flushSync(() => setDelivery(null))
      callbacks.current.onDelivered(camera, null)
      if (!busy && !element.seeking) {
        busy = true
        void (async () => {
          const { frame } = await nearestFrame(project, camera, info.mediaTime, controller.signal)
          if (Math.abs(frame.source_seconds - info.mediaTime) > 1e-6 || frame.source_sha256 !== metadata.source_sha256) throw new Error('Presented frame identity unverified; pause for exact inspection')
          if (element.videoWidth !== frame.oriented_width_px || element.videoHeight !== frame.oriented_height_px) throw new Error('Browser image orientation differs from source; pause for exact inspection')
          let rows: Observation[] = []
          let observationStatus = 'Missing observations'
          try {
            rows = await observationsAt(project, frame, mapping.effective_offset_seconds, mapping.revision, controller.signal)
            observationStatus = rows.length ? 'Frame-matched observations' : 'Missing observations'
          } catch (error) { observationStatus = `Observations unavailable — ${error}` }
          if (!controller.signal.aborted && generation === epoch.current && serial === presentation && !element.seeking) {
            setDelivery({ frame, requested, image: null, observations: rows, observationStatus })
            callbacks.current.onDelivered(camera, frame.source_seconds + mapping.effective_offset_seconds)
          }
        })().catch(error => { if (!controller.signal.aborted) setStatus(`Playback observation limitation — ${error}`) }).finally(() => { busy = false })
      }
      callbackId = element.requestVideoFrameCallback(presented)
    }
    callbackId = element.requestVideoFrameCallback(presented)
    return () => { controller.abort(); element.cancelVideoFrameCallback(callbackId) }
  }, [project, camera, playing, mapping, metadata, inCoverage])

  const visible = inCoverage && delivery && (playing || delivery.requested === seconds)
  const frame = visible && (playing || imageLoaded) ? delivery.frame : null
  useEffect(() => {
    if (selected) callbacks.current.onDelivered(camera, frame && offset !== undefined ? frame.source_seconds + offset : null)
  }, [selected, camera, frame?.pts, frame?.source_seconds, offset])
  const points = frame && visible && overlays ? delivery.observations.flatMap(observation => observation.landmarks.map(point => ({ observation, point }))) : []
  return <article className="camera-card" aria-label={`Camera ${camera}`}>
    <h3>{camera}</h3>
    <label><input type="checkbox" checked={overlays} onChange={e => setOverlays(e.target.checked)} />Observation overlays</label>
    <p>{mapping ? `${mapping.offset.retained ? 'Retained' : 'Excluded'} · offset ${offset?.toFixed(6)} s · confidence ${mapping.offset.quality.score ?? 'unknown'} (${mapping.offset.quality.state})${mapping.offset.quality.score !== null && mapping.offset.quality.score !== undefined && mapping.offset.quality.score < 0.5 ? ' · Low confidence' : ''}` : 'Synchronization unavailable'}</p>
    <div className="camera-viewport">
      {playing && browserPlayable && <video ref={video} src={mediaPath(project, camera)} muted playsInline preload="metadata" style={{ visibility: inCoverage ? 'visible' : 'hidden' }} onSeeking={() => { epoch.current++; setDelivery(null); callbacks.current.onDelivered(camera, null) }} onError={() => setStatus('Browser playback unavailable; pause for exact source frames.')} />}
      {!playing && visible && delivery.image && <img src={delivery.image} alt={`${camera} frame ${delivery.frame.ordinal}`} onLoad={() => { setImageLoaded(true); callbacks.current.onDelivered(camera, delivery.frame.source_seconds + offset!) }} onError={() => { setImageLoaded(false); setStatus('Exact image failed to display'); callbacks.current.onDelivered(camera, null) }} />}
      {frame && <svg viewBox={`0 0 ${frame.oriented_width_px} ${frame.oriented_height_px}`} preserveAspectRatio="xMidYMid meet" aria-label={`${camera} observation overlay`}>
        {points.map(({ observation, point }, index) => point.xy_px && <g key={`${observation.id}/${point.name}/${index}`}>
          <circle cx={point.xy_px[0] + 0.5} cy={point.xy_px[1] + 0.5} r={Math.max(frame.oriented_width_px / 120, 2)} className={`landmark ${point.quality.state}`} role="button" tabIndex={0}
            aria-label={`${camera} ${point.name}`} onClick={() => onSelect(`${observation.id}/${point.name}`, `${camera} · source PTS ${observation.frame.pts} · source ${observation.frame.source_seconds} s · ${pointDescription(point)} · producer ${observation.provenance.producer}`)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(`${observation.id}/${point.name}`, `${camera} · source PTS ${observation.frame.pts} · source ${observation.frame.source_seconds} s · ${pointDescription(point)} · producer ${observation.provenance.producer}`) } }}>
            <title>{pointDescription(point)}</title>
          </circle>
        </g>)}
        {visible && delivery.observations.flatMap(row => row.regional_geometry ?? []).map((region, index) => region.axis_start_px && region.axis_end_px && <line key={`axis/${index}`} className="region-axis" x1={region.axis_start_px[0] + 0.5} y1={region.axis_start_px[1] + 0.5} x2={region.axis_end_px[0] + 0.5} y2={region.axis_end_px[1] + 0.5}><title>{region.part} · {region.availability} · {region.orientation_state} · {region.provider}</title></line>)}
        {visible && delivery.observations.flatMap(row => row.regions ?? []).map((region, index) => <rect key={index} className="crop-region" x={region.xywh_px[0]} y={region.xywh_px[1]} width={region.xywh_px[2]} height={region.xywh_px[3]}><title>{region.part} original-image crop</title></rect>)}
      </svg>}
      {(!inCoverage || (!playing && !visible) || (playing && !browserPlayable)) && <div className="viewport-placeholder">{playing && !browserPlayable ? 'Browser playback unsupported for this source timeline. Pause for exact inspection.' : !inCoverage && mapping ? 'Unavailable — outside camera coverage' : status}</div>}
    </div>
    <p role="status">{status}</p>
    <p>Requested global {seconds.toFixed(6)} s · source {offset === undefined ? 'unavailable' : (seconds - offset).toFixed(6) + ' s'}</p>
    <p>{frame ? `Delivered ordinal ${frame.ordinal} · PTS ${frame.pts} · source ${frame.source_seconds.toFixed(6)} s · global ${(frame.source_seconds + offset!).toFixed(6)} s · sample mismatch ${(frame.source_seconds + offset! - seconds).toFixed(6)} s` : 'Delivered frame identity unavailable'}</p>
    <p>{visible ? delivery.observationStatus : 'Overlays hidden until source frame identity is established'}</p>
    {visible && overlays && delivery.observations.map(row => <details key={row.id}><summary>Observation confidence and provenance</summary><p>{row.id}</p>
      <p>{row.provenance.producer} · {row.provenance.model ?? 'no model'} · {row.provenance.model_version ?? 'no version'}</p>
      {row.region_quality?.map(region => <p key={region.part}>{region.part}: {region.usable ? 'usable' : 'unusable'} {region.reasons?.join('; ')}</p>)}
      {row.landmarks.map(point => <p key={point.name}>{point.name}: {pointDescription(point)}</p>)}
    </details>)}
  </article>
}
function pointDescription(point: Landmark2D) {
  return `${point.xy_px ? `Original image (${point.xy_px.join(', ')})` : 'Missing'} · raw score ${point.raw_score?.value ?? 'missing'} (${point.raw_score?.domain ?? 'unspecified domain'}; range ${point.raw_score?.range_min ?? '?'}–${point.raw_score?.range_max ?? '?'}) · visibility ${point.raw_visibility ?? 'missing'} · derived quality ${point.quality.score ?? 'unknown'} · ${point.quality.state}`
}
