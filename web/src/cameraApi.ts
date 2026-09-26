import type { Observation, SyncOffset } from '../../contracts/types'

import { API_ROOT as root } from './projectApi'
export function mediaPath(project: string, camera: string) {
  return `${root}/api/projects/${encodeURIComponent(project)}/media/${encodeURIComponent(camera)}`
}
function inspectionPath(project: string) { return `${root}/api/projects/${encodeURIComponent(project)}/inspection` }
export interface NativeFrame {
  ordinal: number; pts: number; time_base_num: number; time_base_den: number
  source_seconds: number; source_id: string; source_sha256: string; camera_id: string
  oriented_width_px: number; oriented_height_px: number; rotation_degrees: number
}
export interface MediaMetadata {
  camera_id: string; source_id: string; source_sha256: string; frame_count: number
  browser_playback: boolean; browser_playback_reason: string | null
  first_frame: NativeFrame; last_frame: NativeFrame
}
export interface TimeMapping {
  revision: string; camera: string; source_id: string; source_seconds: number
  effective_offset_seconds: number; offset: SyncOffset
  nearest: NativeFrame; nearest_gap_seconds: number; interpolation: string
}
async function json<T>(url: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal })
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.detail || `Local service returned ${response.status}`)
  }
  return response.json()
}
export const readMedia = (project: string, camera: string, signal: AbortSignal) =>
  json<MediaMetadata>(`${mediaPath(project, camera)}/metadata`, signal)
export const mapTime = (project: string, camera: string, seconds: number, signal: AbortSignal) =>
  json<TimeMapping>(`${inspectionPath(project)}/time/${encodeURIComponent(camera)}?seconds=${seconds}`, signal)
export const readFramePage = (project: string, camera: string, ordinal: number, signal: AbortSignal) =>
  json<{ source_sha256: string; frames: NativeFrame[] }>(`${mediaPath(project, camera)}/frames?start=${Math.max(0, ordinal - 128)}&limit=256`, signal)
export const nearestFrame = (project: string, camera: string, seconds: number, signal: AbortSignal) =>
  json<{ frame: NativeFrame }>(`${mediaPath(project, camera)}/frames/nearest?seconds=${seconds}`, signal)

// Match the local service's two decoder slots; variable camera counts must not race them.
let activeDecodes = 0
const decodeWaiters: Array<() => void> = []
async function acquireDecode() {
  if (activeDecodes < 2) activeDecodes++
  else await new Promise<void>(resolve => decodeWaiters.push(resolve))
  return () => {
    const next = decodeWaiters.shift()
    if (next) next()
    else activeDecodes--
  }
}
export async function exactImage(project: string, camera: string, frame: NativeFrame, signal: AbortSignal): Promise<string> {
  const release = await acquireDecode()
  try {
    signal.throwIfAborted()
    const response = await fetch(`${mediaPath(project, camera)}/frames/${frame.ordinal}/image`, { signal })
    if (!response.ok) throw new Error(`Exact frame unavailable (${response.status})`)
    if (response.headers.get('X-Source-Ordinal') !== String(frame.ordinal) ||
        response.headers.get('X-Source-PTS') !== String(frame.pts) ||
        response.headers.get('X-Source-SHA256') !== frame.source_sha256) throw new Error('Exact frame identity changed')
    return URL.createObjectURL(await response.blob())
  } finally { release() }
}
export async function observationsAt(project: string, frame: NativeFrame, offset: number, revision: string, signal: AbortSignal) {
  const time = frame.source_seconds + offset
  const rows: Observation[] = []
  let cursor: number | null = 0
  while (cursor !== null) {
    const page: { revision: string; available: boolean; reason: string | null; rows: Observation[]; next_cursor: number | null } = await json(
      `${inspectionPath(project)}/observations/window?collection=observations&start=${time - 1e-8}&end=${time + 1e-8}&limit=256&cursor=${cursor}&expected_revision=${encodeURIComponent(revision)}`, signal)
    if (!page.available) throw new Error(page.reason || 'Observations unavailable')
    if (page.revision !== revision) throw new Error('Observation revision changed')
    rows.push(...page.rows.filter(row => matchesObservation(row, frame)))
    if (page.next_cursor !== null && page.next_cursor <= cursor) throw new Error('Invalid observation page cursor')
    cursor = page.next_cursor
  }
  return rows
}
export function matchesObservation(row: Observation, frame: NativeFrame): boolean {
  return row.frame.source_id === frame.source_id && row.frame.camera_id === frame.camera_id &&
    row.frame.pts === frame.pts && row.frame.time_base_num === frame.time_base_num &&
    row.frame.time_base_den === frame.time_base_den && Math.abs(row.frame.source_seconds - frame.source_seconds) < 1e-8
}
