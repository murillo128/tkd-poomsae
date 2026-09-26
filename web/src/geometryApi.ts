import type { Action, Calibration, MotionSample } from '../../contracts/types'
import { readJson } from './projectApi'

export interface Sample extends MotionSample { id: string; missing_mask?: Record<string, boolean> }
export interface Window<T> { revision: string; available: boolean; reason: string | null; unit: string; rows: T[]; next_cursor: number | null }
export interface GeometryData {
  revision: string; samples: Sample[]; actions: Action[]; calibration: Calibration | null
  unit: string; cameraSizes?: Record<string, [number, number]>; reasons: string[]; truncated: boolean
}
export function inspectionPath(project: string) { return `/api/projects/${encodeURIComponent(project)}/inspection` }
export async function readGeometry(project: string, seconds: number, signal: AbortSignal): Promise<GeometryData> {
  const path = inspectionPath(project)
  const info = await readJson<{ revision: string; products: Record<string, { available: boolean; reason: string | null }> }>(path, signal)
  const start = Math.max(0, seconds - 1), end = seconds + 1
  const query = `start=${start}&end=${end}&limit=256&expected_revision=${encodeURIComponent(info.revision)}`
  const [motion, cameras, semantic] = await Promise.all([
    readJson<Window<Sample>>(`${path}/reconstruction/window?${query}`, signal),
    readJson<{ revision: string; available: boolean; reason: string | null; calibration: Calibration | null }>(`${path}/calibration?expected_revision=${encodeURIComponent(info.revision)}`, signal),
    info.products.semantics?.available
      ? readJson<Window<Action>>(`${path}/semantics/window?collection=actions&${query}`, signal)
      : Promise.resolve(null),
  ])
  if ([motion, cameras, semantic].some(value => value && value.revision !== info.revision)) throw new Error('Geometry revision changed; refresh project')
  const cameraSizes: Record<string, [number, number]> = {}
  const cameraReasons: string[] = []
  await Promise.all((cameras.calibration?.cameras ?? []).slice(0, 16).map(async camera => {
    try {
      const media = await readJson<{ source_id: string; first_frame: { oriented_width_px: number; oriented_height_px: number } }>(
        `/api/projects/${encodeURIComponent(project)}/media/${encodeURIComponent(camera.camera_id)}/metadata`, signal)
      const { oriented_width_px: width, oriented_height_px: height } = media.first_frame
      if (media.source_id !== camera.source_id || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) throw new Error('camera/source dimensions changed')
      cameraSizes[camera.camera_id] = [width, height]
    } catch (error) {
      if (signal.aborted) throw error
      cameraReasons.push(`${camera.camera_id}: image bounds unavailable; frustum omitted`)
    }
  }))
  return { revision: info.revision, samples: motion.available ? motion.rows : [], actions: semantic?.rows ?? [],
    calibration: cameras.available ? cameras.calibration : null, unit: motion.unit, cameraSizes,
    truncated: (cameras.calibration?.cameras.length ?? 0) > 16 || motion.next_cursor !== null || (semantic?.next_cursor ?? null) !== null,
    reasons: [motion.reason, cameras.reason, semantic?.reason, ...cameraReasons].filter((reason): reason is string => Boolean(reason)) }
}
export function readEntity(project: string, id: string, revision: string, signal: AbortSignal) {
  return readJson<{ entity: unknown; source_evidence_reason: string | null; source_evidence: unknown[] }>(
    `${inspectionPath(project)}/entities?id=${encodeURIComponent(id)}&expected_revision=${encodeURIComponent(revision)}`, signal)
}
