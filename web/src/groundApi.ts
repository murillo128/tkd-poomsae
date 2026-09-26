import { readJson } from './projectApi'
export type XY = [number, number]
export interface Quality { state: string; score?: number | null; uncertainty?: number | null }
export interface Geometry { kind: string; axis_start: XY | null; axis_end: XY | null; supported_points: XY[]; polygon: XY[] | null; approximated: boolean }
export interface Foot { foot: 'left' | 'right'; global_seconds: number; xy_ground: XY | null; geometry: Geometry; status: string; event_id: string | null; contact: { state: string; quality: Quality }; reasons: string[]; position_uncertainty: number | null; angle_uncertainty: number | null }
export interface RootPoint { id: string; global_seconds: number; xy_ground: XY | null; quality: Quality }
export interface GroundFrame { id: string; status: string; requested_seconds: number; sampled_seconds: number; root: RootPoint | null; feet: Foot[]; path_run: number | null; contact_event: { id: string; sample: { support: string; left: { state: string }; right: { state: string } } } | null }
export interface Placement { footprint: { id: string; foot: 'left' | 'right'; xy_ground: XY | null; interval: { start: number; end: number }; quality: Quality }; kind: string; event_seconds: number; geometry: Geometry }
export interface Rotation { pivot: { id: string; foot: 'left' | 'right'; interval: { start: number; end: number }; region: string; rotation_rad: number | null; quality: Quality }; placement_ids: string[]; translations: { landmark: string; positions: XY[] }[]; classification: string }
export interface Relation { first_id: string; second_id: string; measurements: { name: string; value: number | null; unit: string; quality: Quality }[] }
export interface GroundMetadata { world_unit: 'm' | 'arbitrary'; participant_id: string; scene_bounds: { minimum_xy: XY; maximum_xy: XY } | null; start_seconds: number; end_seconds: number }
export interface SnapshotResponse { revision: string; available: boolean; reason: string | null; ground_view: GroundMetadata | null; snapshot: GroundFrame | null }
interface Page<T> { revision: string; available: boolean; reason: string | null; rows: T[]; next_cursor: number | null }
export interface GroundSummary { frames: GroundFrame[]; placements: Placement[]; rotations: Rotation[]; relations: Relation[] }
const path = (id: string) => `/api/projects/${encodeURIComponent(id)}/inspection/ground`
export function readGroundSnapshot(id: string, seconds: number, signal: AbortSignal, revision?: string): Promise<SnapshotResponse> {
  const query = new URLSearchParams({ seconds: String(seconds) })
  if (revision) query.set('expected_revision', revision)
  return readJson(`${path(id)}/snapshot?${query}`, signal)
}
async function collection<T>(id: string, name: string, meta: GroundMetadata, revision: string, signal: AbortSignal): Promise<T[]> {
  const rows: T[] = []
  // Every request respects service window/page bounds. Deduplicate overlapping intervals.
  for (let start = meta.start_seconds; start <= meta.end_seconds; start += 30) {
    signal.throwIfAborted()
    let cursor: number | null = 0
    do {
      const query = new URLSearchParams({ collection: name, start: String(start), end: String(Math.min(start + 30, meta.end_seconds)), limit: '256', cursor: String(cursor), expected_revision: revision })
      const page: Page<T> = await readJson(`${path(id)}/window?${query}`, signal)
      if (page.revision !== revision || !page.available) throw new Error(page.reason || 'Ground revision changed')
      signal.throwIfAborted()
      if (page.next_cursor !== null && page.next_cursor <= cursor!) throw new Error('Ground pagination did not advance')
      rows.push(...page.rows)
      cursor = page.next_cursor
    } while (cursor !== null)
  }
  return [...new Map(rows.map(row => [(row as { id: string }).id, row])).values()]
}
export async function readGroundSummary(id: string, meta: GroundMetadata, revision: string, signal: AbortSignal): Promise<GroundSummary> {
  const [frames, placements, rotations, relations] = await Promise.all([
    collection<GroundFrame>(id, 'ground_frames', meta, revision, signal),
    collection<Placement>(id, 'placements', meta, revision, signal),
    collection<Rotation>(id, 'rotations', meta, revision, signal),
    collection<Relation>(id, 'placement_relations', meta, revision, signal),
  ])
  return { frames, placements, rotations, relations }
}
