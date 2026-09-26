import type { Action, Interval, Keyframe, Phase, SequenceStep, StanceState } from '../../contracts/types'
import { API_ROOT, readJson } from './projectApi'
export interface ProductInfo {
  available: boolean; reason: string | null; id: string; artifact_revision: string
  effective_edit_revision: number; origin: 'automatic' | 'manual'; time_bounds: Interval | null
  arrays: { id: string }[]
}
interface Metadata { revision: string; products: Record<string, ProductInfo> }
export interface TimelineData {
  revision: string; semantic: ProductInfo | null; motion: ProductInfo | null; bounds: Interval | null
  steps: SequenceStep[]; actions: Action[]; phases: Phase[]; keyframes: Keyframe[]; stances: StanceState[]
}
interface Page<T> { revision: string; available: boolean; reason: string | null; rows: T[]; next_cursor: number | null }
export type EditOperation = { kind: 'boundary'; target_id: string; interval: Interval } | { kind: 'keyframe_move'; target_id: string; global_seconds: number }
export interface EditBase { expected_revision: number; automatic_revision: string }
const path = (id: string) => `/api/projects/${encodeURIComponent(id)}/inspection`
async function collection<T extends { id: string }>(id: string, name: string, bounds: Interval, revision: string, signal: AbortSignal): Promise<T[]> {
  const rows = new Map<string, T>()
  // Leave headroom below the strict 30s API cap: fractional timestamps can
  // make a nominal 30s difference slightly larger. Shared endpoints keep
  // consecutive closed windows gap-free despite floating-point rounding.
  const windowSeconds = 29
  for (let start = bounds.start; start <= bounds.end; start += windowSeconds) {
    let cursor: number | null = 0
    do {
      signal.throwIfAborted()
      const query = new URLSearchParams({ collection: name, start: String(start), end: String(Math.min(start + windowSeconds, bounds.end)), limit: '256', cursor: String(cursor), expected_revision: revision })
      const page: Page<T> = await readJson(`${path(id)}/semantics/window?${query}`, signal)
      if (page.revision !== revision || !page.available) throw new Error(page.reason || 'Timeline revision changed')
      if (page.next_cursor !== null && page.next_cursor <= cursor!) throw new Error('Timeline pagination did not advance')
      for (const row of page.rows) rows.set(row.id, row)
      cursor = page.next_cursor
    } while (cursor !== null)
  }
  return [...rows.values()]
}
export async function readTimeline(id: string, signal: AbortSignal): Promise<TimelineData> {
  const meta = await readJson<Metadata>(path(id), signal)
  if (!meta.products || !meta.revision) throw new Error('Inspection metadata unavailable')
  const semantic = meta.products.semantics ?? null
  const motion = meta.products.reconstruction ?? null
  const bounds = semantic?.available && semantic.time_bounds ? semantic.time_bounds : motion?.available ? motion.time_bounds : null
  const data: TimelineData = { revision: meta.revision, semantic, motion, bounds, steps: [], actions: [], phases: [], keyframes: [], stances: [] }
  if (!semantic?.available || !semantic.time_bounds) return data
  const [steps, actions, phases, keyframes, stances] = await Promise.all([
    collection<SequenceStep>(id, 'steps', semantic.time_bounds, meta.revision, signal),
    collection<Action>(id, 'actions', semantic.time_bounds, meta.revision, signal),
    collection<Phase>(id, 'phases', semantic.time_bounds, meta.revision, signal),
    collection<Keyframe>(id, 'keyframes', semantic.time_bounds, meta.revision, signal),
    collection<StanceState>(id, 'stances', semantic.time_bounds, meta.revision, signal),
  ])
  return { ...data, steps, actions, phases, keyframes, stances }
}
export async function editTimeline(id: string, base: EditBase, command: 'apply' | 'undo' | 'reset', operations: EditOperation[], author: string, reason: string, signal: AbortSignal): Promise<void> {
  const response = await fetch(`${API_ROOT}${path(id)}/parser-edits`, {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json', 'x-tkd-local-request': '1' },
    body: JSON.stringify({ ...base, command, operations, author, reason, source: 'web-timeline' }),
  })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(`${response.status === 409 ? 'Stale revision / conflict' : 'Edit rejected'} (${response.status}): ${JSON.stringify(body?.detail ?? 'local service error')}. Draft retained.`)
  }
}
