import { afterEach, expect, it, vi } from 'vitest'
import { readGroundSummary, type GroundMetadata } from './groundApi'
afterEach(() => vi.unstubAllGlobals())
const meta: GroundMetadata = { world_unit: 'arbitrary', participant_id: 'person', scene_bounds: null, start_seconds: 0, end_seconds: 31 }
it('pages bounded windows on one revision and removes interval overlap duplicates', async () => {
  const calls: URL[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = new URL(input); calls.push(url)
    const collection = url.searchParams.get('collection')
    const cursor = Number(url.searchParams.get('cursor'))
    const start = Number(url.searchParams.get('start'))
    const rows = collection === 'ground_frames' ? [{ id: start === 0 && cursor === 0 ? 'first' : 'boundary' }] : []
    return { ok: true, json: async () => ({ available: true, revision: 'r1', rows, next_cursor: collection === 'ground_frames' && cursor === 0 && start === 0 ? 256 : null }) }
  }))
  const result = await readGroundSummary('synthetic', meta, 'r1', new AbortController().signal)
  expect(result.frames.map(row => row.id)).toEqual(['first', 'boundary'])
  expect(calls.every(url => Number(url.searchParams.get('end')) - Number(url.searchParams.get('start')) <= 30)).toBe(true)
  expect(calls.every(url => url.searchParams.get('expected_revision') === 'r1')).toBe(true)
  expect(calls.some(url => url.searchParams.get('cursor') === '256')).toBe(true)
})
it('rejects a mixed revision rather than combining stale geometry', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ available: true, revision: 'changed', rows: [], next_cursor: null }) })))
  await expect(readGroundSummary('synthetic', meta, 'r1', new AbortController().signal)).rejects.toThrow('Ground revision changed')
})
