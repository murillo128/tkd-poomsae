// @vitest-environment jsdom
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, expect, it, vi } from 'vitest'
import { useGroundSnapshot } from './useGroundSnapshot'
import { readGroundSnapshot, type SnapshotResponse } from './groundApi'
vi.mock('./groundApi', () => ({ readGroundSnapshot: vi.fn() }))
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
let root: Root | null = null
let host: HTMLDivElement | null = null
function Harness({ id, revision, seconds }: { id: string; revision: string; seconds: number }) {
  const { current, error } = useGroundSnapshot(id, revision, seconds)
  return <p>{error || `${current?.projectId}/${current?.revision}/${current?.seconds ?? 'loading'}`}</p>
}
async function render(id: string, revision: string, seconds: number) {
  if (!host) { host = document.createElement('div'); document.body.append(host); root = createRoot(host) }
  await act(async () => { root!.render(<Harness id={id} revision={revision} seconds={seconds} />) })
}
afterEach(async () => { if (root) await act(async () => root!.unmount()); host?.remove(); host = null; root = null; vi.resetAllMocks() })
function deferred() {
  let resolve!: (value: SnapshotResponse) => void
  const promise = new Promise<SnapshotResponse>(answer => { resolve = answer })
  return { promise, resolve }
}
const answer = (revision: string): SnapshotResponse => ({ revision, available: false, reason: 'outside_execution', ground_view: null, snapshot: null })
it('lets a slow response finish and coalesces intervening cursor ticks', async () => {
  const first = deferred(), next = deferred()
  vi.mocked(readGroundSnapshot).mockReturnValueOnce(first.promise).mockReturnValueOnce(next.promise)
  await render('a', 'r1', 0)
  const signal = vi.mocked(readGroundSnapshot).mock.calls[0][2]
  await render('a', 'r1', .05); await render('a', 'r1', .1); await render('a', 'r1', .15)
  expect(readGroundSnapshot).toHaveBeenCalledTimes(1)
  expect(signal.aborted).toBe(false)
  await act(async () => { first.resolve(answer('r1')) })
  expect(host!.textContent).toBe('a/r1/0')
  expect(readGroundSnapshot).toHaveBeenCalledTimes(2)
  expect(vi.mocked(readGroundSnapshot).mock.calls[1][1]).toBe(.15)
  await act(async () => { next.resolve(answer('r1')) })
  expect(host!.textContent).toBe('a/r1/0.15')
})
it('aborts old project/revision requests and ignores their late responses', async () => {
  const first = deferred(), second = deferred(), third = deferred()
  vi.mocked(readGroundSnapshot).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise).mockReturnValueOnce(third.promise)
  await render('a', 'r1', 0)
  const oldSignal = vi.mocked(readGroundSnapshot).mock.calls[0][2]
  await render('b', 'r1', .4)
  expect(oldSignal.aborted).toBe(true)
  await act(async () => { first.resolve(answer('r1')) })
  expect(host!.textContent).not.toBe('a/r1/0')
  const previousRevision = vi.mocked(readGroundSnapshot).mock.calls[1][2]
  await render('b', 'r2', .4)
  expect(previousRevision.aborted).toBe(true)
  await act(async () => { second.resolve(answer('r1')); third.resolve(answer('r2')) })
  expect(host!.textContent).toBe('b/r2/0.4')
})
it('rejects responses from a different revision', async () => {
  vi.mocked(readGroundSnapshot).mockResolvedValue(answer('changed'))
  await render('a', 'r1', 0)
  expect(host!.textContent).toContain('Ground revision changed')
})
