// @vitest-environment jsdom
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'

// React's act helper expects this flag in a jsdom test environment.
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
let root: Root | null = null
let host: HTMLDivElement | null = null
async function mount() {
  host = document.createElement('div')
  document.body.appendChild(host)
  root = createRoot(host)
  await act(async () => { root!.render(<App />) })
  return host
}
afterEach(async () => {
  if (root) await act(async () => root!.unmount())
  host?.remove()
  root = null
  host = null
  vi.unstubAllGlobals()
})

const stages = Object.fromEntries(['ingest', 'sync', 'calibration', 'observations', 'attachment', 'reconstruction', 'ground', 'parsing'].map(name => [name, {
  available: false, reason: `${name} producer unavailable`, schema_version: '1.0.0', software_revision: '1', model_revision: null,
  status: { status: 'unavailable', diagnostics: [`${name} producer unavailable`] },
}]))
function response(body: unknown, status = 200) { return { ok: status < 400, status, json: async () => body } as Response }
function typeInto(input: HTMLInputElement, value: string) {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value)
  input.dispatchEvent(new Event('input', { bubbles: true }))
}

describe('rendered local inspection shell', () => {
  it('opens a synthetic project and shows explicit missing capability states with shared controls', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/projects')) return response({ projects: ['synthetic'] })
      if (url.endsWith('/capabilities')) return response({ project: 'synthetic', stages })
      return response({ id: 'synthetic', sources: ['front', 'side'], state: { stages: Object.fromEntries(Object.entries(stages).map(([name, stage]) => [name, stage.status])) } })
    }))
    const element = await mount()
    const select = element.querySelector<HTMLSelectElement>('.project-actions select')!
    await act(async () => { select.value = 'synthetic'; select.dispatchEvent(new Event('change', { bubbles: true })) })
    expect(element.textContent).toContain('2 camera sources')
    expect(element.textContent).toContain('reconstruction producer unavailable')
    expect(element.textContent).toContain('Video frame unavailable')
    expect(element.textContent).toContain('Frame stepping unavailable')
    expect(element.querySelector<HTMLButtonElement>('[aria-label="Next native frame"]')?.disabled).toBe(true)
    const seek = element.querySelector<HTMLInputElement>('.seek input')!
    await act(async () => { seek.focus() })
    for (const value of ['', '0', '0.', '0.1', '0.125']) {
      await act(async () => { typeInto(seek, value) })
      expect(seek.value).toBe(value)
    }
    expect(element.querySelector('.time-readout')?.textContent).toContain('0.000 s')
    await act(async () => { seek.blur() })
    expect(element.querySelector('.time-readout')?.textContent).toContain('0.125 s')
    expect(seek.value).toBe('0.125')
    await act(async () => { seek.focus() })
    await act(async () => { typeInto(seek, '') })
    expect(seek.value).toBe('')
    await act(async () => { seek.blur() })
    expect(element.querySelector('.time-readout')?.textContent).toContain('0.125 s')
    expect(element.querySelector('.seek-error')?.textContent).toContain('non-negative time')
    const track = [...element.querySelectorAll<HTMLButtonElement>('.track-list button')].find(button => button.textContent === 'left leg')!
    await act(async () => { track.click() })
    expect(element.querySelector('.inspector')?.textContent).toContain('left_leg')
  })

  it('reports service failure and retries discovery', async () => {
    let calls = 0
    vi.stubGlobal('fetch', vi.fn(async () => ++calls === 1 ? response({}, 503) : response({ projects: ['synthetic'] })))
    const element = await mount()
    expect(element.querySelector('[role="alert"]')?.textContent).toContain('503')
    await act(async () => { [...element.querySelectorAll('button')].find(button => button.textContent === 'Retry projects')!.click() })
    expect(element.querySelector('select')?.textContent).toContain('synthetic')
  })

  it('reports project failure and retries its metadata request', async () => {
    let detailCalls = 0
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/projects')) return response({ projects: ['synthetic'] })
      if (url.endsWith('/capabilities')) return response({ project: 'synthetic', stages })
      if (++detailCalls === 1) return response({}, 500)
      return response({ id: 'synthetic', sources: ['front'], state: { stages: Object.fromEntries(Object.entries(stages).map(([name, stage]) => [name, stage.status])) } })
    }))
    const element = await mount()
    const select = element.querySelector<HTMLSelectElement>('.project-actions select')!
    await act(async () => { select.value = 'synthetic'; select.dispatchEvent(new Event('change', { bubbles: true })) })
    expect(element.querySelector('[role="alert"]')?.textContent).toContain('500')
    await act(async () => { [...element.querySelectorAll('button')].find(button => button.textContent === 'Retry project')!.click() })
    expect(element.textContent).toContain('1 camera source')
  })

  it('ignores a late response from a previously selected project', async () => {
    let resolveOld!: (response: Response) => void
    const oldDetail = new Promise<Response>(resolve => { resolveOld = resolve })
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/projects')) return response({ projects: ['old', 'new'] })
      if (url.endsWith('/old')) return oldDetail
      if (url.endsWith('/capabilities')) return response({ project: url.includes('/old/') ? 'old' : 'new', stages })
      return response({ id: 'new', sources: ['new-camera'], state: { stages: Object.fromEntries(Object.entries(stages).map(([name, stage]) => [name, stage.status])) } })
    }))
    const element = await mount()
    const select = element.querySelector<HTMLSelectElement>('.project-actions select')!
    await act(async () => { select.value = 'old'; select.dispatchEvent(new Event('change', { bubbles: true })) })
    await act(async () => { select.value = 'new'; select.dispatchEvent(new Event('change', { bubbles: true })) })
    await act(async () => { resolveOld(response({ id: 'old', sources: ['old-camera'], state: { stages: {} } })) })
    expect(element.querySelector('.project-summary h2')?.textContent).toBe('new')
    expect(element.querySelector('.camera-grid')?.textContent).toContain('new-camera')
    expect(element.querySelector('.camera-grid')?.textContent).not.toContain('old-camera')
  })
})
