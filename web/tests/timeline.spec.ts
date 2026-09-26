import { test, expect, type Page, type APIRequestContext } from '@playwright/test'
import type { Action, Interval, Semantics } from '../../contracts/types'
const fixtureRoot = 'http://127.0.0.1:5197'
// Complete fixture transports before disposing their shared API responses.
test.afterEach(async ({ page }) => { await page.unrouteAll({ behavior: 'wait' }) })
async function open(page: Page, id = 'demo', pagingBounds?: Interval) {
  // Only redirect transport; every semantic read/write uses the real service,
  // immutable ArtifactStore, SemanticEditor and optimistic SQLite session.
  await page.route('http://127.0.0.1:8000/**', async route => {
    const url = new URL(route.request().url())
    if (pagingBounds && url.pathname === `/api/projects/${id}/inspection`) {
      // A synthetic long extent isolates paging from parser fixture duration.
      // All window reads still run against the unchanged real service validator.
      const response = await route.fetch({ url: `${fixtureRoot}${url.pathname}${url.search}` })
      const metadata = await response.json()
      metadata.products.semantics.time_bounds = pagingBounds
      await route.fulfill({ response, json: metadata })
    } else await route.continue({ url: `${fixtureRoot}${url.pathname}${url.search}` })
  })
  await page.goto('/')
  await page.getByRole('combobox', { name: 'Project', exact: true }).selectOption(id)
  await expect(page.getByRole('region', { name: 'Semantic timeline' })).not.toContainText('Loading timeline')
}
async function originals(request: APIRequestContext): Promise<{ unchanged: boolean; demo: Semantics; special: Semantics }> {
  return (await request.get(`${fixtureRoot}/fixture/originals`)).json()
}
const panel = (page: Page) => page.getByRole('region', { name: 'Semantic timeline' })
function entity(page: Page, id: string) { return panel(page).locator(`[data-entity-id="${id}"]`).first() }
async function attribution(page: Page) {
  await panel(page).getByLabel('Author', { exact: true }).fill('browser-operator')
  await panel(page).getByLabel('Reason', { exact: true }).fill('Synthetic timing inspection')
}
async function keyframeDraft(page: Page, id: string, seconds: number) {
  await entity(page, id).click()
  await panel(page).getByRole('button', { name: 'Draft timing edit' }).click()
  await panel(page).getByLabel('Keyframe time', { exact: true }).fill(String(seconds))
  await panel(page).getByRole('button', { name: 'Queue timing edit' }).click()
}

test('overlapping physical tracks, independent boundaries, nonuniform events and shared navigation', async ({ page, request }) => {
  const { demo } = await originals(request)
  await open(page)
  const timeline = panel(page)
  const arms = demo.actions.filter(a => a.category === 'arm')
  expect(arms.length).toBeGreaterThanOrEqual(2)
  expect(arms[0].interval).not.toEqual(arms[1].interval)
  expect(arms[0].interval.end).toBeGreaterThan(arms[1].interval.start)
  for (const action of demo.actions) {
    for (const track of action.tracks) {
      const block = timeline.locator(`[data-track="${track}"] [data-entity-id="${action.id}"]`)
      await expect(block).toHaveAttribute('data-start', String(action.interval.start))
      await expect(block).toHaveAttribute('data-end', String(action.interval.end))
    }
  }
  expect(demo.actions.some(a => a.tracks.includes('left_leg'))).toBe(true)
  expect(demo.actions.some(a => a.tracks.includes('right_leg'))).toBe(true)
  expect(demo.actions.some(a => a.role === 'unknown' || a.category === 'transition')).toBe(true)
  await expect(timeline.locator('[data-track]')).toHaveCount(6)
  await expect(timeline).toContainText('Dense motion: available')
  const times = demo.keyframes.map(k => k.global_seconds).sort((a, b) => a - b)
  expect(new Set(times.slice(1).map((time, i) => (time - times[i]).toFixed(3))).size).toBeGreaterThan(1)
  for (const k of demo.keyframes) await expect(entity(page, k.id)).toHaveAttribute('data-start', String(k.global_seconds))
  const action = arms[0]
  await entity(page, action.id).click()
  await expect(page.locator('.seek input')).toHaveValue(action.interval.start.toFixed(3))
  await expect(page.locator('.inspector')).toContainText(action.id)
  await expect(page.locator('.three-d')).toContainText(action.id)
  for (const track of action.tracks) await expect(timeline.locator(`[data-track="${track}"]`)).toHaveAttribute('data-highlighted', 'true')
  await timeline.focus(); await page.keyboard.press('a')
  const sorted = demo.actions.toSorted((a, b) => a.interval.start - b.interval.start || a.id.localeCompare(b.id))
  const next = sorted[sorted.findIndex(a => a.id === action.id) + 1]
  await expect(page.locator('.inspector')).toContainText(next.id)
  await page.keyboard.press('Shift+A')
  await expect(page.locator('.inspector')).toContainText(action.id)
  await entity(page, demo.keyframes[0].id).click()
  await timeline.focus(); await page.keyboard.press('k')
  const orderedKeyframes = demo.keyframes.toSorted((a, b) => a.global_seconds - b.global_seconds || a.id.localeCompare(b.id))
  const nextKeyframe = orderedKeyframes[orderedKeyframes.findIndex(k => k.id === demo.keyframes[0].id) + 1]
  await expect(page.locator('.inspector')).toContainText(nextKeyframe.id)
  await expect(page.locator('.seek input')).toHaveValue(nextKeyframe.global_seconds.toFixed(3))
  await timeline.getByRole('button', { name: 'Next step', exact: true }).click()
  await timeline.getByLabel('Zoom', { exact: true }).fill('4')
  await timeline.getByLabel('Pan', { exact: true }).fill('0.5')
  await expect(timeline.locator('.timeline-ruler')).not.toContainText(`${demo.execution!.start.toFixed(3)} s`)
  await timeline.screenshot({ path: test.info().outputPath('timeline.png') })
})

test('a SpecialAction stays on both participating tracks with one stable selection', async ({ page, request }) => {
  const { special } = await originals(request)
  await open(page, 'special')
  const action = special.actions.find(a => a.category === 'special') as Action
  expect(action.tracks).toHaveLength(2)
  await expect(panel(page).locator(`[data-entity-id="${action.id}"]`)).toHaveCount(2)
  await entity(page, action.id).click()
  for (const track of action.tracks) await expect(panel(page).locator(`[data-track="${track}"]`)).toHaveAttribute('data-highlighted', 'true')
  await expect(page.locator('.inspector')).toContainText(action.id)
})

test('real edit, reload, undo and reset preserve every original automatic artifact byte', async ({ page, request }) => {
  const { demo } = await originals(request)
  await open(page); await attribution(page)
  const k = demo.keyframes.find(k => {
    const a = demo.actions.find(a => a.id === k.action_id)!
    return k.global_seconds + .001 < a.interval.end && !k.phase_id
  }) ?? demo.keyframes[0]
  // Move very slightly inside its owning phase/action; keyframes are not assumed uniform.
  const owner = demo.phases.find(p => p.id === k.phase_id)?.interval ?? demo.actions.find(a => a.id === k.action_id)!.interval
  const moved = k.global_seconds < owner.end ? Math.min(owner.end, k.global_seconds + .001) : k.global_seconds - .001
  await keyframeDraft(page, k.id, moved)
  await expect(entity(page, k.id)).toHaveAttribute('data-start', String(k.global_seconds))
  await panel(page).getByRole('button', { name: 'Apply edits', exact: true }).click()
  await expect(panel(page)).toContainText('Manual edits over preserved automatic artifact')
  await expect(entity(page, k.id)).toHaveAttribute('data-start', String(moved))
  await expect(page.locator('.seek input')).toHaveValue(k.global_seconds.toFixed(3))
  await expect(page.locator('.inspector')).toContainText('Origin manual')
  const inspected = await page.locator('.inspector .geometry-values').evaluate(element => JSON.parse(element.textContent!))
  expect(inspected.global_seconds).toBe(moved)
  await page.reload(); await page.getByRole('combobox', { name: 'Project', exact: true }).selectOption('demo')
  await expect(entity(page, k.id)).toHaveAttribute('data-start', String(moved))
  await attribution(page)
  await panel(page).getByRole('button', { name: 'Undo edit', exact: true }).click()
  await expect(entity(page, k.id)).toHaveAttribute('data-start', String(k.global_seconds))
  await expect(panel(page)).toContainText('Original automatic output')
  await keyframeDraft(page, k.id, moved)
  await panel(page).getByRole('button', { name: 'Apply edits', exact: true }).click()
  await expect(entity(page, k.id)).toHaveAttribute('data-start', String(moved))
  await panel(page).getByRole('button', { name: 'Reset to automatic', exact: true }).click()
  await expect(entity(page, k.id)).toHaveAttribute('data-start', String(k.global_seconds))

  const action = demo.actions.find(a => a.category === 'arm')!
  await entity(page, action.id).click()
  await panel(page).getByRole('button', { name: 'Draft timing edit' }).click()
  await panel(page).getByLabel('Boundary start', { exact: true }).fill(String(action.interval.start - .001))
  await panel(page).getByRole('button', { name: 'Queue timing edit' }).click()
  await panel(page).getByRole('button', { name: 'Apply edits', exact: true }).click()
  await expect(entity(page, action.id)).toHaveAttribute('data-start', String(action.interval.start - .001))
  for (const other of demo.actions.filter(a => a.id !== action.id)) {
    await expect(entity(page, other.id)).toHaveAttribute('data-start', String(other.interval.start))
    await expect(entity(page, other.id)).toHaveAttribute('data-end', String(other.interval.end))
  }
  await panel(page).getByRole('button', { name: 'Undo edit', exact: true }).click()
  await expect(entity(page, action.id)).toHaveAttribute('data-start', String(action.interval.start))
  expect((await originals(request)).unchanged).toBe(true)
})

test('invalid intervals and stale revisions preserve queued user intent; cancel is explicit', async ({ page, request }) => {
  const { special } = await originals(request)
  await open(page, 'special'); await attribution(page)
  const action = special.actions.find(a => a.category === 'special')!
  await entity(page, action.id).click()
  await panel(page).getByRole('button', { name: 'Draft timing edit' }).click()
  await panel(page).getByLabel('Boundary start', { exact: true }).fill(String(action.interval.start + .001))
  await panel(page).getByRole('button', { name: 'Queue timing edit' }).click()
  const intent = await panel(page).getByLabel('Queued edits').textContent()
  const meta = await (await request.get(`${fixtureRoot}/api/projects/special/inspection`)).json()
  const response = await request.post(`${fixtureRoot}/api/projects/special/inspection/parser-edits`, {
    headers: { 'x-tkd-local-request': '1' }, data: { expected_revision: meta.products.semantics.effective_edit_revision, automatic_revision: meta.products.semantics.artifact_revision, command: 'reset', operations: [], source: 'browser-concurrent-writer', author: 'other', reason: 'stale revision acceptance' },
  })
  expect(response.ok()).toBe(true)
  await panel(page).getByRole('button', { name: 'Apply edits', exact: true }).click()
  await expect(panel(page).getByRole('alert')).toContainText('Stale revision / conflict')
  await expect(panel(page).getByLabel('Queued edits')).toHaveText(intent!)
  await panel(page).getByRole('button', { name: 'Reload timeline' }).click()
  await expect(panel(page).getByLabel('Queued edits')).toHaveText(intent!)
  await panel(page).getByRole('button', { name: 'Apply edits', exact: true }).click()
  await expect(panel(page).getByRole('alert')).toContainText('Stale revision / conflict')
  await panel(page).getByRole('button', { name: 'Cancel drafts' }).click()
  await expect(panel(page).getByLabel('Queued edits')).toHaveCount(0)
  await entity(page, action.id).click()
  await panel(page).getByRole('button', { name: 'Draft timing edit' }).click()
  await panel(page).getByLabel('Boundary end', { exact: true }).fill(String(action.interval.start - 1))
  await panel(page).getByRole('button', { name: 'Queue timing edit' }).click()
  await expect(panel(page).getByRole('alert')).toContainText('end must follow start')
  await expect(panel(page).getByLabel('Boundary end', { exact: true })).toHaveValue(String(action.interval.start - 1))
  await panel(page).getByLabel('Boundary end', { exact: true }).fill(String(action.interval.start + .001))
  await panel(page).getByRole('button', { name: 'Queue timing edit' }).click()
  await panel(page).getByRole('button', { name: 'Apply edits', exact: true }).click()
  await expect(panel(page).getByRole('alert')).toContainText('Edit rejected (422)')
  await expect(panel(page).getByLabel('Queued edits')).toContainText(String(action.interval.start + .001))
  expect((await originals(request)).unchanged).toBe(true)
})

test('unavailable parsing leaves raw track selection and time navigation usable', async ({ page }) => {
  await open(page, 'raw')
  await expect(panel(page)).toContainText('Parsing unavailable')
  await expect(panel(page)).toContainText('Dense motion: available')
  await panel(page).getByRole('button', { name: 'left arm', exact: true }).click()
  await expect(page.locator('.inspector')).toContainText('left_arm')
  await panel(page).getByLabel('Timeline time', { exact: true }).fill('24.1')
  await expect(page.locator('.seek input')).toHaveValue('24.100')
})

test('fractional long bounds page gap-free within the real API limit', async ({ page, request }) => {
  // Keep the API's strict cap: this nominal 30s span actually subtracts to
  // 30.000000000000004. The client must avoid it rather than relaxing validation.
  const invalid = await request.get(`${fixtureRoot}/api/projects/demo/inspection/semantics/window?collection=steps&start=2.2&end=32.2`)
  expect(invalid.status()).toBe(422)
  const windows: { collection: string; start: number; end: number; status: number }[] = []
  // Timeline paging carries a cursor; the synchronized 3D panel's separate
  // short action windows must not be counted as timeline collection pages.
  page.on('response', response => {
    const url = new URL(response.url())
    if (url.pathname.includes('/demo/inspection/semantics/window') && url.searchParams.has('cursor')) {
      windows.push({ collection: url.searchParams.get('collection')!, start: Number(url.searchParams.get('start')), end: Number(url.searchParams.get('end')), status: response.status() })
    }
  })
  await open(page, 'demo', { start: 2.2, end: 65 })
  await expect.poll(() => windows.length).toBeGreaterThanOrEqual(5)
  await expect(panel(page)).toContainText('Original automatic output')
  await expect(panel(page).getByRole('alert')).toHaveCount(0)
  for (const collection of ['steps', 'actions', 'phases', 'keyframes', 'stances']) {
    const pages = windows.filter(window => window.collection === collection).sort((a, b) => a.start - b.start)
    expect(pages.length).toBeGreaterThan(2)
    expect(pages[0].start).toBe(2.2)
    expect(pages.at(-1)!.end).toBe(65)
    pages.forEach((window, i) => {
      expect(window.status).toBe(200)
      expect(window.end - window.start).toBeLessThanOrEqual(30)
      if (i) expect(window.start).toBe(pages[i - 1].end)
    })
  }
  // Existing canonical entities must still load and remain inspectable.
  const steps = panel(page).getByLabel('SequenceSteps').locator('[data-entity-id]')
  await expect(steps).toHaveCount(1)
  const { demo } = await originals(request)
  await expect(steps).toHaveAttribute('data-start', String(demo.steps[0].interval.start))
  await expect(steps).toHaveAttribute('data-end', String(demo.steps[0].interval.end))
  await steps.click()
  await expect(page.locator('.seek input')).toHaveValue(demo.steps[0].interval.start.toFixed(3))
  await expect(panel(page).getByRole('button', { name: 'Draft timing edit' })).toBeEnabled()
  expect((await originals(request)).unchanged).toBe(true)
})

test('parser keyframes reach persisted physical samples while preserving the shared clock', async ({ page, request }) => {
  const { demo } = await originals(request)
  const keyframe = demo.keyframes.find(k => k.motion_sample_indices.length > 0)!
  expect(keyframe).toBeTruthy()
  await open(page)
  await entity(page, keyframe.id).click()
  const inspector = page.locator('.inspector')
  await inspector.getByText('Contributing physical samples', { exact: true }).click()
  await inspector.getByRole('button', { name: /Inspect physical sample/ }).first().click()
  await expect(inspector.locator('.geometry-values')).toContainText('global_seconds')
  const selected = await inspector.locator('.geometry-values').evaluate(element => JSON.parse(element.textContent!))
  expect(selected.id).toContain('/samples/')
  expect(selected).toHaveProperty('root_xyz_world')
  await expect(page.locator('.seek input')).toHaveValue(keyframe.global_seconds.toFixed(3))
  await expect(page.locator('.timeline')).toContainText(`${keyframe.global_seconds.toFixed(3)} s`)
})
