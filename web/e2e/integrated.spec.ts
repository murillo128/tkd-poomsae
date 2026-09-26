import { expect, test, type Page } from '@playwright/test'

const service = `http://127.0.0.1:${process.env.TKD_BROWSER_SERVICE_PORT ?? '18044'}`
const projectSelect = (page: Page) => page.getByRole('combobox', { name: 'Project', exact: true })
const timeline = (page: Page) => page.getByRole('region', { name: 'Semantic timeline' })
const ground = (page: Page) => page.getByRole('region', { name: 'Ground view', exact: true })
async function open(page: Page, count = 2) {
  await page.goto('/')
  await projectSelect(page).selectOption(`cameras-${count}`)
  await page.getByRole('combobox', { name: 'Camera', exact: true }).selectOption('front')
  await expect(page.getByLabel('3D world viewport')).toBeVisible()
  await expect(timeline(page)).toContainText('Original automatic output')
}
async function seek(page: Page, seconds: number) {
  const input = page.locator('.seek input')
  await input.fill(String(seconds)); await input.press('Enter')
}
async function sharedInstant(page: Page, seconds: number) {
  const label = `${seconds.toFixed(3)} s`
  await expect(page.locator('.seek input')).toHaveValue(seconds.toFixed(3))
  await expect(timeline(page)).toContainText(`Global cursor: ${label}`)
  await expect(page.locator('.three-d')).toContainText(`Cursor ${label} · native sample ${label}`)
  await expect(ground(page)).toContainText(`Native sample at ${label}`)
  await expect(ground(page).locator('.current-root')).toHaveCount(1)
  await expect(page.getByRole('article', { name: 'Camera front', exact: true })).toContainText(`Delivered ordinal ${Math.round(seconds * 25)}`)
}

for (const count of [2, 3, 4]) {
  test(`${count} cameras: positive geometry, events and source evidence follow one cursor`, async ({ page }) => {
    await open(page, count)
    await seek(page, 0.4)
    await sharedInstant(page, 0.4)
    await expect(page.locator('.camera-card')).toHaveCount(count)
    await expect(page.getByRole('article', { name: 'Camera front', exact: true }).locator('circle')).toHaveCount(4)
    const arms = timeline(page).locator('[data-entity-id^="synthetic-arm-"]')
    await expect(arms).toHaveCount(2)
    await expect(arms.nth(0)).toHaveAttribute('data-start', '0.2')
    await expect(arms.nth(0)).toHaveAttribute('data-end', '0.8')
    await expect(arms.nth(1)).toHaveAttribute('data-start', '0.4')
    await expect(arms.nth(1)).toHaveAttribute('data-end', '0.9')
    await arms.first().focus(); await page.keyboard.press('Enter')
    await sharedInstant(page, 0.2)
    await expect(page.locator('.inspector')).toContainText('synthetic-arm-0')
    await expect(timeline(page).locator('[data-track="left_arm"]')).toHaveAttribute('data-highlighted', 'true')
    await timeline(page).focus(); await page.keyboard.press('a')
    await sharedInstant(page, 0.4)
    await expect(page.locator('.inspector')).toContainText('synthetic-arm-1')
    await page.keyboard.press('Shift+A')
    await sharedInstant(page, 0.2)
    await timeline(page).locator('[data-entity-id="synthetic-event"]').click()
    await sharedInstant(page, 0.4)
    const inspector = page.locator('.inspector')
    await inspector.getByText('Contributing physical samples', { exact: true }).click()
    await inspector.getByRole('button', { name: 'Inspect physical sample 0.4 s', exact: true }).click()
    await expect(inspector.locator('.geometry-values')).toContainText('root_xyz_world')
    await page.getByLabel('Pick 3D landmark').selectOption('left_wrist')
    await expect(inspector).toContainText(`synthetic-motion-cameras-${count}/samples/20/left_wrist`)
    await inspector.getByText('Contributing camera evidence', { exact: true }).click()
    await inspector.getByRole('button', { name: /Inspect front PTS/ }).click()
    await expect(inspector.getByRole('img', { name: /Contributing front PTS/ })).toBeVisible()
    await sharedInstant(page, 0.4)
    await ground(page).getByRole('combobox', { name: 'View', exact: true }).selectOption('summary')
    const placement = ground(page).getByRole('button', { name: /left .* placement/ }).first()
    await placement.focus(); await page.keyboard.press('Enter')
    await expect(placement).toHaveAttribute('aria-pressed', 'true')
    await expect(inspector).toContainText('footprint')
    await ground(page).getByRole('combobox', { name: 'View', exact: true }).selectOption('dynamic')
    await seek(page, 0.4)
    await page.getByRole('button', { name: 'Next native frame' }).click()
    await sharedInstant(page, 0.44)
    await page.getByRole('button', { name: 'Previous native frame' }).click()
    await sharedInstant(page, 0.4)
  })
}

test('integrated parser edits persist through reload and undo without changing physical evidence', async ({ page, request }) => {
  await open(page)
  const panel = timeline(page)
  await panel.locator('[data-entity-id="synthetic-event"]').click()
  await panel.getByLabel('Author', { exact: true }).fill('browser-fixture-operator')
  await panel.getByLabel('Reason', { exact: true }).fill('Synthetic event timing check')
  await panel.getByRole('button', { name: 'Draft timing edit' }).click()
  await panel.getByLabel('Keyframe time', { exact: true }).fill('0.44')
  await panel.getByRole('button', { name: 'Queue timing edit' }).click()
  const api = `${service}/api/projects/cameras-2/inspection`
  const before = await (await request.get(`${api}/ground/snapshot?seconds=0.4`)).json()
  await panel.getByRole('button', { name: 'Apply edits', exact: true }).click()
  await expect(panel).toContainText('Manual edits over preserved automatic artifact')
  await expect(panel.locator('[data-entity-id="synthetic-event"]')).toHaveAttribute('data-start', '0.44')
  await page.reload(); await projectSelect(page).selectOption('cameras-2')
  await expect(panel.locator('[data-entity-id="synthetic-event"]')).toHaveAttribute('data-start', '0.44')
  await panel.getByLabel('Author', { exact: true }).fill('browser-fixture-operator')
  await panel.getByLabel('Reason', { exact: true }).fill('Undo timing check')
  await panel.getByRole('button', { name: 'Undo edit', exact: true }).click()
  await expect(panel.locator('[data-entity-id="synthetic-event"]')).toHaveAttribute('data-start', '0.4')
  const after = await (await request.get(`${api}/ground/snapshot?seconds=0.4`)).json()
  expect(after.snapshot).toEqual(before.snapshot)
  await seek(page, 0.4); await sharedInstant(page, 0.4)
})

test('service failures are visible and retry restores real persisted geometry', async ({ page }) => {
  await open(page)
  await page.route('**/reconstruction/window?**', route => route.fulfill({ status: 503, json: { detail: 'synthetic service outage' } }))
  await seek(page, 0.6)
  await expect(page.locator('.three-d')).toContainText('3D unavailable')
  await expect(page.getByLabel('Pick 3D landmark')).toHaveCount(0)
  await page.unroute('**/reconstruction/window?**')
  await page.getByRole('button', { name: 'Retry 3D', exact: true }).click()
  await sharedInstant(page, 0.6)
  await projectSelect(page).selectOption('provenance')
  await seek(page, 0.2)
  await expect(page.locator('.three-d')).toContainText('Ground plane unavailable')
  await expect(page.locator('.three-d')).toContainText('Units: arbitrary')
  await expect(ground(page)).toContainText('Ground view unavailable')
  await expect(page.getByLabel('Pick 3D landmark')).toBeVisible()
})

for (const width of [1024, 1280, 1600]) {
  test(`accessible controls, textual uncertainty and layout at desktop width ${width}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1100 })
    await open(page, 4); await seek(page, 0.4)
    await sharedInstant(page, 0.4)
    await expect(page.getByRole('article', { name: 'Camera slow', exact: true })).toContainText('Low confidence')
    await expect(page.getByRole('article', { name: 'Camera excluded', exact: true })).toContainText('generated excluded view')
    await expect(page.locator('.three-d')).toContainText('purple interpolated')
    await expect(ground(page)).toContainText('uncertainty')
    const unnamed = await page.locator('button, input, select').evaluateAll(elements => elements.filter(element => {
      const control = element as HTMLInputElement
      return !control.getAttribute('aria-label') && !control.getAttribute('aria-labelledby') && !control.textContent?.trim() && !control.labels?.length
    }).map(element => element.outerHTML))
    expect(unnamed).toEqual([])
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    const next = page.getByRole('button', { name: 'Next native frame' })
    await next.focus(); await page.keyboard.press('Enter')
    await expect(next).toBeFocused()
    await sharedInstant(page, 0.44)
    expect(await next.evaluate(element => getComputedStyle(element).outlineStyle)).not.toBe('none')
    if (width === 1280) await page.screenshot({ path: test.info().outputPath('integrated-desktop.png'), fullPage: true })
  })
}

test('late cross-project responses cannot replace current evidence; requests and resources remain bounded', async ({ page }) => {
  await page.addInitScript(() => {
    const blobs = new Set<string>(), intervals = new Set<number>()
    const contexts = new WeakSet<object>()
    let liveContexts = 0
    const getContext = HTMLCanvasElement.prototype.getContext
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, ...args: Parameters<typeof getContext>) {
      const context = getContext.apply(this, args)
      if (args[0] === 'webgl2' && context && !contexts.has(context)) {
        contexts.add(context); liveContexts++
        this.addEventListener('webglcontextlost', () => { liveContexts-- }, { once: true })
      }
      return context
    } as typeof getContext
    const create = URL.createObjectURL.bind(URL), revoke = URL.revokeObjectURL.bind(URL)
    URL.createObjectURL = object => { const url = create(object); blobs.add(url); return url }
    URL.revokeObjectURL = url => { blobs.delete(url); revoke(url) }
    const set = window.setInterval.bind(window), clear = window.clearInterval.bind(window)
    window.setInterval = ((...args: Parameters<typeof set>) => { const id = set(...args); intervals.add(id); return id }) as typeof set
    window.clearInterval = id => { intervals.delete(Number(id)); clear(id) }
    Object.assign(window, { fixtureResources: () => ({ blobs: blobs.size, intervals: intervals.size, webgl: liveContexts }) })
  })
  const windows: URL[] = []
  page.on('request', request => { if (new URL(request.url()).pathname.endsWith('/window')) windows.push(new URL(request.url())) })
  await open(page)
  await seek(page, 0.4); await sharedInstant(page, 0.4)
  const resources = () => page.evaluate(() => (window as unknown as { fixtureResources: () => { blobs: number; intervals: number; webgl: number } }).fixtureResources())
  await expect.poll(async () => (await resources()).blobs).toBe(2)
  const baseline = await resources()
  let release!: () => void, received!: () => void, settled!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  const pending = new Promise<void>(resolve => { received = resolve })
  const complete = new Promise<void>(resolve => { settled = resolve })
  await page.route('**/cameras-2/inspection/reconstruction/window?**', async route => {
    const response = await route.fetch()
    received(); await gate
    await route.fulfill({ response }).catch(() => {})
    settled()
  })
  await seek(page, 0.6); await pending
  await projectSelect(page).selectOption('cameras-3')
  await expect(page.locator('.project-summary h2')).toHaveText('cameras-3')
  await seek(page, 0.4); await sharedInstant(page, 0.4)
  release(); await complete
  await page.unroute('**/cameras-2/inspection/reconstruction/window?**')
  await page.getByLabel('Pick 3D landmark').selectOption('left_wrist')
  await expect(page.locator('.inspector')).toContainText('synthetic-motion-cameras-3/samples/20/left_wrist')
  await expect(page.locator('.inspector')).not.toContainText('synthetic-motion-cameras-2')
  for (let i = 0; i < 6; i++) {
    const id = `cameras-${i % 2 ? 3 : 2}`
    await projectSelect(page).selectOption(id)
    await expect(page.locator('.project-summary h2')).toHaveText(id)
    await seek(page, 0.4); await sharedInstant(page, 0.4)
    await expect(page.locator('.three-viewport canvas')).toHaveCount(1)
    const live = await resources()
    expect(live.blobs).toBeLessThanOrEqual(baseline.blobs + 1)
    expect(live.intervals).toBe(baseline.intervals)
    await expect.poll(async () => (await resources()).webgl).toBe(1)
  }
  const before = windows.length
  // Commit seeks faster than the debounced native geometry batch can complete.
  for (const seconds of [0.2, 0.4, 0.6, 0.8, 0.4]) await seek(page, seconds)
  await sharedInstant(page, 0.4)
  expect(windows.length - before).toBeLessThan(50)
  for (const url of windows) {
    expect(Number(url.searchParams.get('limit'))).toBeLessThanOrEqual(256)
    expect(Number(url.searchParams.get('end')) - Number(url.searchParams.get('start'))).toBeLessThanOrEqual(30)
  }
  const idle = windows.length
  await page.waitForTimeout(400)
  expect(windows.length).toBe(idle)
})

test('real video playback advances the shared views and pause restores exact evidence', async ({ page }) => {
  await open(page)
  await seek(page, 0.4); await sharedInstant(page, 0.4)
  await page.getByRole('combobox', { name: 'Speed', exact: true }).selectOption('0.25')
  await page.getByRole('button', { name: 'Play', exact: true }).click()
  const front = page.getByRole('article', { name: 'Camera front', exact: true })
  await expect(front.locator('video')).toBeVisible()
  await expect.poll(async () => front.locator('video').evaluate(video => (video as HTMLVideoElement).currentTime)).toBeGreaterThan(0.5)
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  const seconds = Number(await page.locator('.seek input').inputValue())
  expect(seconds).toBeGreaterThan(0.4)
  await expect(timeline(page)).toContainText(`Global cursor: ${seconds.toFixed(3)} s`)
  await expect(page.locator('.three-d')).toContainText(`Cursor ${seconds.toFixed(3)} s`)
  // Browser/CI speed must not determine acceptance. The synthetic geometry ends
  // at 1 s; a pause beyond it must report absence instead of inventing a pose.
  if (seconds <= 1) {
    await expect(ground(page)).toContainText(`cursor ${seconds.toFixed(3)} s`)
    await expect(ground(page).locator('.current-root')).toHaveCount(1)
  } else {
    await expect(ground(page)).toContainText('Current geometry unavailable: outside_execution')
    await expect(ground(page).locator('.current-root')).toHaveCount(0)
    await expect(page.locator('.three-d')).toContainText('No native sample at this cursor')
  }
  await seek(page, 0.4); await sharedInstant(page, 0.4)
  await expect(front.locator('img')).toBeVisible()
  await expect(front.locator('circle')).toHaveCount(4)
})
