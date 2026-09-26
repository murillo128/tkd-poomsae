import { expect, test, type Page } from '@playwright/test'
import { fixture } from './geometryFixture'

async function service(page: Page, geometry = fixture) {
  await page.addInitScript(() => {
    const getContext = HTMLCanvasElement.prototype.getContext
    let lost = 0
    Object.defineProperty(window, 'lost3DContexts', { get: () => lost })
    HTMLCanvasElement.prototype.getContext = function (...args: Parameters<typeof getContext>) {
      const context = getContext.apply(this, args)
      if (args[0] === 'webgl2') this.addEventListener('webglcontextlost', () => lost++)
      return context
    } as typeof getContext
  })
  const stages = Object.fromEntries(['ingest','sync','calibration','observations','attachment','reconstruction','ground','parsing'].map(name => [name, {
    available: true, reason: null, schema_version: '1.0.0', software_revision: 'fixture', model_revision: null, status: { status: 'complete', key: name },
  }]))
  await page.route('http://127.0.0.1:8000/**', async route => {
    const url = new URL(route.request().url()), path = url.pathname
    const project = path.includes('/other') ? 'other' : 'synthetic'
    let body: unknown
    if (path === '/api/projects') body = { projects: ['synthetic','other'] }
    else if (path.endsWith('/capabilities')) body = { project, stages }
    else if (path.endsWith('/media/front/metadata')) body = { source_id: 'front', first_frame: { oriented_width_px: 640, oriented_height_px: 480 } }
    else if (path.endsWith('/inspection')) body = { revision: 'r1', products: { semantics: { available: true } } }
    else if (path.endsWith('/inspection/calibration')) body = { revision: 'r1', available: true, calibration: geometry.calibration, reason: null }
    else if (path.includes('/window')) {
      const start = Number(url.searchParams.get('start')), end = Number(url.searchParams.get('end'))
      const rows = path.includes('/semantics/') ? geometry.actions : geometry.samples.filter(row => row.global_seconds >= start && row.global_seconds <= end)
      body = { revision: 'r1', available: true, unit: 'm', reason: null, rows, next_cursor: null }
      expect(end - start).toBeLessThanOrEqual(2); expect(url.searchParams.get('limit')).toBe('256'); expect(url.searchParams.get('expected_revision')).toBe('r1')
    } else if (path.endsWith('/entities')) body = entityResponse(url.searchParams.get('id')!, geometry)
    else body = { id: project, sources: ['front'], state: { stages: Object.fromEntries(Object.entries(stages).map(([name, stage]) => [name, stage.status])) } }
    await route.fulfill({ json: body, headers: { 'access-control-allow-origin': '*' } })
  })
  await page.goto('/')
  await page.locator('.project-actions select').selectOption('synthetic')
  await expect(page.getByLabel('3D native sample')).toHaveValue('motion/samples/0')
}
function entityResponse(id: string, geometry = fixture) {
  const sample = geometry.samples.find(row => id.startsWith(`${row.id}/`))
  const point = sample?.landmarks.find(value => id.endsWith(`/${value.name}`))
  return {
    entity: sample && point ? { ...point, id, global_seconds: sample.global_seconds } : { id },
    source_evidence: sample ? [{ observation_id: `native-front-${sample.global_seconds}`, frame: {
      camera_id: 'front', pts: sample.global_seconds * 1000, global_seconds: sample.global_seconds,
    } }] : [],
    source_evidence_reason: null,
  }
}
test('actual WebGL panel: native topology, picking, layers, clock, uncertainty, and project disposal', async ({ page }, testInfo) => {
  await service(page)
  const panel = page.locator('.three-d'), canvas = panel.locator('canvas')
  await expect(canvas).toBeVisible()
  const topology = await page.evaluate(async data => {
    const geometry = await import('/src/sceneGeometry.ts')
    const group = geometry.buildScene(data, 0, null, geometry.defaultLayers)
    const result = {
      hands: group.children.filter(object => object.userData.layer === 'Hands' && object.type === 'Line').length,
      feet: group.children.filter(object => object.userData.layer === 'Feet' && object.type === 'Line').length,
      center: geometry.calibratedFrustum(data.calibration.cameras[0], [640,480]).center.toArray(),
      xyz: geometry.worldTuple(geometry.worldVector([1,2,3])),
      cross: geometry.worldVector([1,0,0]).cross(geometry.worldVector([0,1,0])).toArray(),
    }
    geometry.disposeGroup(group); return result
  }, fixture)
  expect(topology).toEqual({ hands: 39, feet: 10, center: [0,-2,1], xyz: [1,2,3], cross: [0,0,1] })
  const before = await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())
  await panel.getByLabel('Hands', { exact: true }).uncheck()
  const hidden = await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())
  expect(hidden).not.toBe(before)
  await panel.getByLabel('Hands', { exact: true }).check()
  for (const layer of ['Body', 'Feet', 'Head orientation', 'Ground', 'Root trajectory', 'Cameras']) {
    await panel.getByLabel(layer, { exact: true }).uncheck()
    expect(await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())).not.toBe(before)
    await panel.getByLabel(layer, { exact: true }).check()
  }
  await panel.getByLabel('Pick 3D landmark').selectOption('left_index_tip')
  await expect(page.locator('.inspector')).toContainText('motion/samples/0/left_index_tip')
  await expect(page.locator('.track-list button').filter({ hasText: 'left arm' })).toHaveAttribute('aria-pressed', 'true')
  const selectedPath = await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())
  await panel.getByLabel('Selected trajectories', { exact: true }).uncheck()
  expect(await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())).not.toBe(selectedPath)
  await panel.getByLabel('Selected trajectories', { exact: true }).check()
  await panel.getByLabel('3D native sample').selectOption('motion/samples/1')
  await expect(page.locator('.timeline')).toContainText('0.500 s')
  await page.locator('.track-list button').filter({ hasText: 'right leg' }).click()
  await expect(page.locator('.inspector')).toContainText('right_leg')
  await panel.locator('summary').click()
  await expect(panel).toContainText('right_pinky_tip: unavailable coordinates')
  await panel.screenshot({ path: testInfo.outputPath('rendered-panel.png') })
  await testInfo.attach('Rendered synthetic panel', { path: testInfo.outputPath('rendered-panel.png'), contentType: 'image/png' })
  // Ray picking at the projected canonical head position in the unchanged initial view.
  await panel.getByRole('button', { name: 'Reset view' }).click()
  const point = await page.evaluate(async () => {
    const THREE = await import('/node_modules/three/build/three.module.js')
    const canvas = document.querySelector('.three-d canvas')!, box = canvas.getBoundingClientRect()
    const camera = new THREE.PerspectiveCamera(45, box.width / box.height, .01, 1000)
    camera.up.set(0,0,1); camera.position.set(3,-4,2.8); camera.lookAt(0,0,.9); camera.updateMatrixWorld()
    const point = new THREE.Vector3(.1,0,1.68).project(camera)
    return { x: box.left+(point.x+1)*box.width/2, y: box.top+(1-point.y)*box.height/2 }
  })
  await page.mouse.click(point.x, point.y)
  await expect(page.locator('.inspector')).toContainText('motion/samples/1/')
  // Orbit, pan and zoom each alter actual rendered pixels.
  const box = (await canvas.boundingBox())!
  const initial = await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())
  await page.mouse.move(box.x+box.width/2, box.y+box.height/2); await page.mouse.down(); await page.mouse.move(box.x+box.width/2+50, box.y+box.height/2+20); await page.mouse.up()
  expect(await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())).not.toBe(initial)
  const orbit = await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())
  await page.mouse.move(box.x+box.width/2, box.y+box.height/2); await page.mouse.down({ button: 'right' }); await page.mouse.move(box.x+box.width/2+25, box.y+box.height/2); await page.mouse.up({ button: 'right' })
  expect(await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())).not.toBe(orbit)
  const pan = await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())
  await page.mouse.wheel(0, 150)
  await expect.poll(() => canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())).not.toBe(pan)
  await page.locator('.project-actions select').selectOption('other')
  await expect(page.getByLabel('3D native sample')).toHaveValue('motion/samples/0')
  await expect.poll(() => page.evaluate(() => (window as unknown as { lost3DContexts: number }).lost3DContexts)).toBeGreaterThan(0)
  await expect(canvas).toHaveCount(1)
  const input = page.locator('.seek input'); await input.fill('0.25'); await input.blur()
  await expect(panel).toContainText('No native sample at this cursor')
})

test('late geometry after a seek or project change never replaces the current window', async ({ page }) => {
  await service(page)
  let release!: () => void, started!: () => void, finished!: () => void
  const pending = new Promise<void>(resolve => { release = resolve })
  const requested = new Promise<void>(resolve => { started = resolve })
  const delivered = new Promise<void>(resolve => { finished = resolve })
  await page.route('**/synthetic/inspection/reconstruction/window?**', async route => {
    if (new URL(route.request().url()).searchParams.get('start') !== '1') return route.fallback()
    started(); await pending
    try { await route.fulfill({ json: { revision: 'r1', available: true, unit: 'm', reason: null, rows: fixture.samples, next_cursor: null }, headers: { 'access-control-allow-origin': '*' } }) }
    catch { /* The browser may have already cancelled the transport. */ }
    finally { finished() }
  })
  const input = page.locator('.seek input')
  await input.fill('2'); await input.blur(); await requested
  await input.fill('4'); await input.blur()
  await expect(page.locator('.three-d')).toContainText('No native sample at this cursor')
  await expect(page.getByLabel('3D native sample').locator('option')).toHaveCount(1)
  await page.locator('.project-actions select').selectOption('other')
  await expect(page.getByLabel('3D native sample')).toHaveValue('motion/samples/0')
  release(); await delivered
  await expect(page.locator('.project-summary')).toContainText('other')
  await expect(page.locator('.timeline')).toContainText('0.000 s')
  await expect(page.getByLabel('3D native sample')).toHaveValue('motion/samples/0')
})

test('selected landmark values and contributing evidence follow native seeks and clear at unavailable instants', async ({ page }) => {
  const geometry = structuredClone(fixture)
  geometry.samples[1].landmarks.find(point => point.name === 'left_index_tip')!.quality = { state: 'interpolated', score: .35, source_ids: ['native-front-0.5'] }
  geometry.samples[2].landmarks.find(point => point.name === 'left_index_tip')!.xyz_world = null
  geometry.samples.push({ ...structuredClone(geometry.samples[1]), id: 'motion/samples/3', global_seconds: 1.5, missing_mask: { left_index_tip: true } })
  await service(page, geometry)
  const inspector = page.locator('.inspector'), panel = page.locator('.three-d')
  const entity = () => inspector.locator('pre').first().evaluate(element => JSON.parse(element.textContent!))
  const evidence = () => inspector.locator('details pre').evaluate(element => JSON.parse(element.textContent!))
  const requests: string[] = []
  page.on('request', request => {
    if (request.url().includes('/inspection/entities?')) requests.push(new URL(request.url()).searchParams.get('id')!)
  })
  await panel.getByLabel('Pick 3D landmark').selectOption('left_index_tip')
  await expect(inspector).toContainText('motion/samples/0/left_index_tip')
  expect((await entity()).global_seconds).toBe(0)
  expect((await evidence())[0].frame.pts).toBe(0)
  const initialPosition = (await entity()).xyz_world
  await panel.getByLabel('3D native sample').selectOption('motion/samples/1')
  await expect(inspector).toContainText('motion/samples/1/left_index_tip')
  expect((await entity()).global_seconds).toBe(.5)
  expect((await entity()).quality).toMatchObject({ state: 'interpolated', score: .35 })
  await expect(inspector).not.toContainText('motion/samples/0/left_index_tip')
  expect((await entity()).xyz_world).toEqual(geometry.samples[1].landmarks.find(point => point.name === 'left_index_tip')!.xyz_world)
  expect((await entity()).xyz_world[0]).toBeCloseTo(initialPosition[0] + .1)
  expect((await evidence())[0]).toMatchObject({ observation_id: 'native-front-0.5', frame: { pts: 500, global_seconds: .5 } })
  await expect(page.locator('.track-list button').filter({ hasText: 'left arm' })).toHaveAttribute('aria-pressed', 'true')
  expect(requests).toEqual(['motion/samples/0/left_index_tip', 'motion/samples/1/left_index_tip'])

  const cursor = page.locator('.seek input')
  await cursor.fill('0.25'); await cursor.blur()
  await expect(inspector).toContainText('No native sample at 0.250 s')
  await expect(inspector.locator('pre')).toHaveCount(0)
  await panel.getByLabel('3D native sample').selectOption('motion/samples/2')
  await expect(inspector).toContainText('left_index_tip unavailable at 1.000 s: landmark coordinates are missing or masked')
  await expect(inspector.locator('pre')).toHaveCount(0)
  await panel.getByLabel('3D native sample').selectOption('motion/samples/3')
  await expect(inspector).toContainText('left_index_tip unavailable at 1.500 s: landmark coordinates are missing or masked')
  await expect(inspector.locator('pre')).toHaveCount(0)
  expect(requests).toEqual(['motion/samples/0/left_index_tip', 'motion/samples/1/left_index_tip'])
  await panel.getByLabel('3D native sample').selectOption('motion/samples/0')
  await expect(inspector).toContainText('motion/samples/0/left_index_tip')
  expect((await entity()).global_seconds).toBe(0)
  expect((await evidence())[0].frame.pts).toBe(0)
})

test('late landmark inspector responses cannot restore values or evidence after a seek or project change', async ({ page }) => {
  await service(page)
  const inspector = page.locator('.inspector'), panel = page.locator('.three-d')
  await panel.getByLabel('Pick 3D landmark').selectOption('left_index_tip')
  await expect(inspector).toContainText('motion/samples/0/left_index_tip')
  for (const changeProject of [false, true]) {
    let release!: () => void, started!: () => void, finished!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    const requested = new Promise<void>(resolve => { started = resolve })
    const delivered = new Promise<void>(resolve => { finished = resolve })
    await page.route('**/synthetic/inspection/entities?**', async route => {
      const id = new URL(route.request().url()).searchParams.get('id')!
      if (id !== 'motion/samples/1/left_index_tip') return route.fallback()
      started(); await pending
      try { await route.fulfill({ json: entityResponse(id), headers: { 'access-control-allow-origin': '*' } }) }
      catch { /* The browser may already have cancelled the transport. */ }
      finally { finished() }
    })
    await panel.getByLabel('3D native sample').selectOption('motion/samples/1'); await requested
    await expect(inspector.locator('pre')).toHaveCount(0)
    if (changeProject) {
      await page.locator('.project-actions select').selectOption('other')
      await expect(page.getByLabel('3D native sample')).toHaveValue('motion/samples/0')
    } else {
      const cursor = page.locator('.seek input'); await cursor.fill('0.25'); await cursor.blur()
      await expect(inspector).toContainText('No native sample at 0.250 s')
    }
    release(); await delivered
    await expect(inspector.locator('pre')).toHaveCount(0)
    if (!changeProject) {
      await panel.getByLabel('3D native sample').selectOption('motion/samples/0')
      await expect(inspector).toContainText('motion/samples/0/left_index_tip')
    }
  }
})
