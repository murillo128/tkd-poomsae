import { expect, test, type Page } from '@playwright/test'

async function open(page: Page, project: string, seconds: number) {
  await page.goto('/')
  await page.getByRole('combobox', { name: 'Project', exact: true }).selectOption(project)
  const seek = page.locator('.seek input')
  await seek.fill(String(seconds)); await seek.press('Enter')
}

test.beforeEach(async ({ page }) => {
  await page.route('**/*', async route => {
    const url = new URL(route.request().url())
    expect(['127.0.0.1', 'localhost']).toContain(url.hostname)
    await route.continue()
  })
})

for (let form = 1; form <= 8; form++) {
  test(`real MMPose form ${form} reopens with visible native observations`, async ({ page, request }, testInfo) => {
    const project = `mendeley-bjy7vr4xkt-v1-taegeuk-${form}`
    const root = `http://127.0.0.1:18550/api/projects/${project}/inspection`
    const meta = await (await request.get(root)).json()
    expect(meta.products.reconstruction?.available ?? false).toBe(false)
    const mapping = await request.get(`${root}/time/mendeley-frontal?seconds=6`)
    expect(mapping.ok()).toBe(true)
    const frontal = await mapping.json()
    // Source 6 s mapped through the recorded automatic offset, never a made-up pose.
    const time = 6 + frontal.effective_offset_seconds
    const nativeResponse = await request.get(`${root}/observations/window?collection=observations&start=${time - 1e-8}&end=${time + 1e-8}&limit=256`)
    expect(nativeResponse.ok(), await nativeResponse.text()).toBe(true)
    const native = await nativeResponse.json()
    expect(native.available).toBe(true)
    expect(native.rows.some((row: { provenance: { producer: string }; frame: { camera_id: string } }) =>
      row.frame.camera_id === 'mendeley-frontal' && row.provenance.producer === 'pose.observation_run')).toBe(true)
    await open(page, project, time)
    await page.getByRole('combobox', { name: 'Camera', exact: true }).selectOption('mendeley-frontal')
    const card = page.getByRole('article', { name: 'Camera mendeley-frontal', exact: true })
    await expect(card.locator('img')).toBeVisible()
    await expect.poll(() => card.locator('circle').count()).toBeGreaterThan(0)
    await expect(card).toContainText('raw score')
    await card.getByRole('button', { name: /mendeley-frontal .*wrist/ }).first().focus()
    await page.keyboard.press('Enter')
    await expect(page.locator('.inspector')).toContainText('pixels (px)')
    await page.screenshot({ path: testInfo.outputPath(`real-form-${form}.png`), fullPage: true })
    await page.reload()
    await open(page, project, time)
    await expect(page.locator('.camera-card')).toHaveCount(2)
    await expect(page.locator('.three-d')).toContainText(/unavailable/i)
  })
}

test('synthetic production products reopen with positive 3D, ground and parser capability inspection', async ({ page }, testInfo) => {
  await open(page, 'issue-50-synthetic', 1.5)
  await expect(page.getByLabel('3D world viewport')).toBeVisible()
  const floor = page.getByRole('region', { name: 'Ground view', exact: true })
  await expect(floor).toContainText('Native sample at 1.500 s')
  await expect(floor.locator('.current-root')).toHaveCount(1)
  const timeline = page.getByRole('region', { name: 'Semantic timeline' })
  await expect(timeline).toContainText('Original automatic output')
  await page.getByLabel('Pick 3D landmark').selectOption('left_wrist')
  await expect(page.locator('.inspector')).toContainText('left_wrist')
  await page.screenshot({ path: testInfo.outputPath('synthetic-production.png'), fullPage: true })
  await page.reload()
  await open(page, 'issue-50-synthetic', 1.5)
  await expect(floor.locator('.current-root')).toHaveCount(1)
})
