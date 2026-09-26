import { expect, test, type Page } from '@playwright/test'

async function open(page: Page, count: number) {
  await page.goto('/')
  await page.locator('.project-actions select').selectOption(`cameras-${count}`)
  await expect(page.locator('.camera-card')).toHaveCount(count)
  await page.getByRole('combobox', { name: 'Camera', exact: true }).selectOption('front')
  await expect(page.getByRole('article', { name: 'Camera front', exact: true }).locator('img')).toBeVisible()
}
async function seek(page: Page, seconds: string) {
  const input = page.locator('.seek input')
  await input.fill(seconds)
  await input.press('Enter')
}
for (const count of [2, 3, 4]) {
  test(`${count} generated cameras preserve source identities, offsets, and native stepping`, async ({ page }) => {
    await open(page, count)
    await seek(page, '1')
    const front = page.getByRole('article', { name: 'Camera front', exact: true })
    const rotated = page.getByRole('article', { name: 'Camera rotated', exact: true })
    await expect(front).toContainText('Delivered ordinal 25')
    await expect(front).toContainText('sample mismatch 0.000000 s')
    await expect(rotated).toContainText('offset -0.200000 s')
    await expect(rotated).toContainText('source 1.200000 s')
    await expect(front.locator('circle')).toHaveCount(4)
    await expect(front).toContainText('raw score 7')
    await expect(front).toContainText('derived quality 0.9')
    await page.getByRole('button', { name: 'Next native frame' }).click()
    await expect(front).toContainText('Delivered ordinal 26')
    await expect(page.locator('.time-readout')).toContainText('1.040 s')
    await page.getByRole('button', { name: 'Previous native frame' }).click()
    await expect(front).toContainText('Delivered ordinal 25')
    await front.getByRole('button', { name: 'front left_index_tip', exact: true }).focus()
    await page.keyboard.press('Enter')
    await expect(page.locator('.inspector')).toContainText('left_index_tip')
    if (count >= 3) await expect(page.getByRole('article', { name: 'Camera slow', exact: true })).toContainText('Low confidence')
    if (count === 4) {
      const excluded = page.getByRole('article', { name: 'Camera excluded', exact: true })
      await expect(excluded).toContainText('generated excluded view')
      await expect(excluded.locator('img, circle')).toHaveCount(0)
    }
    await seek(page, '10')
    await expect(front).toContainText('outside camera coverage')
    await expect(front.locator('img, circle')).toHaveCount(0)
    await expect(page.locator('.time-readout')).toContainText('Unavailable')
  })
}

test('oriented image pixels, crop provenance and overlays share letterboxing at every size', async ({ page }) => {
  await open(page, 2)
  await seek(page, '1')
  const panel = page.getByRole('article', { name: 'Camera rotated', exact: true })
  await expect(panel.locator('circle')).toHaveCount(4)
  for (const width of [1280, 640, 1600]) {
    await page.setViewportSize({ width, height: 900 })
    const geometry = await panel.evaluate(element => {
      const image = element.querySelector('img')!
      const svg = element.querySelector('svg')!
      const point = element.querySelector('circle')!
      const box = image.getBoundingClientRect()
      const scale = Math.min(box.width / image.naturalWidth, box.height / image.naturalHeight)
      const matrix = point.getScreenCTM()!
      const screen = new DOMPoint(point.cx.baseVal.value, point.cy.baseVal.value).matrixTransform(matrix)
      return { actual: [screen.x, screen.y], expected: [
        box.x + (box.width - image.naturalWidth * scale) / 2 + point.cx.baseVal.value * scale,
        box.y + (box.height - image.naturalHeight * scale) / 2 + point.cy.baseVal.value * scale,
      ], size: [image.naturalWidth, image.naturalHeight], viewBox: [svg.viewBox.baseVal.width, svg.viewBox.baseVal.height] }
    })
    expect(geometry.size).toEqual([96, 160])
    expect(geometry.viewBox).toEqual(geometry.size)
    expect(geometry.actual[0]).toBeCloseTo(geometry.expected[0], 3)
    expect(geometry.actual[1]).toBeCloseTo(geometry.expected[1], 3)
    await expect(panel.locator('.crop-region')).toHaveCount(1)
    // The known green patch in the stored clip rotates with the exact image.
    const pixel = await panel.locator('img').evaluate(image => {
      const canvas = document.createElement('canvas')
      canvas.width = image.naturalWidth; canvas.height = image.naturalHeight
      const context = canvas.getContext('2d')!
      context.drawImage(image, 0, 0)
      const point = image.parentElement!.querySelector('circle')!
      return [...context.getImageData(Math.floor(point.cx.baseVal.value), Math.floor(point.cy.baseVal.value), 1, 1).data]
    })
    expect(pixel[1]).toBeGreaterThan(180)
  }
})

test('rapid seeks discard late observations and failed exact frame identities', async ({ page }) => {
  await open(page, 2)
  let releaseOld!: () => void
  let receivedOld!: () => void
  let settledOld!: () => void
  const oldRequest = new Promise<void>(resolve => { receivedOld = resolve })
  const oldGate = new Promise<void>(resolve => { releaseOld = resolve })
  const oldSettled = new Promise<void>(resolve => { settledOld = resolve })
  await page.route('**/inspection/observations/window?**', async route => {
    if (route.request().url().includes('start=0.999')) {
      const response = await route.fetch()
      receivedOld()
      await oldGate
      await route.fulfill({ response }).catch(() => {})
      settledOld()
    } else await route.continue().catch(() => {})
  })
  await seek(page, '1')
  await oldRequest
  await seek(page, '1.6')
  const front = page.getByRole('article', { name: 'Camera front', exact: true })
  await expect(front).toContainText('Delivered ordinal 40')
  await expect(front.locator('circle')).toHaveCount(4)
  releaseOld()
  await oldSettled
  await expect(front).toContainText('cameras-2-front-40')
  await expect(front).not.toContainText('cameras-2-front-25')
  await page.route('**/media/front/frames/*/image', async route => {
    const response = await route.fetch()
    await route.fulfill({ response, headers: { ...response.headers(), 'x-source-pts': '999' } })
  })
  await seek(page, '1.8')
  await expect(front).toContainText('Exact frame identity changed')
  await expect(front.locator('img, circle')).toHaveCount(0)
})

test('playback follows the common cursor, reports presentation and hides unverifiable overlays', async ({ page }) => {
  await open(page, 2)
  await seek(page, '1')
  await page.getByRole('button', { name: 'Play', exact: true }).click()
  const front = page.getByRole('article', { name: 'Camera front', exact: true })
  await expect(front.locator('video')).toBeVisible()
  await expect(front).toContainText('skipped frames are not temporal evidence')
  await expect(page.getByRole('article', { name: 'Camera rotated', exact: true })).toContainText('Pause for exact inspection')
  await expect.poll(async () => front.locator('video').evaluate(video => video.currentTime)).toBeGreaterThan(1.1)
  await seek(page, '0.5')
  await expect.poll(async () => front.locator('video').evaluate(video => Math.abs(video.currentTime - Number((document.querySelector('.seek input') as HTMLInputElement).value)))).toBeLessThan(0.2)
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  await expect(front.locator('img')).toBeVisible()
  await expect(front.locator('circle')).toHaveCount(4)
})

test('missing and wrong-frame observations stay absent while exact image inspection remains available', async ({ page }) => {
  await open(page, 2)
  let wrongFrame = true
  await page.route('**/inspection/observations/window?**', async route => {
    const response = await route.fetch()
    const body = await response.json()
    body.rows = wrongFrame ? body.rows.map((row: { frame: { pts: number } }) => ({ ...row, frame: { ...row.frame, pts: row.frame.pts + 1 } })) : []
    await route.fulfill({ response, json: body })
  })
  await seek(page, '1')
  const front = page.getByRole('article', { name: 'Camera front', exact: true })
  await expect(front).toContainText('Delivered ordinal 25')
  await expect(front).toContainText('Missing observations')
  await expect(front.locator('circle')).toHaveCount(0)
  wrongFrame = false
  await seek(page, '1.04')
  await expect(front).toContainText('Delivered ordinal 26')
  await expect(front).toContainText('Missing observations')
  await expect(front.locator('circle')).toHaveCount(0)
})

test('browsers without presented-frame metadata suppress playback overlays and permit paused stepping', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(HTMLVideoElement.prototype, 'requestVideoFrameCallback', { value: undefined })
  })
  await open(page, 2)
  await page.getByRole('button', { name: 'Play', exact: true }).click()
  const front = page.getByRole('article', { name: 'Camera front', exact: true })
  await expect(front).toContainText('Presented frame metadata unavailable')
  await expect(front.locator('circle')).toHaveCount(0)
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  await expect(front.locator('img')).toBeVisible()
  await expect(front.locator('circle')).toHaveCount(4)
})
