import { expect, test } from '@playwright/test'

test('persisted contributors, camera scores, sync conflict and stale reconstruction survive reopening', async ({ page, request }) => {
  const writes: string[] = []
  page.on('request', req => { if (req.method() === 'POST') writes.push(new URL(req.url()).pathname) })
  await page.goto('/')
  await page.locator('.project-actions select').selectOption('provenance')
  const seek = page.locator('.seek input')
  await seek.fill('0.2'); await seek.press('Enter')
  await page.getByLabel('Pick 3D landmark').selectOption('left_wrist')
  const inspector = page.locator('.inspector')
  await expect(inspector).toContainText('inspection-motion/samples/0/left_wrist')
  await expect(inspector).toContainText('metric scale unresolved')
  await inspector.getByText('Contributing camera evidence', { exact: true }).click()
  await inspector.getByRole('button', { name: /Inspect rotated PTS/ }).click()
  await expect(inspector.getByRole('img', { name: /Contributing rotated PTS/ })).toBeVisible()
  await expect(seek).toHaveValue('0.200')
  await expect(inspector).toContainText('sample-time mismatch 0.000000 s')
  await inspector.getByRole('button', { name: /Inspect front PTS/ }).click()
  await expect(inspector.getByRole('img', { name: /Contributing front PTS/ })).toBeVisible()
  await expect(inspector).toContainText('sample-time mismatch -0.040000 s')
  await expect(seek).toHaveValue('0.200')
  await inspector.getByText('Native scores and generating provenance', { exact: true }).first().click()
  await expect(inspector).toContainText('"value": 7')
  await inspector.getByText('Contributing physical samples', { exact: true }).click()
  await inspector.getByRole('button', { name: 'Inspect physical sample 0.2 s', exact: true }).click()
  await expect(inspector).toContainText('inspection-motion/samples/0')
  await expect(inspector.locator('.geometry-values')).toContainText('root_xyz_world')
  await expect(seek).toHaveValue('0.200')
  // Inspecting a camera landmark does not depend on a reconstruction artifact.
  const front = page.getByRole('article', { name: 'Camera front', exact: true })
  await front.getByRole('button', { name: 'front left_wrist', exact: true }).focus(); await page.keyboard.press('Enter')
  await expect(inspector).toContainText('pixels (px)')
  await expect(inspector).toContainText('"raw_score"')
  expect(writes).toEqual([])

  const controls = page.getByRole('group', { name: 'front synchronization', exact: true })
  await controls.getByLabel('front offset seconds').fill('0.04')
  await controls.getByLabel('front sync author').fill('browser-operator')
  await controls.getByLabel('front sync reason').fill('inspect synchronization')
  // A failed transport must retain the draft and preserved automatic estimate.
  await page.route('**/inspection/sync-offset', route => route.fulfill({ status: 503, json: { detail: 'temporary service failure' } }))
  await controls.getByRole('button', { name: 'Save front offset' }).click()
  await expect(controls.getByRole('alert')).toContainText('Save rejected (503)')
  await expect(controls.getByLabel('front offset seconds')).toHaveValue('0.04')
  await page.unroute('**/inspection/sync-offset')
  const root = process.env.TKD_BROWSER_SERVICE_PORT ?? '18044'
  const api = `http://127.0.0.1:${root}/api/projects/provenance/inspection`
  const meta = await (await request.get(api)).json()
  const concurrent = await request.post(`${api}/sync-offset`, {
    headers: { 'x-tkd-local-request': '1' }, data: {
      camera: 'rotated', offset_seconds: -0.19, expected_revision: meta.sync_revision,
      author: 'other-operator', reason: 'concurrent edit', source: 'browser-test',
    },
  })
  expect(concurrent.ok()).toBe(true)
  await controls.getByRole('button', { name: 'Save front offset' }).click()
  await expect(controls.getByRole('alert')).toContainText('Stale revision / conflict')
  await expect(controls.getByLabel('front offset seconds')).toHaveValue('0.04')
  await expect(controls).toContainText('Automatic estimate 0 s')
  // Explicitly discard, load the new base, then submit this operator's revision.
  await controls.getByRole('button', { name: 'Discard front offset draft' }).click()
  await controls.getByRole('button', { name: 'Reload front synchronization' }).click()
  await expect(controls.getByLabel('front offset seconds')).toBeEnabled()
  await controls.getByLabel('front offset seconds').fill('0.04')
  await controls.getByRole('button', { name: 'Save front offset' }).click()
  await expect(page.locator('.three-d')).toContainText('artifact unavailable or stale')
  await expect(page.getByLabel('Pick 3D landmark')).toHaveCount(0)
  await expect(seek).toHaveValue('0.200')
  await expect(page.getByRole('group', { name: 'front synchronization', exact: true })).toContainText('effective 0.04 s')
  await page.reload(); await page.locator('.project-actions select').selectOption('provenance')
  const reopened = page.getByRole('group', { name: 'front synchronization', exact: true })
  await expect(reopened).toContainText('Automatic estimate 0 s · effective 0.04 s')
  await reopened.getByText('Manual edit provenance', { exact: true }).click()
  await expect(reopened).toContainText('browser-operator')
  await expect(page.locator('.three-d')).toContainText('artifact unavailable or stale')
  expect(writes.every(path => path.endsWith('/inspection/sync-offset'))).toBe(true)
})
