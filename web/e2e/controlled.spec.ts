import { expect, test } from '@playwright/test'

const service = `http://127.0.0.1:${process.env.TKD_BROWSER_SERVICE_PORT ?? '18044'}`
const api = `${service}/api/projects/controlled-acceptance/inspection`

interface Action { id: string; category: string; interval: { start: number; end: number }; step_id: string }
interface Step { id: string; action_ids: string[] }
interface Keyframe { id: string; action_id: string; global_seconds: number }

test('reconstructed controlled motion generates navigable automatic arm/kick semantics', async ({ page, request }) => {
  const read = async (collection: string) => {
    const response = await request.get(`${api}/semantics/window?collection=${collection}&start=0&end=4&limit=256`)
    expect(response.ok()).toBe(true)
    return (await response.json()).rows
  }
  const actions: Action[] = await read('actions')
  const steps: Step[] = await read('steps')
  const frames: Keyframe[] = await read('keyframes')
  const arm = actions.find(a => a.category === 'arm')!
  const kick = actions.find(a => a.category === 'kick')!
  expect(arm).toBeDefined(); expect(kick).toBeDefined()
  expect(steps.some(s => s.action_ids.includes(arm.id))).toBe(true)
  expect(actions.some(a => a.category === 'arm' && a.interval.start < kick.interval.end && kick.interval.start < a.interval.end)).toBe(true)

  await page.goto('/')
  await page.getByRole('combobox', { name: 'Project', exact: true }).selectOption('controlled-acceptance')
  const panel = page.getByRole('region', { name: 'Semantic timeline' })
  const ground = page.getByRole('region', { name: 'Ground view', exact: true })
  await expect(panel).toContainText('Original automatic output')
  await expect(page.getByLabel('3D world viewport')).toBeVisible()
  await expect(page.locator('.camera-card')).toHaveCount(2)

  async function at(seconds: number) {
    const label = `${seconds.toFixed(3)} s`
    await expect(page.locator('.seek input')).toHaveValue(seconds.toFixed(3))
    await expect(panel).toContainText(`Global cursor: ${label}`)
    await expect(page.locator('.three-d')).toContainText(`Cursor ${label} · native sample ${label}`)
    await expect(ground).toContainText(`Native sample at ${label}`)
    await expect(ground.locator('.current-root')).toHaveCount(1)
    await expect(page.getByRole('article', { name: 'Camera camera-0', exact: true })).toContainText(`Delivered ordinal ${Math.round(seconds * 25)}`)
  }
  for (const action of [arm, kick]) {
    // Track lanes can repeat the same semantic entity; select the first instance.
    const button = panel.locator(`[data-entity-id="${action.id}"]`).first()
    await button.focus(); await page.keyboard.press('Enter')
    await at(action.interval.start)
    await expect(page.locator('.inspector')).toContainText(action.id)
  }
  const frame = frames.find(k => k.action_id === kick.id)!
  expect(frame).toBeDefined()
  await panel.locator(`[data-entity-id="${frame.id}"]`).first().click()
  await at(frame.global_seconds)
  const inspector = page.locator('.inspector')
  await inspector.getByText('Contributing physical samples', { exact: true }).click()
  await inspector.getByRole('button', { name: /Inspect physical sample/ }).first().click()
  await expect(inspector.locator('.geometry-values')).toContainText('root_xyz_world')
  await inspector.getByText('Contributing camera evidence', { exact: true }).click()
  await inspector.getByRole('button', { name: /Inspect camera-0 PTS/ }).first().click()
  await expect(inspector.getByRole('img', { name: /Contributing camera-0 PTS/ })).toBeVisible()
  await page.screenshot({ path: test.info().outputPath('controlled-automatic.png'), fullPage: true })
})
