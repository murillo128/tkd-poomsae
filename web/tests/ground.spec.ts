import { expect, test, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import type { GroundSummary, GroundMetadata } from '../src/groundApi'
const fixture = JSON.parse(readFileSync(new URL('./fixtures/ground.json', import.meta.url), 'utf8')) as { meta: GroundMetadata; summary: { ground_frames: GroundSummary['frames']; placements: GroundSummary['placements']; rotations: GroundSummary['rotations']; placement_relations: GroundSummary['relations'] } }

async function open(page: Page, arbitrary = false, unavailable = false) {
  const data = structuredClone(fixture)
  if (arbitrary) data.meta.world_unit = 'arbitrary'
  const stage = { available: true, reason: null, schema_version: '1.0.0', software_revision: 'fixture', model_revision: null, status: { status: 'complete', key: 'fixture' } }
  await page.route('http://127.0.0.1:8000/api/**', async route => {
    const url = new URL(route.request().url())
    let body: unknown
    if (url.pathname.endsWith('/api/projects')) body = { projects: ['synthetic'] }
    else if (url.pathname.endsWith('/capabilities')) body = { project: 'synthetic', stages: { ground: stage } }
    else if (url.pathname.endsWith('/snapshot')) {
      const seconds = Number(url.searchParams.get('seconds'))
      const frame = data.summary.ground_frames.find(f => Math.abs(f.sampled_seconds - seconds) < 1e-8)
      body = { revision: 'fixture', available: !unavailable && Boolean(frame), reason: unavailable ? 'ground_unavailable' : frame ? null : 'outside_execution', ground_view: unavailable ? null : data.meta, snapshot: frame ?? null }
    } else if (url.pathname.endsWith('/window')) {
      const name = url.searchParams.get('collection') as keyof typeof data.summary
      body = { revision: 'fixture', available: true, rows: data.summary[name], next_cursor: null }
    } else body = { id: 'synthetic', sources: [], state: { stages: { ground: stage.status } } }
    await route.fulfill({ json: body, headers: { 'access-control-allow-origin': 'http://127.0.0.1:5186' } })
  })
  await page.goto('/')
  await page.getByRole('combobox', { name: 'Project', exact: true }).selectOption('synthetic')
}

test('actual XY rendering, dynamic/summary layers and shared selection', async ({ page }) => {
  await open(page)
  const panel = page.getByRole('region', { name: 'Ground view', exact: true })
  const scene = panel.getByLabel('Top-down ground XY scene')
  await expect(scene).toBeVisible()
  const box = await scene.getAttribute('viewBox')
  await expect(panel).toContainText('Metric ground coordinates (m)')
  // Backend XY maps to SVG X,-Y, with unchanged landmark orientation.
  const root = fixture.summary.ground_frames[0].root!
  await expect(scene.locator('.current-root')).toHaveAttribute('cx', String(root.xy_ground![0]))
  await expect(scene.locator('.current-root')).toHaveAttribute('cy', String(-root.xy_ground![1]))
  const foot = fixture.summary.ground_frames[0].feet[0].geometry
  await expect(scene.locator('.current-foot.left line')).toHaveAttribute('x1', String(foot.axis_start![0]))
  await expect(scene.locator('.current-foot.left line')).toHaveAttribute('y2', String(-foot.axis_end![1]))
  await expect(scene.locator('.pivot-path')).toHaveCount(1)
  await expect(scene.locator('.ground-foot circle').first()).toHaveCSS('stroke', 'none')
  await expect(scene.getByRole('button', { name: /right stable placement/ })).toBeVisible()
  await panel.screenshot({ path: test.info().outputPath('dynamic.png') })
  await panel.getByRole('combobox', { name: 'View', exact: true }).selectOption('summary')
  await expect(scene.locator('.current-foot')).toHaveCount(0)
  await expect(scene.locator('.root-point')).toHaveCount(6)
  const pivot = scene.getByRole('button', { name: /left pivot/ })
  await pivot.focus(); await page.keyboard.press('Enter')
  const time = fixture.summary.rotations[0].pivot.interval.start
  await expect(page.locator('.seek input')).toHaveValue(time.toFixed(3))
  await expect(page.locator('.inspector')).toContainText(fixture.summary.rotations[0].pivot.id)
  await expect(pivot).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('button', { name: 'left leg', exact: true }).click()
  await expect(pivot).toHaveAttribute('aria-pressed', 'true')
  await expect(scene.getByRole('button', { name: /left .* placement/ }).first()).toHaveAttribute('aria-pressed', 'true')
  await panel.getByLabel('Pivots', { exact: true }).uncheck()
  await expect(scene.locator('.pivot-path')).toHaveCount(0)
  await panel.getByLabel('Pivots', { exact: true }).check()
  await panel.getByLabel('Footprints', { exact: true }).uncheck()
  await expect(scene.locator('.ground-foot circle')).toHaveCount(0)
  await panel.getByLabel('Footprints', { exact: true }).check()
  await panel.getByLabel('Foot axes', { exact: true }).uncheck()
  await expect(scene.locator('.ground-foot line')).toHaveCount(0)
  await panel.getByLabel('Foot axes', { exact: true }).check()
  await panel.getByLabel('Root path', { exact: true }).uncheck()
  await expect(scene.locator('.root-point')).toHaveCount(0)
  await panel.getByLabel('Root path', { exact: true }).check()
  await panel.getByLabel('Contacts', { exact: true }).uncheck()
  await expect(panel.getByLabel('Contact and support events')).toHaveCount(0)
  await panel.getByLabel('Contacts', { exact: true }).check()
  await panel.getByLabel('Measurements', { exact: true }).uncheck()
  await expect(panel.locator('.ground-measurements')).toHaveCount(0)
  await panel.getByLabel('Measurements', { exact: true }).check()
  await panel.screenshot({ path: test.info().outputPath('summary.png') })
  await scene.getByRole('button', { name: 'Root path at 0.4 seconds' }).click()
  await panel.getByRole('combobox', { name: 'View', exact: true }).selectOption('dynamic')
  await expect(panel).toContainText('Native sample at 0.400 s')
  await expect(scene).toHaveAttribute('viewBox', box!)
  const later = fixture.summary.ground_frames[2].root!
  await expect(scene.locator('.current-root')).toHaveAttribute('cx', String(later.xy_ground![0]))
  await expect(scene.locator('.current-root')).toHaveAttribute('cy', String(-later.xy_ground![1]))
})

test('unresolved scale and missing ground remain explicit', async ({ page }) => {
  await open(page, true)
  const panel = page.getByRole('region', { name: 'Ground view', exact: true })
  await expect(panel).toContainText('Metric scale unresolved')
  await expect(panel).not.toContainText('coordinates (m)')
  await expect(panel.locator('.ground-measurements')).not.toContainText(/\d+(?:\.\d+)? m(?: |$)/)
  await panel.screenshot({ path: test.info().outputPath('arbitrary.png') })
})

test('missing ground renders an unavailable panel', async ({ page }) => {
  await open(page, false, true)
  await expect(page.getByRole('region', { name: 'Ground view', exact: true })).toContainText('Ground view unavailable')
  await expect(page.getByLabel('Top-down ground XY scene')).toHaveCount(0)
})
