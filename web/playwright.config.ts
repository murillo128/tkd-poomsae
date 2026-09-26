import { defineConfig } from '@playwright/test'

const groundPort = Number(process.env.GROUND_PORT ?? 5186)
const threePort = Number(process.env.TKD_BROWSER_PORT ?? 5173)
for (const port of [groundPort, threePort]) {
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid browser test port')
}
if (groundPort === threePort) throw new Error('Ground and 3D browser suites need distinct ports')
const groundOrigin = `http://127.0.0.1:${groundPort}`
const threeOrigin = `http://127.0.0.1:${threePort}`
const executable = process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}

export default defineConfig({
  fullyParallel: false,
  outputDir: './test-results/components',
  projects: [
    {
      name: 'ground', testDir: './tests',
      outputDir: process.env.GROUND_EVIDENCE_DIR || '/tmp/tkd-ground-browser-evidence',
      use: { baseURL: groundOrigin, viewport: { width: 1280, height: 1100 }, launchOptions: executable },
    },
    {
      name: 'three-d', testDir: './browser',
      use: {
        baseURL: threeOrigin, viewport: { width: 1440, height: 1200 },
        launchOptions: { ...executable, args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] },
      },
    },
  ],
  // Each suite owns its server; never test another worktree or revision silently.
  webServer: [
    { command: '.venv/bin/python -m tests.fixtures.serve_timeline', cwd: '..', url: 'http://127.0.0.1:5197/health', reuseExistingServer: false },
    { command: `npm run dev -- --port ${groundPort} --strictPort`, url: groundOrigin, reuseExistingServer: false },
    { command: `npm run dev -- --port ${threePort} --strictPort`, url: threeOrigin, reuseExistingServer: false },
  ],
})
