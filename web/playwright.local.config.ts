import { defineConfig } from '@playwright/test'

// Explicit opt-in: uses already provisioned host data, never CI-generated fixtures.
export default defineConfig({
  testDir: './local-smoke', workers: 1, timeout: 90000,
  expect: { timeout: 30000 },
  outputDir: './test-results/local-smoke',
  use: {
    baseURL: 'http://127.0.0.1:5350', headless: true,
    viewport: { width: 1440, height: 1100 },
    screenshot: 'only-on-failure', trace: 'retain-on-failure',
    launchOptions: { args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] },
  },
  webServer: [
    { command: 'npm run dev -- --port 5350 --strictPort', env: { VITE_API_ROOT: 'http://127.0.0.1:18550' }, url: 'http://127.0.0.1:5350', reuseExistingServer: false },
    { command: '../.venv/bin/uvicorn tests.functional_service:app --host 127.0.0.1 --port 18550 --app-dir ..', env: { TKD_BROWSER_ORIGIN: 'http://127.0.0.1:5350' }, url: 'http://127.0.0.1:18550/health', reuseExistingServer: false },
  ],
})
