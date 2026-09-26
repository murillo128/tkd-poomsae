import { defineConfig } from '@playwright/test'

const webPort = Number(process.env.TKD_BROWSER_WEB_PORT ?? 5173)
const servicePort = Number(process.env.TKD_BROWSER_SERVICE_PORT ?? 18044)
for (const port of [webPort, servicePort]) {
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Browser test ports must be integers in 1–65535')
}
const baseURL = `http://127.0.0.1:${webPort}`
const serviceURL = `http://127.0.0.1:${servicePort}`

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  outputDir: './test-results/integrated',
  use: {
    baseURL, headless: true, viewport: { width: 1440, height: 1100 },
    trace: 'retain-on-failure', screenshot: 'only-on-failure',
    launchOptions: { args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] },
  },
  webServer: [
    { command: `npm run dev -- --port ${webPort} --strictPort`, env: { VITE_API_ROOT: serviceURL }, url: baseURL, reuseExistingServer: false },
    { command: `../.venv/bin/uvicorn tests.browser_service:app --host 127.0.0.1 --port ${servicePort} --app-dir ..`, env: { TKD_BROWSER_ORIGIN: baseURL }, url: `${serviceURL}/health`, timeout: 120000, reuseExistingServer: false },
  ],
})
