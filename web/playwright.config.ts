import { defineConfig } from '@playwright/test'
const origin = `http://127.0.0.1:${process.env.GROUND_PORT || '5186'}`
export default defineConfig({
  testDir: './tests',
  outputDir: process.env.GROUND_EVIDENCE_DIR || '/tmp/tkd-ground-browser-evidence',
  use: { baseURL: origin, viewport: { width: 1280, height: 1100 }, launchOptions: process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {} },
  webServer: { command: `npm run dev -- --port ${new URL(origin).port} --strictPort`, url: origin, reuseExistingServer: false },
})
