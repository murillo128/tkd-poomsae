import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: './tests',
  outputDir: process.env.GROUND_EVIDENCE_DIR || '/tmp/tkd-ground-browser-evidence',
  use: { baseURL: 'http://127.0.0.1:5186', viewport: { width: 1280, height: 1100 }, launchOptions: process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {} },
  webServer: { command: 'npm run dev -- --port 5186 --strictPort', url: 'http://127.0.0.1:5186', reuseExistingServer: false },
})
