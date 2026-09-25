import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { App } from './App'

describe('local observation shell', () => {
  it('renders the starting workspace without requiring an API response', () => {
    const html = renderToStaticMarkup(<App />)
    expect(html).toContain('Local motion inspection workspace')
    expect(html).toContain('The observation interface is being built.')
  })
})
