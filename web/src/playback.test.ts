import { describe, expect, it } from 'vitest'
import { canStep, initialPlayback, playbackReducer, type PlaybackState } from './playback'

const reduce = (state: PlaybackState, action: Parameters<typeof playbackReducer>[1]) => playbackReducer(state, action)

describe('shared playback controller', () => {
  const cameraState = () => reduce(initialPlayback, { type: 'project', id: 'synthetic', cameras: [
    { id: 'front', offsetSeconds: 0.1, frames: [
      { pts: 0, timeBaseNum: 1, timeBaseDen: 10 },
      { pts: 1, timeBaseNum: 1, timeBaseDen: 10 },
      { pts: 3, timeBaseNum: 1, timeBaseDen: 10 },
    ] },
    { id: 'side', offsetSeconds: 0.05, frames: [
      { pts: 0, timeBaseNum: 1, timeBaseDen: 8 },
      { pts: 2, timeBaseNum: 1, timeBaseDen: 8 },
    ] },
  ], reconstructionTimes: [0.15, 0.35] })

  it('seeks, plays and pauses at the selected speed without changing delivered frame time', () => {
    let state = cameraState()
    state = reduce(state, { type: 'seek', seconds: 1.5 })
    state = reduce(state, { type: 'frameDelivered', seconds: 1.42 })
    state = reduce(state, { type: 'speed', speed: 2 })
    state = reduce(state, { type: 'play', playing: true })
    state = reduce(state, { type: 'tick', elapsedSeconds: 0.25 })
    expect(state.cursorSeconds).toBe(2)
    expect(state.deliveredFrameSeconds).toBe(1.42)
    state = reduce(state, { type: 'play', playing: false })
    expect(reduce(state, { type: 'tick', elapsedSeconds: 1 }).cursorSeconds).toBe(2)
  })

  it('steps by native PTS of the selected camera mapped into global time', () => {
    let state = cameraState()
    state = reduce(state, { type: 'seek', seconds: 0.2 })
    state = reduce(state, { type: 'step', direction: 1 })
    expect(state.cursorSeconds).toBeCloseTo(0.4)
    state = reduce(state, { type: 'camera', id: 'side' })
    state = reduce(state, { type: 'step', direction: -1 })
    expect(state.cursorSeconds).toBeCloseTo(0.3)
    state = reduce(state, { type: 'mode', mode: 'reconstruction' })
    state = reduce(state, { type: 'step', direction: 1 })
    expect(state.cursorSeconds).toBeCloseTo(0.35)
  })

  it('propagates one selection and resets clock, selection and frame on project change', () => {
    let state = cameraState()
    state = reduce(state, { type: 'select', selection: { kind: 'entity', id: 'footprint-1', tracks: ['left_leg'] } })
    expect(state.selection?.tracks).toEqual(['left_leg'])
    state = reduce(state, { type: 'seek', seconds: 4 })
    state = reduce(state, { type: 'frameDelivered', seconds: 3.9 })
    state = reduce(state, { type: 'project', id: 'other' })
    expect(state).toMatchObject({ projectId: 'other', cursorSeconds: 0, deliveredFrameSeconds: null, selection: null })
    expect(canStep(state)).toBe(false)
  })
})
