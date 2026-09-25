import type { Track } from '../../contracts/types'

export interface FrameSample { pts: number; timeBaseNum: number; timeBaseDen: number }
export interface CameraClock { id: string; offsetSeconds: number; frames: FrameSample[] }
export interface Selection { kind: 'track' | 'entity'; id: string; tracks: Track[] }
export interface PlaybackState {
  projectId: string | null
  cursorSeconds: number
  deliveredFrameSeconds: number | null
  playing: boolean
  speed: number
  selectedCameraId: string | null
  stepMode: 'camera' | 'reconstruction'
  cameras: CameraClock[]
  reconstructionTimes: number[]
  selection: Selection | null
}
export type PlaybackAction =
  | { type: 'project'; id: string | null; cameras?: CameraClock[]; reconstructionTimes?: number[] }
  | { type: 'seek'; seconds: number }
  | { type: 'play'; playing: boolean }
  | { type: 'speed'; speed: number }
  | { type: 'tick'; elapsedSeconds: number }
  | { type: 'camera'; id: string }
  | { type: 'mode'; mode: PlaybackState['stepMode'] }
  | { type: 'step'; direction: -1 | 1 }
  | { type: 'frameDelivered'; seconds: number | null }
  | { type: 'select'; selection: Selection | null }

export const initialPlayback: PlaybackState = {
  projectId: null, cursorSeconds: 0, deliveredFrameSeconds: null,
  playing: false, speed: 1, selectedCameraId: null, stepMode: 'camera',
  cameras: [], reconstructionTimes: [], selection: null,
}

function sampleTimes(state: PlaybackState): number[] {
  if (state.stepMode === 'reconstruction') return state.reconstructionTimes.filter(Number.isFinite).sort((a, b) => a - b)
  const camera = state.cameras.find(item => item.id === state.selectedCameraId)
  if (!camera || !Number.isFinite(camera.offsetSeconds)) return []
  return camera.frames
    .filter(frame => Number.isFinite(frame.pts) && frame.timeBaseNum > 0 && frame.timeBaseDen > 0)
    .map(frame => frame.pts * frame.timeBaseNum / frame.timeBaseDen + camera.offsetSeconds)
    .sort((a, b) => a - b)
}

export function canStep(state: PlaybackState): boolean { return sampleTimes(state).length > 0 }

export function playbackReducer(state: PlaybackState, action: PlaybackAction): PlaybackState {
  switch (action.type) {
    case 'project': {
      const cameras = action.cameras ?? []
      return { ...initialPlayback, projectId: action.id, cameras,
        selectedCameraId: cameras[0]?.id ?? null,
        reconstructionTimes: action.reconstructionTimes ?? [] }
    }
    case 'seek':
      return Number.isFinite(action.seconds) ? { ...state, cursorSeconds: Math.max(0, action.seconds) } : state
    case 'play': return { ...state, playing: action.playing }
    case 'speed': return Number.isFinite(action.speed) && action.speed > 0 ? { ...state, speed: action.speed } : state
    case 'tick':
      return state.playing && Number.isFinite(action.elapsedSeconds) && action.elapsedSeconds > 0
        ? { ...state, cursorSeconds: state.cursorSeconds + action.elapsedSeconds * state.speed }
        : state
    case 'camera': return state.cameras.some(camera => camera.id === action.id)
      ? { ...state, selectedCameraId: action.id } : state
    case 'mode': return { ...state, stepMode: action.mode }
    case 'step': {
      const times = sampleTimes(state)
      const epsilon = 1e-9
      const target = action.direction > 0
        ? times.find(time => time > state.cursorSeconds + epsilon)
        : [...times].reverse().find(time => time < state.cursorSeconds - epsilon)
      return target === undefined ? state : { ...state, cursorSeconds: Math.max(0, target), playing: false }
    }
    case 'frameDelivered': return { ...state, deliveredFrameSeconds: action.seconds }
    case 'select': return { ...state, selection: action.selection }
  }
}
