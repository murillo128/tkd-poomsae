import { expect, it } from 'vitest'
import type { Observation } from '../../contracts/types'

const observation = {
  id: 'observation-1',
  schema_version: '1.0.0',
  provenance: { producer: 'pose', config_digest: '0'.repeat(64) },
  kind: 'observation',
  frame: {
    source_id: 'source-1', camera_id: 'front', frame_index: 0,
    source_seconds: 0, offset_seconds: 0, global_seconds: 0,
  },
  landmarks: [{
    name: 'left_heel', xy_px: null,
    raw_score: { value: 0.91, range_min: 0, range_max: 1 },
    raw_visibility: 0,
    quality: { state: 'unknown' },
  }],
  wholebody_landmarks: [{
    name: 'left_heel', xy_px: [20, 85],
    raw_score: { value: 0.91, range_min: 0, range_max: 1 },
    raw_visibility: 0,
    quality: { state: 'observed' },
  }],
  refined_landmarks: [],
  source_regional_geometry: [{
    part: 'left_foot', availability: 'complete', orientation_state: 'available',
    provider: 'wholebody', supporting_landmarks: ['left_heel'],
    axis_start_px: [20, 85], axis_end_px: [25, 92], orientation_rad: 1,
  }],
  regional_geometry: [{
    part: 'left_foot', availability: 'missing', orientation_state: 'unavailable',
    provider: 'wholebody',
  }],
  subject_selection: {
    state: 'ambiguous', track_id: 'front:practitioner:0', candidate_index: null,
    method: null, reasons: ['candidate_match_ambiguous'],
    candidates: [{
      index: 0, bbox_xyxy_px: [0, 0, 80, 100],
      detector_score: { value: 0.8, range_min: 0, range_max: 1 },
      match_cost: 0.4,
    }],
  },
  region_quality: [{ part: 'left_foot', usable: false, reasons: ['low_raw_visibility'] }],
} satisfies Observation

it('exposes practitioner selection and independent source/derived regional evidence to clients', () => {
  const restored: Observation = JSON.parse(JSON.stringify(observation))
  expect(restored.subject_selection?.candidates?.[0].detector_score.value).toBe(0.8)
  expect(restored.region_quality?.[0].usable).toBe(false)
  expect(restored.landmarks[0].xy_px).toBeNull()
  expect(restored.landmarks[0].raw_visibility).toBe(0)
  expect(restored.wholebody_landmarks?.[0].xy_px).toEqual([20, 85])
  expect(restored.source_regional_geometry?.[0].availability).toBe('complete')
  expect(restored.regional_geometry?.[0].availability).toBe('missing')
})
