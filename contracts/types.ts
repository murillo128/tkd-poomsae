/** Portable metadata types. Runtime validation is defined by artifact.schema.json. */
export type EvidenceState = 'observed' | 'interpolated' | 'inferred' | 'unknown'
export type Track = 'left_arm' | 'right_arm' | 'left_leg' | 'right_leg' | 'body_root' | 'head'
export type Landmark =
  | 'pelvis' | 'spine' | 'neck' | 'head' | 'nose' | 'left_eye' | 'right_eye'
  | 'left_ear' | 'right_ear' | 'left_shoulder' | 'right_shoulder'
  | 'left_elbow' | 'right_elbow' | 'left_wrist' | 'right_wrist'
  | 'left_hip' | 'right_hip' | 'left_knee' | 'right_knee'
  | 'left_ankle' | 'right_ankle' | 'left_heel' | 'right_heel'
  | 'left_forefoot' | 'right_forefoot' | 'left_foot_outer' | 'right_foot_outer'
  | 'left_thumb_cmc' | 'left_thumb_mcp' | 'left_thumb_ip' | 'left_thumb_tip'
  | 'left_index_mcp' | 'left_index_pip' | 'left_index_dip' | 'left_index_tip'
  | 'left_middle_mcp' | 'left_middle_pip' | 'left_middle_dip' | 'left_middle_tip'
  | 'left_ring_mcp' | 'left_ring_pip' | 'left_ring_dip' | 'left_ring_tip'
  | 'left_pinky_mcp' | 'left_pinky_pip' | 'left_pinky_dip' | 'left_pinky_tip'
  | 'right_thumb_cmc' | 'right_thumb_mcp' | 'right_thumb_ip' | 'right_thumb_tip'
  | 'right_index_mcp' | 'right_index_pip' | 'right_index_dip' | 'right_index_tip'
  | 'right_middle_mcp' | 'right_middle_pip' | 'right_middle_dip' | 'right_middle_tip'
  | 'right_ring_mcp' | 'right_ring_pip' | 'right_ring_dip' | 'right_ring_tip'
  | 'right_pinky_mcp' | 'right_pinky_pip' | 'right_pinky_dip' | 'right_pinky_tip'
export type BodyEntity = Landmark | 'root' | 'torso' | 'left_upper_arm' | 'right_upper_arm' | 'left_forearm' | 'right_forearm' | 'left_hand' | 'right_hand' | 'left_thigh' | 'right_thigh' | 'left_shank' | 'right_shank' | 'left_foot' | 'right_foot'

export interface Provenance { producer: string; model?: string | null; model_version?: string | null; config_digest: string }
export interface ArtifactBase { id: string; schema_version: '1.0.0'; provenance: Provenance }
export interface Interval { start: number; end: number }
export interface Quality { score?: number | null; uncertainty?: number | null; state: EvidenceState; source_ids?: string[] }
export interface DenseArray { id: string; dtype: 'float32' | 'float64' | 'int32' | 'int64' | 'uint8' | 'bool'; shape: number[]; axes: string[]; unit?: string | null; missing_mask_id?: string | null }
export interface Project extends ArtifactBase { kind: 'project'; source_ids: string[]; participant_ids: string[] }
export interface Source extends ArtifactBase { kind: 'source'; camera_id: string; width_px: number; height_px: number; time_base_num: number; time_base_den: number }
export interface SyncOffset { source_id: string; automatic_seconds: number | null; manual_correction_seconds?: number | null; manual_seconds?: number | null; manual_author?: string | null; manual_source?: string | null; manual_reason?: string | null; timing_reference?: boolean; retained?: boolean; exclusion_reason?: string | null; source_interval?: Interval | null; global_interval?: Interval | null; quality: Quality }
export interface PairEstimate { first: string; second: string; shift_seconds: number | null; score: number; peak_separation: number; overlap_seconds: number; window_scores: number[]; cue_kinds: string[]; reliable: boolean; diagnostics: string[] }
export interface Synchronization extends ArtifactBase { kind: 'synchronization'; offsets: SyncOffset[]; reference_source_id?: string | null; common_interval?: Interval | null; pair_estimates?: PairEstimate[]; diagnostics?: string[] }
export interface FrameTime { source_id: string; camera_id: string; frame_index?: number | null; pts?: number | null; time_base_num?: number | null; time_base_den?: number | null; source_seconds: number; offset_seconds: number; global_seconds: number }
export interface Intrinsics { fx: number; fy: number; cx: number; cy: number; distortion?: number[] }
export interface CameraCalibration { camera_id: string; source_id: string; intrinsics: Intrinsics; world_to_camera: number[][]; quality: Quality; intrinsic_source?: 'estimated' | 'imported' | null; capture_ids?: string[]; corner_count?: number | null; rms_reprojection_px?: number | null; pose_ambiguity_px?: number | null }
export interface Calibration extends ArtifactBase { kind: 'calibration'; scale: 'metric' | 'arbitrary'; world_unit: 'm' | 'arbitrary'; cameras: CameraCalibration[]; ground_z?: number; quality: Quality }
export interface RawScore { value: number; range_min: number; range_max: number }
export interface Landmark2D { name: Landmark; xy_px: [number, number] | null; raw_score?: RawScore | null; quality: Quality }
export interface RegionOfInterest { part: 'left_hand' | 'right_hand' | 'left_foot' | 'right_foot' | 'head'; xywh_px: [number, number, number, number] }
export interface RegionalGeometry2D { part: 'left_foot' | 'right_foot' | 'head'; availability: 'complete' | 'partial' | 'missing'; orientation_state: 'available' | 'degenerate' | 'unavailable'; provider: string; supporting_landmarks?: Landmark[]; axis_start_px?: [number, number] | null; axis_end_px?: [number, number] | null; orientation_rad?: number | null }
export interface Observation extends ArtifactBase { kind: 'observation'; frame: FrameTime; landmarks: Landmark2D[]; regions?: RegionOfInterest[]; regional_geometry?: RegionalGeometry2D[]; arrays?: DenseArray[] }
export interface Landmark3D { name: Landmark; xyz_world: [number, number, number] | null; quality: Quality }
export interface Quaternion { wxyz: [number, number, number, number] }
export interface SegmentFrame { segment: string; parent: 'world' | 'root' | 'torso' | 'left_forearm' | 'right_forearm' | 'left_hand' | 'right_hand' | 'left_foot' | 'right_foot' | 'head'; orientation: Quaternion | null; quality: Quality }
export interface MotionSample { global_seconds: number; root_xyz_world: [number, number, number] | null; root_orientation: Quaternion | null; landmarks: Landmark3D[]; segments?: SegmentFrame[]; quality: Quality }
export interface Reconstruction extends ArtifactBase { kind: 'reconstruction'; calibration_id: string; participant_id: string; scale: 'metric' | 'arbitrary'; samples: MotionSample[]; arrays?: DenseArray[] }
export interface Measurement { name: string; value: number | null; unit: 'm' | 'arbitrary' | 'body_ratio' | 'deg' | 'rad'; quality: Quality }
export interface Morphology extends ArtifactBase { kind: 'morphology'; participant_id: string; measurements: Measurement[] }
export interface Contact { state: 'contact' | 'no_contact' | 'unknown'; region?: 'heel' | 'forefoot' | 'flat' | null; quality: Quality }
export interface GroundSample { global_seconds: number; left: Contact; right: Contact; support: 'both' | 'left' | 'right' | 'neither' | 'unknown' }
export interface Footprint { id: string; foot: 'left' | 'right'; interval: Interval; xy_ground: [number, number] | null; yaw_rad: number | null; quality: Quality }
export interface Pivot { id: string; foot: 'left' | 'right'; interval: Interval; region: 'heel' | 'forefoot' | 'flat' | 'unknown'; rotation_rad: number | null; quality: Quality }
export interface Ground extends ArtifactBase { kind: 'ground'; reconstruction_id: string; scale: 'metric' | 'arbitrary'; samples: GroundSample[]; footprints?: Footprint[]; pivots?: Pivot[]; measurements?: Measurement[]; arrays?: DenseArray[] }
export interface SequenceStep { id: string; interval: Interval; action_ids: string[] }
export interface StanceState { id: string; interval: Interval; label: string; quality: Quality }
export interface Action { id: string; interval: Interval; step_id: string; tracks: Track[]; category: 'arm' | 'placement' | 'pivot' | 'kick' | 'stance_transition' | 'special' | 'transition'; role?: 'attack' | 'defense' | 'preparation' | 'special' | 'unknown' }
export interface Phase { id: string; action_id: string; interval: Interval; name: string }
export interface Keyframe { id: string; action_id: string; phase_id?: string | null; global_seconds: number; event: string }
export interface SpatialRelation { id: string; subject: BodyEntity; object: BodyEntity; relation: 'in_front_of' | 'behind' | 'above' | 'below' | 'left_of' | 'right_of' | 'crossed'; interval: Interval; front_entity?: BodyEntity | null; quality: Quality }
export interface Semantics extends ArtifactBase { kind: 'semantics'; reconstruction_id: string; ground_id: string; execution: Interval; steps: SequenceStep[]; stances: StanceState[]; actions: Action[]; phases: Phase[]; keyframes: Keyframe[]; relations: SpatialRelation[] }
export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue }
export interface ManualEdit { id: string; target_id: string; field_path: string; replacement: JsonValue; author: string; reason?: string | null }
export interface ManualEdits extends ArtifactBase { kind: 'manual_edits'; automatic_semantics_id: string; edits: ManualEdit[] }
export type Artifact = Project | Source | Synchronization | Calibration | Observation | Reconstruction | Morphology | Ground | Semantics | ManualEdits
