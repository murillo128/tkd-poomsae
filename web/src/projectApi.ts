export interface ProjectList { projects: string[] }
export interface ProjectDetail {
  id: string
  sources: string[]
  state: { stages: Record<string, StageStatus> }
}
export interface StageStatus {
  status: string
  key?: string
  diagnostics?: string[]
  software_revision?: string
  model_revision?: string | null
}
export interface StageCapability {
  available: boolean
  reason: string | null
  schema_version: string
  software_revision: string
  model_revision: string | null
  status: StageStatus
}
export interface ProjectCapabilities { project: string; stages: Record<string, StageCapability> }
export interface ProjectSnapshot { detail: ProjectDetail; capabilities: ProjectCapabilities; revision: string }

// Local service metadata and bounded inspection endpoints.
export const API_ROOT = import.meta.env.VITE_API_ROOT || 'http://127.0.0.1:8000'
if (!['127.0.0.1', 'localhost', '[::1]'].includes(new URL(API_ROOT).hostname)) {
  throw new Error('Inspection requires a local service URL')
}
export async function readJson<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, { signal })
  if (!response.ok) throw new Error(`Local service returned ${response.status} for ${path}`)
  return response.json() as Promise<T>
}
export function listProjects(signal: AbortSignal): Promise<ProjectList> {
  return readJson('/api/projects', signal)
}
export async function readProject(id: string, signal: AbortSignal): Promise<ProjectSnapshot> {
  const path = `/api/projects/${encodeURIComponent(id)}`
  const [detail, capabilities] = await Promise.all([
    readJson<ProjectDetail>(path, signal),
    readJson<ProjectCapabilities>(`${path}/capabilities`, signal),
  ])
  if (detail.id !== id || capabilities.project !== id) throw new Error('Project response identity changed')
  const revision = JSON.stringify(Object.entries(capabilities.stages).map(([name, stage]) => [
    name, stage.status.key ?? null, stage.status.status, stage.software_revision,
    stage.model_revision, stage.schema_version,
  ]))
  return { detail, capabilities, revision }
}
