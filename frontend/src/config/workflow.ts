export type ServiceId = "data_operations" | "defect_detection"

export type StepId = "augmentation" | "draw-rectangles" | "training" | "testing"

export type WorkflowStep = {
  id: StepId
  label: string
  path: string
  service: ServiceId
}

export type Service = {
  id: ServiceId
  label: string
  port: number
  description: string
}

export const MAIN_MENU_PATH = "/"
export const MAIN_MENU_LABEL = "Main menu"

export const SERVICES: Record<ServiceId, Service> = {
  data_operations: {
    id: "data_operations",
    label: "Data operations",
    port: 8000,
    description: "Prepare augmented and noisy images for the pipeline.",
  },
  defect_detection: {
    id: "defect_detection",
    label: "Defect detection",
    port: 8001,
    description: "Train autoencoders and evaluate visual defects.",
  },
}

export const WORKFLOW_STEPS: WorkflowStep[] = [
  {
    id: "augmentation",
    label: "Augmentation",
    path: "/augmentation",
    service: "data_operations",
  },
  {
    id: "draw-rectangles",
    label: "Draw rectangles",
    path: "/draw-rectangles",
    service: "data_operations",
  },
  {
    id: "training",
    label: "Training",
    path: "/training",
    service: "defect_detection",
  },
  {
    id: "testing",
    label: "Testing",
    path: "/testing",
    service: "defect_detection",
  },
]

export function getStep(id: StepId): WorkflowStep {
  const step = WORKFLOW_STEPS.find((s) => s.id === id)
  if (!step) {
    throw new Error(`Unknown workflow step: ${id}`)
  }
  return step
}

export function getStepNumber(id: StepId): number {
  return WORKFLOW_STEPS.findIndex((s) => s.id === id) + 1
}

export function getPreviousStep(id: StepId): WorkflowStep | null {
  const index = WORKFLOW_STEPS.findIndex((s) => s.id === id)
  return index > 0 ? WORKFLOW_STEPS[index - 1] : null
}

export function getNextStep(id: StepId): WorkflowStep | null {
  const index = WORKFLOW_STEPS.findIndex((s) => s.id === id)
  return index >= 0 && index < WORKFLOW_STEPS.length - 1
    ? WORKFLOW_STEPS[index + 1]
    : null
}

export function getServiceSteps(service: ServiceId): WorkflowStep[] {
  return WORKFLOW_STEPS.filter((s) => s.service === service)
}

export function getService(id: StepId): Service {
  return SERVICES[getStep(id).service]
}