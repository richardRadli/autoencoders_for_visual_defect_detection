import { DEFECT, apiGet, apiPost, buildUrl } from "./client"
import type {
  AbortResult,
  DatasetType,
  DefectPreviewList,
  DefectPreviewType,
  DefectReadiness,
  DeviceStatus,
  NetworkType,
  QueuedTask,
  SubtestFolder,
  TestStatus,
  TestingParams,
  TrainStatus,
  TrainingParams,
  WeightsList,
} from "./types"

/* Device ----------------------------------------------------------------- */

export function getDeviceStatus(signal?: AbortSignal): Promise<DeviceStatus> {
  return apiGet<DeviceStatus>(`${DEFECT}/device-status`, signal)
}

/* Training --------------------------------------------------------------- */

/* Unlike the data_operations runs, these return immediately with a task id —
   the work happens in a Celery worker and is followed through the status
   endpoint. */

export function runTraining(params: TrainingParams): Promise<QueuedTask> {
  return apiPost<QueuedTask>(buildUrl(DEFECT, "/train/run", params))
}

export function getTrainingStatus(
  taskId: string,
  signal?: AbortSignal,
): Promise<TrainStatus> {
  return apiGet<TrainStatus>(
    `${DEFECT}/train/status/${encodeURIComponent(taskId)}`,
    signal,
  )
}

export function stopTraining(taskId: string): Promise<AbortResult> {
  return apiPost<AbortResult>(`${DEFECT}/train/stop/${encodeURIComponent(taskId)}`)
}

/* Testing ---------------------------------------------------------------- */

export function runTesting(params: TestingParams): Promise<QueuedTask> {
  return apiPost<QueuedTask>(buildUrl(DEFECT, "/test/run", params))
}

export function getTestingStatus(
  taskId: string,
  signal?: AbortSignal,
): Promise<TestStatus> {
  return apiGet<TestStatus>(
    `${DEFECT}/test/status/${encodeURIComponent(taskId)}`,
    signal,
  )
}

export function stopTesting(taskId: string): Promise<AbortResult> {
  return apiPost<AbortResult>(`${DEFECT}/test/stop/${encodeURIComponent(taskId)}`)
}

/* Dataset ---------------------------------------------------------------- */

export function getDefectReadiness(
  dataset_type: DatasetType,
  signal?: AbortSignal,
): Promise<DefectReadiness> {
  return apiGet<DefectReadiness>(
    buildUrl(DEFECT, "/dataset/readiness", { dataset_type }),
    signal,
  )
}

export function getWeightsList(
  dataset_type: DatasetType,
  network_type: NetworkType,
  signal?: AbortSignal,
): Promise<WeightsList> {
  return apiGet<WeightsList>(
    buildUrl(DEFECT, "/dataset/weights", {
      dataset_type,
      network_type,
    }),
    signal,
  )
}

export type DefectPreviewOptions = {
  subtest_folder?: SubtestFolder
  network_type?: NetworkType
  limit?: number
}

/* 'test' previews are keyed by subtest_folder, the generated outputs
   (reconstruction / reconstruction_vis / roc_plot) by network_type. Sending
   the wrong one, or neither, returns 422. */
export function getDefectPreview(
  dataset_type: DatasetType,
  preview_type: DefectPreviewType,
  options: DefectPreviewOptions = {},
  signal?: AbortSignal,
): Promise<DefectPreviewList> {
  const { subtest_folder, network_type, limit = 5 } = options
  return apiGet<DefectPreviewList>(
    buildUrl(DEFECT, "/dataset/preview", {
      dataset_type,
      preview_type,
      subtest_folder,
      network_type,
      limit,
    }),
    signal,
  )
}

export function defectPreviewImageUrl(
  dataset_type: DatasetType,
  preview_type: DefectPreviewType,
  name: string,
  options: Omit<DefectPreviewOptions, "limit"> = {},
): string {
  const { subtest_folder, network_type } = options
  return buildUrl(DEFECT, "/dataset/preview/image", {
    dataset_type,
    preview_type,
    name,
    subtest_folder,
    network_type,
  })
}