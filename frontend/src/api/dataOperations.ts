import { DATA_OPS, apiGet, apiPost, buildUrl } from "./client"
import type {
  AugmentationParams,
  AugmentationResult,
  DatasetType,
  DrawRectanglesParams,
  DrawRectanglesResult,
  OpsPreviewList,
  OpsPreviewType,
  OpsReadiness,
  StopResult,
} from "./types"

/* Runs ------------------------------------------------------------------- */

/* The run endpoints block until the job finishes and return the whole result
   in one response — there is no task id and nothing to poll. They take no
   AbortSignal on purpose: aborting the fetch would drop the response but keep
   the job running on the server. Use the stop endpoints for that. */

export function runAugmentation(params: AugmentationParams): Promise<AugmentationResult> {
  return apiPost<AugmentationResult>(buildUrl(DATA_OPS, "/augmentation/run", params))
}

export function stopAugmentation(): Promise<StopResult> {
  return apiPost<StopResult>(`${DATA_OPS}/augmentation/stop`)
}

export function runDrawRectangles(
  params: DrawRectanglesParams,
): Promise<DrawRectanglesResult> {
  return apiPost<DrawRectanglesResult>(buildUrl(DATA_OPS, "/draw-rectangles/run", params))
}

export function stopDrawRectangles(): Promise<StopResult> {
  return apiPost<StopResult>(`${DATA_OPS}/draw-rectangles/stop`)
}

/* Dataset ---------------------------------------------------------------- */

export function getOpsReadiness(
  dataset_type: DatasetType,
  signal?: AbortSignal,
): Promise<OpsReadiness> {
  return apiGet<OpsReadiness>(
    buildUrl(DATA_OPS, "/dataset/readiness", { dataset_type }),
    signal,
  )
}

export function getOpsPreview(
  dataset_type: DatasetType,
  preview_type: OpsPreviewType,
  limit = 5,
  signal?: AbortSignal,
): Promise<OpsPreviewList> {
  return apiGet<OpsPreviewList>(
    buildUrl(DATA_OPS, "/dataset/preview", { dataset_type, preview_type, limit }),
    signal,
  )
}

/* Preview images are rendered by the browser, not fetched as JSON, so this
   returns a URL for <img src> rather than a promise. The name must be an
   entry taken verbatim from getOpsPreview's images list. */
export function opsPreviewImageUrl(
  dataset_type: DatasetType,
  preview_type: OpsPreviewType,
  name: string,
): string {
  return buildUrl(DATA_OPS, "/dataset/preview/image", {
    dataset_type,
    preview_type,
    name,
  })
}