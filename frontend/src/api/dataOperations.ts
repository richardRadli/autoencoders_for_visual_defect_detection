import { DATA_OPS, apiPost, buildQuery } from "./client"

export type AugmentationParams = {
  dataset_type: string
  img_size?: number
  crop_size?: number
  rotate_count?: number
  horizontal_flip_count?: number
  vertical_flip_count?: number
}

export type AugmentationResult = {
  status: string
  dataset_type: string
  source_images: number
  augmented_images: number
  processed_images: number
  source_dir: string
  target_dir: string
  warning?: string
}

export type DrawRectanglesParams = {
  dataset_type: string
  source?: string
  size_of_cover?: number
}

export type DrawRectanglesResult = {
  status: string
  dataset_type: string
  source: string
  processed_images: number
  source_dir: string
  target_dir: string
}

export const runAugmentation = (params: AugmentationParams) =>
  apiPost<AugmentationResult>(`${DATA_OPS}/augmentation/run${buildQuery(params)}`)

export const stopAugmentation = () =>
  apiPost<{ status: string; service: string }>(`${DATA_OPS}/augmentation/stop`)

export const runDrawRectangles = (params: DrawRectanglesParams) =>
  apiPost<DrawRectanglesResult>(`${DATA_OPS}/draw-rectangles/run${buildQuery(params)}`)

export const stopDrawRectangles = () =>
  apiPost<{ status: string; service: string }>(`${DATA_OPS}/draw-rectangles/stop`)