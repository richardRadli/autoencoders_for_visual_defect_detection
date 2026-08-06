/* Shared API enums -------------------------------------------------------- */

export type DatasetType = "texture_1" | "texture_2" | "cpu"
export type NetworkType = "AE" | "AEE" | "DAE" | "DAEE"
export type AEType = "plain" | "denoising"
export type ModelSize = "base" | "extended"
export type SubtestFolder = "defective" | "added" | "contamination" | "missing"
export type SourceImage = "good" | "aug"

export type ImgSize = 256 | 512 | 1024
export type CropSize = 64 | 128 | 256 | 512
export type Stride = 4 | 8 | 16 | 32 | 64

export const DATASET_TYPES: DatasetType[] = ["texture_1", "texture_2", "cpu"]
export const AE_TYPES: AEType[] = ["plain", "denoising"]
export const MODEL_SIZES: ModelSize[] = ["base", "extended"]
export const SOURCE_IMAGES: SourceImage[] = ["good", "aug"]
export const IMG_SIZES: ImgSize[] = [256, 512, 1024]
export const CROP_SIZES: CropSize[] = [64, 128, 256, 512]
export const STRIDES: Stride[] = [4, 8, 16, 32, 64]

/* Which subtest folders each dataset actually has. Mirrors VALID_SUBTESTS
   in the test API — sending anything else returns 422. */
export const VALID_SUBTESTS: Record<DatasetType, SubtestFolder[]> = {
  texture_1: ["defective"],
  texture_2: ["defective"],
  cpu: ["added", "contamination", "missing"],
}

/* Bounds enforced by the API — the forms mirror them so the user does not
   have to discover them through a 422. */
export const LIMITS = {
  augmentedTotalMin: 5000,
  augmentedTotalMax: 20000,
  sizeOfCoverMin: 4,
  sizeOfCoverMax: 64,
  thresholdEndMax: 2,
  testVisIntervalMax: 50,
} as const

/* data_operations --------------------------------------------------------- */

export type AugmentationParams = {
  dataset_type: DatasetType
  img_size: ImgSize
  crop_size: CropSize
  rotate_count?: number
  horizontal_flip_count?: number
  vertical_flip_count?: number
}

export type AugmentationResult = {
  status: string
  dataset_type: DatasetType
  source_images: number
  augmented_images: number
  processed_images: number
  source_dir: string
  target_dir: string
  warning?: string
}

export type DrawRectanglesParams = {
  dataset_type: DatasetType
  source?: SourceImage
  size_of_cover?: number
}

export type DrawRectanglesResult = {
  status: string
  dataset_type: DatasetType
  source: SourceImage
  processed_images: number
  source_dir: string
  target_dir: string
}

export type StopResult = {
  status: string
  service: string
}

export type ReadinessFact = {
  ready: boolean
  images: number
}

export type OpsReadiness = {
  dataset_type: DatasetType
  augmentation: ReadinessFact
  draw_rectangles: ReadinessFact
}

export type OpsPreviewType = "good" | "aug" | "noise"

export type OpsPreviewList = {
  dataset_type: DatasetType
  preview_type: OpsPreviewType
  images: string[]
}

/* defect_detection -------------------------------------------------------- */

export type DeviceStatus =
  | {
      device: "cpu"
      cuda_available: false
      message: string
    }
  | {
      device: "cuda"
      cuda_available: true
      device_name: string
      free_vram_gb: number
      total_vram_gb: number
      usage: number
    }

export type DefectReadiness = {
  dataset_type: DatasetType
  aug: ReadinessFact
  noise: ReadinessFact
  trained_networks: NetworkType[]
}

export type WeightRun = {
  run: string
  weights_file: string
}

export type WeightsList = {
  dataset_type: DatasetType
  network_type: NetworkType
  weights: WeightRun[]
}

export type DefectPreviewType =
  | "test"
  | "reconstruction"
  | "reconstruction_vis"
  | "roc_plot"

export type DefectPreviewList = {
  dataset_type: DatasetType
  preview_type: DefectPreviewType
  subtest_folder: SubtestFolder | null
  network_type: NetworkType | null
  images: string[]
}

/* Celery task lifecycle --------------------------------------------------- */

export type TaskState =
  | "QUEUED"
  | "PENDING"
  | "STARTED"
  | "PROGRESS"
  | "RETRY"
  | "SUCCESS"
  | "FAILURE"
  | "REVOKED"

export type QueuedTask = {
  task_id: string
  status: TaskState
}

export type TaskProgress = {
  status: string
}

export type TrainResult = {
  status: string
  network_type: NetworkType
  dataset_type: DatasetType
  best_valid_loss: number
  epochs_run: number
  weights_path: string | null
}

/* The test result shape depends on the mode: vis_reconstruction skips the
   metrics entirely and reports the written images instead. */
export type TestMetricsResult = {
  status: string
  network_type: NetworkType
  dataset_type: DatasetType
  subtest_folder: SubtestFolder
  auc_roc: number
  avg_ssim: number
  mse_avg: number
  metrics_path: string
  weights_used: string
}

export type TestReconstructionResult = {
  status: string
  network_type: NetworkType
  dataset_type: DatasetType
  reconstructed_images: number
  save_dir: string
  weights_used: string
}

export type TestResult = TestMetricsResult | TestReconstructionResult

export type TaskStatus<T> = {
  task_id: string
  status: TaskState
  info: T | TaskProgress | string | null
}

export type TrainStatus = TaskStatus<TrainResult>
export type TestStatus = TaskStatus<TestResult>

export type AbortResult = {
  status: string
  message: string
}

/* Training / testing request params --------------------------------------- */

export type TrainingParams = {
  dataset_type: DatasetType
  ae_type?: AEType
  model_size?: ModelSize
  validation_split?: number
  epochs?: number
  batch_size?: number
  learning_rate?: number
  decrease_learning_rate?: boolean
  step_size?: number
  gamma?: number
  grayscale?: boolean
  latent_space_dimension?: number
  vis_during_training?: boolean
  vis_interval?: number
  early_stopping?: number
  seed?: boolean
}

export type TestingParams = {
  dataset_type: DatasetType
  ae_type?: AEType
  model_size?: ModelSize
  subtest_folder?: SubtestFolder
  weights_run?: string
  stride?: Stride
  num_of_steps?: number
  threshold_init?: number
  threshold_end?: number
  vis_results?: boolean
  vis_reconstruction?: boolean
  vis_interval?: number
}