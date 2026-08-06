import { useEffect, useState } from "react"
import { Link } from "react-router-dom"

import {
  getDefectReadiness,
  getTrainingStatus,
  runTraining,
  stopTraining,
} from "../api/defectDetection"
import {
  AE_TYPES,
  DATASET_TYPES,
  MODEL_SIZES,
} from "../api/types"
import type {
  AEType,
  DatasetType,
  ModelSize,
  NetworkType,
  TrainResult,
  TrainingParams,
} from "../api/types"
import { Badge } from "../components/Badge/Badge"
import type { BadgeVariant } from "../components/Badge/Badge"
import { Breadcrumb } from "../components/Breadcrumb/Breadcrumb"
import { JsonPanel } from "../components/JsonPanel/JsonPanel"
import { PageNav } from "../components/PageNav/PageNav"
import { Panel } from "../components/Panel/Panel"
import { ParamField } from "../components/ParamField/ParamField"
import { RunControls } from "../components/RunControls/RunControls"
import { RunningNotice } from "../components/RunningNotice/RunningNotice"
import { StatusGrid } from "../components/StatusGrid/StatusGrid"
import { useDeviceStatus } from "../hooks/useDeviceStatus"
import type { TaskRunState } from "../hooks/useTaskPolling"
import { useTaskPolling } from "../hooks/useTaskPolling"
import { useReadiness } from "../hooks/useReadiness"
import { formatElapsedTime } from "../utils/format"
import styles from "./TrainingPage.module.css"

const TRAINING_TASK_STORAGE_KEY = "defect-detection:training-task"

const NETWORK_TYPE_MAP: Record<
  AEType,
  Record<ModelSize, NetworkType>
> = {
  plain: {
    base: "AE",
    extended: "AEE",
  },
  denoising: {
    base: "DAE",
    extended: "DAEE",
  },
}

const RUN_BADGE: Record<
  TaskRunState,
  { variant: BadgeVariant; label: string }
> = {
  idle: { variant: "neutral", label: "Idle" },
  running: { variant: "accent", label: "Running" },
  done: { variant: "success", label: "Done" },
  error: { variant: "danger", label: "Failed" },
  stopped: { variant: "warning", label: "Stopped" },
}

type OptionalBoolean = "" | "true" | "false"

function toNumber(value: string): number | undefined {
  const trimmed = value.trim()

  if (trimmed === "") {
    return undefined
  }

  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : undefined
}

function toInputValue(value: number | undefined): string {
  return value === undefined ? "" : String(value)
}

function toBooleanField(value: boolean | undefined): OptionalBoolean {
  if (value === true) {
    return "true"
  }

  if (value === false) {
    return "false"
  }

  return ""
}

function toOptionalBoolean(
  value: OptionalBoolean,
): boolean | undefined {
  if (value === "true") {
    return true
  }

  if (value === "false") {
    return false
  }

  return undefined
}

function resolveNetworkType(
  aeType: AEType,
  modelSize: ModelSize,
): NetworkType {
  return NETWORK_TYPE_MAP[aeType][modelSize]
}

export function TrainingPage() {
  const task = useTaskPolling<TrainingParams, TrainResult>(
    runTraining,
    getTrainingStatus,
    stopTraining,
    {
      storageKey: TRAINING_TASK_STORAGE_KEY,
      intervalMs: 2000,
    },
  )

  const restored = task.submittedParams

  const [datasetType, setDatasetType] = useState<DatasetType>(
    restored?.dataset_type ?? "texture_1",
  )
  const [aeType, setAeType] = useState<AEType>(
    restored?.ae_type ?? "plain",
  )
  const [modelSize, setModelSize] = useState<ModelSize>(
    restored?.model_size ?? "base",
  )

  const [validationSplit, setValidationSplit] = useState(
    toInputValue(restored?.validation_split),
  )
  const [epochs, setEpochs] = useState(
    toInputValue(restored?.epochs),
  )
  const [batchSize, setBatchSize] = useState(
    toInputValue(restored?.batch_size),
  )
  const [learningRate, setLearningRate] = useState(
    toInputValue(restored?.learning_rate),
  )
  const [decreaseLearningRate, setDecreaseLearningRate] =
    useState<OptionalBoolean>(
      toBooleanField(restored?.decrease_learning_rate),
    )
  const [stepSize, setStepSize] = useState(
    toInputValue(restored?.step_size),
  )
  const [gamma, setGamma] = useState(
    toInputValue(restored?.gamma),
  )
  const [grayscale, setGrayscale] = useState<OptionalBoolean>(
    toBooleanField(restored?.grayscale),
  )
  const [latentSpaceDimension, setLatentSpaceDimension] = useState(
    toInputValue(restored?.latent_space_dimension),
  )
  const [visDuringTraining, setVisDuringTraining] =
    useState<OptionalBoolean>(
      toBooleanField(restored?.vis_during_training),
    )
  const [visInterval, setVisInterval] = useState(
    toInputValue(restored?.vis_interval),
  )
  const [earlyStopping, setEarlyStopping] = useState(
    toInputValue(restored?.early_stopping),
  )
  const [seed, setSeed] = useState<OptionalBoolean>(
    toBooleanField(restored?.seed),
  )

  const readiness = useReadiness(
    getDefectReadiness,
    datasetType,
  )

  const deviceStatus = useDeviceStatus()

  const selectedNetworkType = resolveNetworkType(
    aeType,
    modelSize,
  )

  const submittedNetworkType = task.submittedParams
    ? resolveNetworkType(
        task.submittedParams.ae_type ?? "plain",
        task.submittedParams.model_size ?? "base",
      )
    : selectedNetworkType

  const augmentationReady =
    readiness.data?.aug.ready === true

  const noiseReady =
    readiness.data?.noise.ready === true

  const requiresNoise = aeType === "denoising"

  const augImages = readiness.data?.aug.images
  const noiseImages = readiness.data?.noise.images

  const countsMismatch =
    requiresNoise &&
    augImages !== undefined &&
    noiseImages !== undefined &&
    augImages !== noiseImages

  const showDenoiseCounts =
    requiresNoise && readiness.data != null

  const prerequisitesReady =
    augmentationReady &&
    (!requiresNoise || (noiseReady && !countsMismatch))

  const startDisabled =
    readiness.loading ||
    readiness.error !== null ||
    !prerequisitesReady

  useEffect(() => {
    if (
      task.state === "done" &&
      task.submittedParams?.dataset_type === datasetType
    ) {
      readiness.reload()
    }
  }, [
    task.state,
    task.submittedParams,
    datasetType,
    readiness.reload,
  ])

  function handleStart() {
    const params: TrainingParams = {
      dataset_type: datasetType,
      ae_type: aeType,
      model_size: modelSize,
      validation_split: toNumber(validationSplit),
      epochs: toNumber(epochs),
      batch_size: toNumber(batchSize),
      learning_rate: toNumber(learningRate),
      decrease_learning_rate: toOptionalBoolean(
        decreaseLearningRate,
      ),
      step_size: toNumber(stepSize),
      gamma: toNumber(gamma),
      grayscale: toOptionalBoolean(grayscale),
      latent_space_dimension: toNumber(
        latentSpaceDimension,
      ),
      vis_during_training: toOptionalBoolean(
        visDuringTraining,
      ),
      vis_interval: toNumber(visInterval),
      early_stopping: toNumber(earlyStopping),
      seed: toOptionalBoolean(seed),
    }

    void task.start(params)
  }

  const badge = RUN_BADGE[task.state]

  const statusOutput = task.taskId
    ? {
        task_id: task.taskId,
        status: task.taskState,
        info: task.result ?? task.error,
      }
    : null

  const trainedDenoising = task.submittedParams
    ? task.submittedParams.ae_type === "denoising"
    : aeType === "denoising"

  const sourceLabel = task.result
    ? trainedDenoising
      ? "aug + noise"
      : "aug"
    : undefined

  const targetLabel = task.result ? "weights" : undefined

  const deviceBadge = deviceStatus.loading ? (
    "Checking..."
  ) : deviceStatus.error ? (
    "Unavailable"
  ) : deviceStatus.data?.device === "cuda" ? (
    <Badge variant="success">GPU</Badge>
  ) : deviceStatus.data?.device === "cpu" ? (
    <Badge variant="danger">CPU</Badge>
  ) : undefined

  const usingCpu = deviceStatus.data?.device === "cpu"

  return (
    <div className={styles.page}>
      <RunningNotice running={task.state === "running"} label="Training" />

      <Breadcrumb step="training" />

      <header className={styles.intro}>
        <h1 className={styles.title}>Training</h1>
        <p className={styles.subtitle}>
          Trains the selected autoencoder using the latest augmentation
          output. Image and crop sizes are inherited from that run.
        </p>
      </header>

      <div className={styles.columns}>
        <Panel title="Parameters" className={styles.params}>
          <div className={styles.fields}>
            <ParamField
              label="dataset_type"
              tooltip="Dataset used for training. Allowed values: texture_1, texture_2, cpu."
            >
              <select
                value={datasetType}
                onChange={(event) =>
                  setDatasetType(
                    event.target.value as DatasetType,
                  )
                }
              >
                {DATASET_TYPES.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </ParamField>

            <ParamField
              label="ae_type"
              tooltip="plain trains AE/AEE. denoising trains DAE/DAEE and also requires noise images."
            >
              <select
                value={aeType}
                onChange={(event) =>
                  setAeType(event.target.value as AEType)
                }
              >
                {AE_TYPES.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </ParamField>

            <ParamField
              label="model_size"
              tooltip="base selects the standard architecture. extended selects the deeper architecture."
            >
              <select
                value={modelSize}
                onChange={(event) =>
                  setModelSize(
                    event.target.value as ModelSize,
                  )
                }
              >
                {MODEL_SIZES.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </ParamField>

            <h3>Optional overrides</h3>
            <p>
              Empty fields use the values from
              training_config.json.
            </p>

            <ParamField
              label="validation_split"
              hint="Server default when empty."
              tooltip="Fraction used for validation. Must be greater than 0 and less than 1. Example: 0.2."
            >
              <input
                type="number"
                min={0}
                max={1}
                step="0.01"
                placeholder="e.g. 0.2"
                value={validationSplit}
                onChange={(event) =>
                  setValidationSplit(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="epochs"
              hint="Server default when empty."
              tooltip="Maximum number of training epochs. Minimum: 1. Example: 200."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 200"
                value={epochs}
                onChange={(event) =>
                  setEpochs(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="batch_size"
              hint="Server default when empty."
              tooltip="Number of images processed in one training batch. Minimum: 1. Example: 128."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 128"
                value={batchSize}
                onChange={(event) =>
                  setBatchSize(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="learning_rate"
              hint="Server default when empty."
              tooltip="Optimizer learning rate. Must be greater than 0. Example: 0.0002."
            >
              <input
                type="number"
                min={0}
                step="any"
                placeholder="e.g. 0.0002"
                value={learningRate}
                onChange={(event) =>
                  setLearningRate(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="decrease_learning_rate"
              tooltip="Controls whether StepLR decreases the learning rate during training."
            >
              <select
                value={decreaseLearningRate}
                onChange={(event) =>
                  setDecreaseLearningRate(
                    event.target.value as OptionalBoolean,
                  )
                }
              >
                <option value="">Server default</option>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </ParamField>

            <ParamField
              label="step_size"
              hint="Server default when empty."
              tooltip="Number of epochs between learning-rate reductions. Minimum: 1. Example: 15."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 15"
                value={stepSize}
                onChange={(event) =>
                  setStepSize(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="gamma"
              hint="Server default when empty."
              tooltip="Learning-rate multiplication factor. Must be greater than 0. Example: 0.5."
            >
              <input
                type="number"
                min={0}
                step="any"
                placeholder="e.g. 0.5"
                value={gamma}
                onChange={(event) =>
                  setGamma(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="grayscale"
              tooltip="true uses one image channel. false uses RGB input."
            >
              <select
                value={grayscale}
                onChange={(event) =>
                  setGrayscale(
                    event.target.value as OptionalBoolean,
                  )
                }
              >
                <option value="">Server default</option>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </ParamField>

            <ParamField
              label="latent_space_dimension"
              hint="Server default when empty."
              tooltip="Number of channels in the latent representation. Minimum: 1. Example: 100."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 100"
                value={latentSpaceDimension}
                onChange={(event) =>
                  setLatentSpaceDimension(
                    event.target.value,
                  )
                }
              />
            </ParamField>

            <ParamField
              label="vis_during_training"
              tooltip="Controls whether reconstruction examples are saved during training."
            >
              <select
                value={visDuringTraining}
                onChange={(event) =>
                  setVisDuringTraining(
                    event.target.value as OptionalBoolean,
                  )
                }
              >
                <option value="">Server default</option>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </ParamField>

            <ParamField
              label="vis_interval"
              hint="Server default when empty."
              tooltip="Visualization interval in epochs. Minimum: 1. Example: 10."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 10"
                value={visInterval}
                onChange={(event) =>
                  setVisInterval(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="early_stopping"
              hint="Server default when empty."
              tooltip="Number of non-improving epochs allowed before training stops. Minimum: 1. Example: 10."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 10"
                value={earlyStopping}
                onChange={(event) =>
                  setEarlyStopping(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="seed"
              tooltip="true fixes the random seed so runs are reproducible. false lets it vary."
            >
              <select
                value={seed}
                onChange={(event) =>
                  setSeed(
                    event.target.value as OptionalBoolean,
                  )
                }
              >
                <option value="">Server default</option>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </ParamField>
          </div>

          {readiness.loading ? (
            <p>Checking training prerequisites...</p>
          ) : null}

          {readiness.error ? (
            <p className={styles.error}>
              Readiness check failed: {readiness.error}
            </p>
          ) : null}

          {!readiness.loading &&
          readiness.data &&
          !augmentationReady ? (
            <p className={styles.error}>
              No augmentation output is available for this dataset.{" "}
              <Link to="/augmentation">Open augmentation</Link>
            </p>
          ) : null}

          {!readiness.loading &&
          readiness.data &&
          requiresNoise &&
          !noiseReady ? (
            <p className={styles.error}>
              Denoising training requires covered noise images.{" "}
              <Link to="/draw-rectangles">
                Open draw rectangles
              </Link>
            </p>
          ) : null}

          {!readiness.loading && countsMismatch ? (
            <p className={styles.error}>
              Augmented ({augImages}) and noise ({noiseImages}) image
              counts must match for denoising.{" "}
              <Link to="/draw-rectangles">
                Open draw rectangles
              </Link>
            </p>
          ) : null}

          <RunControls
            className={styles.controls}
            running={task.state === "running"}
            stopping={task.stopping}
            onStart={handleStart}
            onStop={() => void task.stop()}
            startLabel={`Start training (${selectedNetworkType})`}
            disabled={startDisabled}
          />
        </Panel>

        <div className={styles.results}>
          <Panel title="Status">
            <StatusGrid
              items={[
                {
                  label: "state",
                  value: (
                    <Badge variant={badge.variant}>
                      {badge.label}
                    </Badge>
                  ),
                },
                {
                  label: "elapsed",
                  value: formatElapsedTime(task.elapsedMs),
                },
                {
                  label: "task_status",
                  value: task.taskState,
                },
                {
                  label: "network_type",
                  value:
                    task.result?.network_type ??
                    submittedNetworkType,
                },
                {
                  label: "device",
                  value: deviceBadge,
                },
                ...(showDenoiseCounts
                  ? [
                      {
                        label: "aug_images",
                        value: countsMismatch ? (
                          <span className={styles.mismatch}>
                            {augImages}
                          </span>
                        ) : (
                          augImages
                        ),
                      },
                      {
                        label: "noise_images",
                        value: countsMismatch ? (
                          <span className={styles.mismatch}>
                            {noiseImages}
                          </span>
                        ) : (
                          noiseImages
                        ),
                      },
                    ]
                  : []),
                {
                  label: "source",
                  value: sourceLabel,
                },
                {
                  label: "target",
                  value: targetLabel,
                },
                {
                  label: "epochs_run",
                  value: task.result?.epochs_run,
                },
                {
                  label: "best_valid_loss",
                  value:
                    task.result?.best_valid_loss === undefined
                      ? undefined
                      : task.result.best_valid_loss.toFixed(6),
                },
              ]}
            />

            {usingCpu ? (
              <p className={styles.error} role="alert">
                No GPU available. Training will use the CPU and may take
                significantly longer.
              </p>
            ) : null}

            {task.taskId ? (
              <p title={task.taskId}>
                task_id: <code>{task.taskId}</code>
              </p>
            ) : null}

            {task.error ? (
              <p className={styles.error}>{task.error}</p>
            ) : null}
          </Panel>

          <JsonPanel value={statusOutput} />
        </div>
      </div>

      <PageNav step="training" />
    </div>
  )
}