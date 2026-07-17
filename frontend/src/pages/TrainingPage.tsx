import { ArrowRight } from "lucide-react"
import { useState } from "react"
import { useNavigate } from "react-router-dom"

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
import { Button } from "../components/Button/Button"
import { JsonPanel } from "../components/JsonPanel/JsonPanel"
import { PageNav } from "../components/PageNav/PageNav"
import { Panel } from "../components/Panel/Panel"
import { ParamField } from "../components/ParamField/ParamField"
import { RunControls } from "../components/RunControls/RunControls"
import { StatusGrid } from "../components/StatusGrid/StatusGrid"
import { useReadiness } from "../hooks/useReadiness"
import type { TaskRunState } from "../hooks/useTaskPolling"
import { useTaskPolling } from "../hooks/useTaskPolling"
import { formatElapsedTime } from "../utils/format"
import styles from "./TrainingPage.module.css"

type OptionalBoolean = "" | "true" | "false"

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
  stopped: { variant: "warning", label: "Stopped" },
  error: { variant: "danger", label: "Failed" },
}

function toNumber(value: string): number | undefined {
  const trimmed = value.trim()

  if (trimmed === "") {
    return undefined
  }

  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : undefined
}

function toBoolean(value: OptionalBoolean): boolean | undefined {
  if (value === "") {
    return undefined
  }

  return value === "true"
}

function formatLoss(value: number | undefined): string | undefined {
  if (value === undefined) {
    return undefined
  }

  return Number.isFinite(value) ? value.toFixed(5) : String(value)
}

function optionLabel(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1)
}

export function TrainingPage() {
  const navigate = useNavigate()

  const [datasetType, setDatasetType] =
    useState<DatasetType>("texture_1")
  const [aeType, setAeType] = useState<AEType>("plain")
  const [modelSize, setModelSize] = useState<ModelSize>("base")

  const [validationSplit, setValidationSplit] = useState("")
  const [epochs, setEpochs] = useState("")
  const [batchSize, setBatchSize] = useState("")
  const [learningRate, setLearningRate] = useState("")
  const [decreaseLearningRate, setDecreaseLearningRate] =
    useState<OptionalBoolean>("")
  const [stepSize, setStepSize] = useState("")
  const [gamma, setGamma] = useState("")
  const [grayscale, setGrayscale] =
    useState<OptionalBoolean>("")
  const [latentSpaceDimension, setLatentSpaceDimension] = useState("")
  const [visDuringTraining, setVisDuringTraining] =
    useState<OptionalBoolean>("")
  const [visInterval, setVisInterval] = useState("")
  const [earlyStopping, setEarlyStopping] = useState("")
  const [seed, setSeed] = useState("")

  const [submittedNetwork, setSubmittedNetwork] =
    useState<NetworkType | null>(null)

  const readiness = useReadiness(getDefectReadiness, datasetType)

  const run = useTaskPolling<TrainingParams, TrainResult>(
    runTraining,
    getTrainingStatus,
    stopTraining,
  )

  const selectedNetwork = NETWORK_TYPE_MAP[aeType][modelSize]
  const result = run.result
  const badge = RUN_BADGE[run.state]

  let prerequisiteMessage: string | null = null
  let prerequisitePath: string | null = null
  let prerequisiteLabel: string | null = null

  if (readiness.data && !readiness.data.aug.ready) {
    prerequisiteMessage =
      "No augmentation output is available for this dataset."
    prerequisitePath = "/augmentation"
    prerequisiteLabel = "Go to augmentation"
  } else if (
    readiness.data &&
    aeType === "denoising" &&
    !readiness.data.noise.ready
  ) {
    prerequisiteMessage =
      "Denoising training requires covered images."
    prerequisitePath = "/draw-rectangles"
    prerequisiteLabel = "Go to draw rectangles"
  }

  const startDisabled =
    readiness.loading ||
    readiness.data === null ||
    readiness.error !== null ||
    prerequisiteMessage !== null

  const output =
    result ??
    (run.taskId
      ? {
          task_id: run.taskId,
          status: run.taskState ?? "QUEUED",
        }
      : null)

  function handleStart() {
    const params: TrainingParams = {
      dataset_type: datasetType,
      ae_type: aeType,
      model_size: modelSize,
      validation_split: toNumber(validationSplit),
      epochs: toNumber(epochs),
      batch_size: toNumber(batchSize),
      learning_rate: toNumber(learningRate),
      decrease_learning_rate: toBoolean(decreaseLearningRate),
      step_size: toNumber(stepSize),
      gamma: toNumber(gamma),
      grayscale: toBoolean(grayscale),
      latent_space_dimension: toNumber(latentSpaceDimension),
      vis_during_training: toBoolean(visDuringTraining),
      vis_interval: toNumber(visInterval),
      early_stopping: toNumber(earlyStopping),
      seed: toNumber(seed),
    }

    setSubmittedNetwork(selectedNetwork)
    void run.start(params)
  }

  return (
    <div className={styles.page}>
      <Breadcrumb step="training" />

      <header className={styles.intro}>
        <h1 className={styles.title}>Training</h1>
        <p className={styles.subtitle}>
          Trains an autoencoder using the latest augmentation output for the
          selected dataset.
        </p>
      </header>

      <div className={styles.columns}>
        <Panel title="Parameters" className={styles.params}>
          <div className={styles.fields}>
            <ParamField
              label="Dataset"
              tooltip="The dataset whose latest augmentation output will be used for training."
            >
              <select
                value={datasetType}
                onChange={(event) =>
                  setDatasetType(event.target.value as DatasetType)
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
              label="Autoencoder type"
              tooltip="Plain reconstructs its input. Denoising learns to reconstruct clean images from covered inputs."
            >
              <select
                value={aeType}
                onChange={(event) =>
                  setAeType(event.target.value as AEType)
                }
              >
                {AE_TYPES.map((value) => (
                  <option key={value} value={value}>
                    {optionLabel(value)}
                  </option>
                ))}
              </select>
            </ParamField>

            <ParamField
              label="Model size"
              tooltip="Base uses the standard architecture. Extended uses the deeper architecture."
            >
              <select
                value={modelSize}
                onChange={(event) =>
                  setModelSize(event.target.value as ModelSize)
                }
              >
                {MODEL_SIZES.map((value) => (
                  <option key={value} value={value}>
                    {optionLabel(value)}
                  </option>
                ))}
              </select>
            </ParamField>
          </div>

          <p className={styles.inherited}>
            Image size and crop size are inherited from the latest augmentation
            run.
          </p>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Optional overrides</h2>
            <p className={styles.sectionDescription}>
              Empty fields use the server defaults.
            </p>

            <div className={styles.fieldGrid}>
              <ParamField
                label="Validation split"
                tooltip="Fraction reserved for validation. Must be greater than 0 and less than 1. Example: 0.2."
              >
                <input
                  type="number"
                  min={0}
                  max={1}
                  step="any"
                  placeholder="Server default"
                  value={validationSplit}
                  onChange={(event) =>
                    setValidationSplit(event.target.value)
                  }
                />
              </ParamField>

              <ParamField
                label="Epochs"
                tooltip="Maximum number of training epochs. Minimum: 1. Example: 200."
              >
                <input
                  type="number"
                  min={1}
                  placeholder="Server default"
                  value={epochs}
                  onChange={(event) => setEpochs(event.target.value)}
                />
              </ParamField>

              <ParamField
                label="Batch size"
                tooltip="Number of images processed together. Minimum: 1. Example: 128."
              >
                <input
                  type="number"
                  min={1}
                  placeholder="Server default"
                  value={batchSize}
                  onChange={(event) => setBatchSize(event.target.value)}
                />
              </ParamField>

              <ParamField
                label="Learning rate"
                tooltip="Optimizer learning rate. Must be greater than 0. Example: 0.0002."
              >
                <input
                  type="number"
                  min={0}
                  step="any"
                  placeholder="Server default"
                  value={learningRate}
                  onChange={(event) =>
                    setLearningRate(event.target.value)
                  }
                />
              </ParamField>

              <ParamField
                label="Decrease learning rate"
                tooltip="Enables or disables the learning-rate scheduler."
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
                  <option value="true">Enabled</option>
                  <option value="false">Disabled</option>
                </select>
              </ParamField>

              <ParamField
                label="Scheduler step size"
                tooltip="Number of epochs between learning-rate reductions. Minimum: 1."
              >
                <input
                  type="number"
                  min={1}
                  placeholder="Server default"
                  value={stepSize}
                  onChange={(event) => setStepSize(event.target.value)}
                />
              </ParamField>

              <ParamField
                label="Gamma"
                tooltip="Learning-rate reduction factor. Must be greater than 0. Example: 0.5."
              >
                <input
                  type="number"
                  min={0}
                  step="any"
                  placeholder="Server default"
                  value={gamma}
                  onChange={(event) => setGamma(event.target.value)}
                />
              </ParamField>

              <ParamField
                label="Image channels"
                tooltip="Choose grayscale or RGB input, or leave the server default unchanged."
              >
                <select
                  value={grayscale}
                  onChange={(event) =>
                    setGrayscale(event.target.value as OptionalBoolean)
                  }
                >
                  <option value="">Server default</option>
                  <option value="true">Grayscale</option>
                  <option value="false">RGB</option>
                </select>
              </ParamField>

              <ParamField
                label="Latent dimension"
                tooltip="Number of latent-space channels. Minimum: 1. Example: 100."
              >
                <input
                  type="number"
                  min={1}
                  placeholder="Server default"
                  value={latentSpaceDimension}
                  onChange={(event) =>
                    setLatentSpaceDimension(event.target.value)
                  }
                />
              </ParamField>

              <ParamField
                label="Save visualizations"
                tooltip="Controls whether training reconstruction previews are saved."
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
                  <option value="true">Enabled</option>
                  <option value="false">Disabled</option>
                </select>
              </ParamField>

              <ParamField
                label="Visualization interval"
                tooltip="Number of epochs between saved visualizations. Minimum: 1."
              >
                <input
                  type="number"
                  min={1}
                  placeholder="Server default"
                  value={visInterval}
                  onChange={(event) =>
                    setVisInterval(event.target.value)
                  }
                />
              </ParamField>

              <ParamField
                label="Early stopping"
                tooltip="Number of non-improving epochs allowed before training stops. Minimum: 1."
              >
                <input
                  type="number"
                  min={1}
                  placeholder="Server default"
                  value={earlyStopping}
                  onChange={(event) =>
                    setEarlyStopping(event.target.value)
                  }
                />
              </ParamField>

              <ParamField
                label="Seed"
                tooltip="Optional random-seed setting. Minimum: 0."
              >
                <input
                  type="number"
                  min={0}
                  placeholder="Server default"
                  value={seed}
                  onChange={(event) => setSeed(event.target.value)}
                />
              </ParamField>
            </div>
          </section>

          {readiness.loading ? (
            <p className={styles.readiness}>
              Checking training prerequisites...
            </p>
          ) : null}

          {readiness.error ? (
            <p className={styles.error}>{readiness.error}</p>
          ) : null}

          {prerequisiteMessage &&
          prerequisitePath &&
          prerequisiteLabel ? (
            <div className={styles.prerequisite} role="alert">
              <p className={styles.prerequisiteText}>
                {prerequisiteMessage}
              </p>

              <Button
                icon={ArrowRight}
                iconPosition="right"
                onClick={() => navigate(prerequisitePath)}
              >
                {prerequisiteLabel}
              </Button>
            </div>
          ) : null}

          <RunControls
            className={styles.controls}
            running={run.state === "running"}
            stopping={run.stopping}
            disabled={startDisabled}
            onStart={handleStart}
            onStop={() => void run.stop()}
            startLabel="Start training"
          />
        </Panel>

        <div className={styles.results}>
          <Panel title="Status">
            <StatusGrid
              items={[
                {
                  label: "State",
                  value: (
                    <Badge variant={badge.variant}>
                      {badge.label}
                    </Badge>
                  ),
                },
                {
                  label: "Elapsed",
                  value: formatElapsedTime(run.elapsedMs),
                },
                {
                  label: "Task status",
                  value: run.taskState ?? undefined,
                },
                {
                  label: "Network",
                  value: submittedNetwork ?? undefined,
                },
                {
                  label: "Epochs run",
                  value: result?.epochs_run,
                },
                {
                  label: "Best valid loss",
                  value: formatLoss(result?.best_valid_loss),
                },
              ]}
            />

            {run.error ? (
              <p className={styles.error}>{run.error}</p>
            ) : null}
          </Panel>

          <JsonPanel value={output} />
        </div>
      </div>

      <PageNav step="training" />
    </div>
  )
}