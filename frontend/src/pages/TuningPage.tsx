import { useState } from "react"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { Link } from "react-router-dom"

import {
  getDefectReadiness,
  getTuningStatus,
  runTuning,
  stopTuning,
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
  TuningParams,
  TuningResult,
} from "../api/types"
import { Badge } from "../components/Badge/Badge"
import type { BadgeVariant } from "../components/Badge/Badge"
import { JsonPanel } from "../components/JsonPanel/JsonPanel"
import { Panel } from "../components/Panel/Panel"
import { ParamField } from "../components/ParamField/ParamField"
import { ProgressBar } from "../components/ProgressBar/ProgressBar"
import { RunControls } from "../components/RunControls/RunControls"
import { RunningNotice } from "../components/RunningNotice/RunningNotice"
import { StatusGrid } from "../components/StatusGrid/StatusGrid"
import {
  MAIN_MENU_LABEL,
  MAIN_MENU_PATH,
  getStep,
} from "../config/workflow"
import { useDeviceStatus } from "../hooks/useDeviceStatus"
import { useReadiness } from "../hooks/useReadiness"
import type { TaskRunState } from "../hooks/useTaskPolling"
import { useTaskPolling } from "../hooks/useTaskPolling"
import { formatElapsedTime } from "../utils/format"
import styles from "./TuningPage.module.css"

const TUNING_TASK_STORAGE_KEY = "defect-detection:tuning-task"

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

/* Best-parameter values come back raw. Integers stay integers. Very small
   floats (learning rate) read better in scientific form: 0.0001378124 becomes
   1.38e-4 instead of a long decimal. Everything else is rounded to 4
   significant figures with trailing zeros trimmed (gamma stays 0.9, 0.7345). */
function formatBestValue(value: number): string {
  if (Number.isInteger(value)) {
    return String(value)
  }

  if (Math.abs(value) < 0.001) {
    return value.toExponential(2)
  }

  return Number(value.toPrecision(4)).toString()
}

function resolveNetworkType(
  aeType: AEType,
  modelSize: ModelSize,
): NetworkType {
  return NETWORK_TYPE_MAP[aeType][modelSize]
}

export function TuningPage() {
  const task = useTaskPolling<TuningParams, TuningResult>(
    runTuning,
    getTuningStatus,
    stopTuning,
    {
      storageKey: TUNING_TASK_STORAGE_KEY,
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

  const [nTrials, setNTrials] = useState(
    toInputValue(restored?.n_trials),
  )
  const [epochsPerTrial, setEpochsPerTrial] = useState(
    toInputValue(restored?.epochs_per_trial),
  )
  const [learningRateMin, setLearningRateMin] = useState(
    toInputValue(restored?.learning_rate_min),
  )
  const [learningRateMax, setLearningRateMax] = useState(
    toInputValue(restored?.learning_rate_max),
  )
  const [latentMin, setLatentMin] = useState(
    toInputValue(restored?.latent_space_dimension_min),
  )
  const [latentMax, setLatentMax] = useState(
    toInputValue(restored?.latent_space_dimension_max),
  )
  const [stepMin, setStepMin] = useState(
    toInputValue(restored?.step_size_min),
  )
  const [stepMax, setStepMax] = useState(
    toInputValue(restored?.step_size_max),
  )
  const [gammaMin, setGammaMin] = useState(
    toInputValue(restored?.gamma_min),
  )
  const [gammaMax, setGammaMax] = useState(
    toInputValue(restored?.gamma_max),
  )
  const [batchMin, setBatchMin] = useState(
    toInputValue(restored?.batch_size_min),
  )
  const [batchMax, setBatchMax] = useState(
    toInputValue(restored?.batch_size_max),
  )

  const readiness = useReadiness(getDefectReadiness, datasetType)
  const deviceStatus = useDeviceStatus()

  const selectedNetworkType = resolveNetworkType(aeType, modelSize)

  const submittedNetworkType = task.submittedParams
    ? resolveNetworkType(
        task.submittedParams.ae_type ?? "plain",
        task.submittedParams.model_size ?? "base",
      )
    : selectedNetworkType

  const augmentationReady = readiness.data?.aug.ready === true
  const noiseReady = readiness.data?.noise.ready === true
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

  function handleStart() {
    const params: TuningParams = {
      dataset_type: datasetType,
      ae_type: aeType,
      model_size: modelSize,
      n_trials: toNumber(nTrials),
      epochs_per_trial: toNumber(epochsPerTrial),
      learning_rate_min: toNumber(learningRateMin),
      learning_rate_max: toNumber(learningRateMax),
      latent_space_dimension_min: toNumber(latentMin),
      latent_space_dimension_max: toNumber(latentMax),
      step_size_min: toNumber(stepMin),
      step_size_max: toNumber(stepMax),
      gamma_min: toNumber(gammaMin),
      gamma_max: toNumber(gammaMax),
      batch_size_min: toNumber(batchMin),
      batch_size_max: toNumber(batchMax),
    }

    void task.start(params)
  }

  const isRunning = task.state === "running"
  const isDone = task.state === "done"

  const badge = RUN_BADGE[task.state]

  const showProgress = isRunning || (isDone && task.progress !== null)
  const progressVariant: "accent" | "success" = isDone
    ? "success"
    : "accent"

  // On completion the final "all trials done" progress poll can be missed
  // (SUCCESS lands first), leaving the bar at 99%. Force it full when done.
  const progressTotal = task.progress?.total
  const progressCurrent = isDone ? progressTotal : task.progress?.current

  const bestParams = isDone ? task.result?.best_params : undefined
  const bestValidLoss = task.result?.best_valid_loss

  const statusOutput = task.taskId
    ? {
        task_id: task.taskId,
        status: task.taskState,
        info: task.result ?? task.error,
      }
    : null

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
      <RunningNotice running={isRunning} label="Tuning" />

      <header className={styles.intro}>
        <h1 className={styles.title}>Parameter Tuning</h1>
        <p className={styles.subtitle}>
          How well a model trains depends on a few settings, such as how fast
          it learns. Instead of guessing them by hand, this page trains many
          short test models with different settings and tells you which
          combination worked best. It only searches for good settings and does
          not save a finished model. Number fields can be left empty to use
          their default values.
        </p>
      </header>

      <div className={styles.columns}>
        <Panel title="Search setup" className={styles.params}>
          <div className={styles.fields}>
            <ParamField
              label="dataset_type"
              tooltip="Which set of images to run the search on."
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
              label="ae_type"
              tooltip="The kind of model. 'plain' just rebuilds the image; 'denoising' also learns to remove added noise, and needs the noisy images prepared first."
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
              tooltip="How big the model is. 'base' is smaller and faster; 'extended' is larger and slower but can capture more detail."
            >
              <select
                value={modelSize}
                onChange={(event) =>
                  setModelSize(event.target.value as ModelSize)
                }
              >
                {MODEL_SIZES.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </ParamField>

            <h3>How much to search</h3>
            <p>
              More attempts and longer training usually find better settings,
              but take longer to finish.
            </p>

            <ParamField
              label="n_trials"
              tooltip="How many different setting combinations to try. More gives a better result but takes longer. A whole number, for example 20."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 20"
                value={nTrials}
                onChange={(event) => setNTrials(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="epochs_per_trial"
              tooltip="How long each attempt trains. One epoch is one pass over all the images. Keep it small so the search stays quick. A whole number of at least 2, for example 5."
            >
              <input
                type="number"
                min={2}
                step={1}
                placeholder="e.g. 5"
                value={epochsPerTrial}
                onChange={(event) => setEpochsPerTrial(event.target.value)}
              />
            </ParamField>

            <h3>Value ranges to try</h3>
            <p>
              For each setting below, the search picks a value between the
              smallest and largest you enter. Leave a pair empty to use the
              server defaults.
            </p>

            <ParamField
              label="learning_rate_min"
              tooltip="The learning rate is how fast the model learns. This is the smallest value to try. A decimal above 0, for example 0.00001."
            >
              <input
                type="number"
                min={0}
                step="any"
                placeholder="e.g. 0.00001"
                value={learningRateMin}
                onChange={(event) => setLearningRateMin(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="learning_rate_max"
              tooltip="The largest learning rate to try. A decimal above 0, for example 0.01."
            >
              <input
                type="number"
                min={0}
                step="any"
                placeholder="e.g. 0.01"
                value={learningRateMax}
                onChange={(event) => setLearningRateMax(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="latent_space_dimension_min"
              tooltip="The latent size is how much the model squeezes each image down. This is the smallest value to try. A whole number, for example 16."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 16"
                value={latentMin}
                onChange={(event) => setLatentMin(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="latent_space_dimension_max"
              tooltip="The largest latent size to try. A whole number, for example 256."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 256"
                value={latentMax}
                onChange={(event) => setLatentMax(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="step_size_min"
              tooltip="The model eases off its learning speed every so many epochs. This is the fewest epochs between those slow-downs to try. A whole number, for example 1."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 1"
                value={stepMin}
                onChange={(event) => setStepMin(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="step_size_max"
              tooltip="The most epochs between slow-downs to try. This must be smaller than epochs_per_trial. A whole number, for example 4."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 4"
                value={stepMax}
                onChange={(event) => setStepMax(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="gamma_min"
              tooltip="How hard the model eases off its learning at each slow-down. Use a value above 0 and up to 1, where smaller means a bigger drop. This is the smallest value to try. For example 0.1."
            >
              <input
                type="number"
                min={0}
                max={1}
                step="any"
                placeholder="e.g. 0.1"
                value={gammaMin}
                onChange={(event) => setGammaMin(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="gamma_max"
              tooltip="The largest easing value to try. Above 0 and up to 1. For example 0.9."
            >
              <input
                type="number"
                min={0}
                max={1}
                step="any"
                placeholder="e.g. 0.9"
                value={gammaMax}
                onChange={(event) => setGammaMax(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="batch_size_min"
              tooltip="How many images the model looks at together at once. This is the smallest value to try. A whole number, for example 16."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 16"
                value={batchMin}
                onChange={(event) => setBatchMin(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="batch_size_max"
              tooltip="The largest number of images at once to try. A whole number, for example 256."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 256"
                value={batchMax}
                onChange={(event) => setBatchMax(event.target.value)}
              />
            </ParamField>
          </div>

          {readiness.loading ? (
            <p>Checking what is ready for this dataset...</p>
          ) : null}

          {readiness.error ? (
            <p className={styles.error}>
              Could not check what is ready: {readiness.error}
            </p>
          ) : null}

          {!readiness.loading &&
          readiness.data &&
          !augmentationReady ? (
            <p className={styles.error}>
              This dataset has no prepared images yet. Prepare them first, then
              come back. <Link to="/augmentation">Open augmentation</Link>
            </p>
          ) : null}

          {!readiness.loading &&
          readiness.data &&
          requiresNoise &&
          !noiseReady ? (
            <p className={styles.error}>
              The denoising model needs noisy images, and none have been made
              yet. <Link to="/draw-rectangles">Open draw rectangles</Link>
            </p>
          ) : null}

          {!readiness.loading && countsMismatch ? (
            <p className={styles.error}>
              The prepared images ({augImages}) and noisy images ({noiseImages})
              do not match. Denoising needs the same number of each.{" "}
              <Link to="/draw-rectangles">Open draw rectangles</Link>
            </p>
          ) : null}

          <RunControls
            className={styles.controls}
            running={isRunning}
            stopping={task.stopping}
            onStart={handleStart}
            onStop={() => void task.stop()}
            startLabel={`Start tuning (${selectedNetworkType})`}
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
                    <Badge variant={badge.variant}>{badge.label}</Badge>
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
                  value: submittedNetworkType,
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
                          <span className={styles.mismatch}>{augImages}</span>
                        ) : (
                          augImages
                        ),
                      },
                      {
                        label: "noise_images",
                        value: countsMismatch ? (
                          <span className={styles.mismatch}>{noiseImages}</span>
                        ) : (
                          noiseImages
                        ),
                      },
                    ]
                  : []),
                {
                  label: "best_valid_loss",
                  value:
                    bestValidLoss === undefined
                      ? undefined
                      : bestValidLoss.toFixed(6),
                },
              ]}
            />

            {showProgress ? (
              <ProgressBar
                className={styles.progress}
                current={progressCurrent}
                total={progressTotal}
                phase={task.progress?.phase}
                variant={progressVariant}
              />
            ) : null}

            {usingCpu ? (
              <p className={styles.error} role="alert">
                No GPU was found, so the search runs on the CPU and will be much
                slower.
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

          {bestParams ? (
            <Panel title="Best settings found">
              <StatusGrid
                items={[
                  {
                    label: "learning_rate",
                    value: formatBestValue(bestParams.learning_rate),
                  },
                  {
                    label: "latent_space_dimension",
                    value: formatBestValue(bestParams.latent_space_dimension),
                  },
                  {
                    label: "step_size",
                    value: formatBestValue(bestParams.step_size),
                  },
                  {
                    label: "gamma",
                    value: formatBestValue(bestParams.gamma),
                  },
                  {
                    label: "batch_size",
                    value: formatBestValue(bestParams.batch_size),
                  },
                  {
                    label: "best_valid_loss",
                    value:
                      bestValidLoss === undefined
                        ? undefined
                        : bestValidLoss.toFixed(6),
                  },
                ]}
              />
            </Panel>
          ) : null}

          <JsonPanel value={statusOutput} />
        </div>
      </div>

      <nav className={styles.nav} aria-label="Tuning navigation">
        <Link className={styles.navLink} to={MAIN_MENU_PATH}>
          <ChevronLeft className={styles.navIcon} aria-hidden="true" />
          <span className={styles.navLabel}>{MAIN_MENU_LABEL}</span>
        </Link>

        <Link
          className={`${styles.navLink} ${styles.navNext}`}
          to={getStep("training").path}
        >
          <span className={styles.navLabel}>Training</span>
          <ChevronRight className={styles.navIcon} aria-hidden="true" />
        </Link>
      </nav>
    </div>
  )
}