import { ArrowRight } from "lucide-react"
import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"

import {
  getDefectReadiness,
  getTestingStatus,
  runTesting,
  stopTesting,
} from "../api/defectDetection"
import {
  AE_TYPES,
  DATASET_TYPES,
  LIMITS,
  MODEL_SIZES,
  STRIDES,
  VALID_SUBTESTS,
} from "../api/types"
import type {
  AEType,
  DatasetType,
  DefectPreviewType,
  ModelSize,
  NetworkType,
  Stride,
  SubtestFolder,
  TestMetricsResult,
  TestReconstructionResult,
  TestResult,
  TestingParams,
} from "../api/types"
import { Badge } from "../components/Badge/Badge"
import type { BadgeVariant } from "../components/Badge/Badge"
import { Breadcrumb } from "../components/Breadcrumb/Breadcrumb"
import { Button } from "../components/Button/Button"
import { JsonPanel } from "../components/JsonPanel/JsonPanel"
import { PageNav } from "../components/PageNav/PageNav"
import { Panel } from "../components/Panel/Panel"
import { ParamField } from "../components/ParamField/ParamField"
import { PreviewGrid } from "../components/PreviewGrid/PreviewGrid"
import { RunControls } from "../components/RunControls/RunControls"
import { StatusGrid } from "../components/StatusGrid/StatusGrid"
import type { StatusItem } from "../components/StatusGrid/StatusGrid"
import { useDefectPreview } from "../hooks/usePreview"
import { useReadiness } from "../hooks/useReadiness"
import type { TaskRunState } from "../hooks/useTaskPolling"
import { useTaskPolling } from "../hooks/useTaskPolling"
import { formatElapsedTime } from "../utils/format"
import styles from "./TestingPage.module.css"

type OptionalBoolean = "" | "true" | "false"

type SubmittedTest = {
  datasetType: DatasetType
  networkType: NetworkType
  subtestFolder: SubtestFolder
  visResults: boolean
}

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

function optionLabel(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function formatMetric(
  value: number | undefined,
  digits: number,
): string | undefined {
  if (value === undefined) {
    return undefined
  }

  return Number.isFinite(value) ? value.toFixed(digits) : String(value)
}

function isMetricsResult(
  result: TestResult | null,
): result is TestMetricsResult {
  return result !== null && "auc_roc" in result
}

function isReconstructionResult(
  result: TestResult | null,
): result is TestReconstructionResult {
  return result !== null && "reconstructed_images" in result
}

export function TestingPage() {
  const navigate = useNavigate()

  const [datasetType, setDatasetType] =
    useState<DatasetType>("texture_1")
  const [aeType, setAeType] = useState<AEType>("plain")
  const [modelSize, setModelSize] = useState<ModelSize>("base")
  const [subtestFolder, setSubtestFolder] =
    useState<SubtestFolder>("defective")
  const [stride, setStride] = useState<Stride>(32)

  const [numOfSteps, setNumOfSteps] = useState("")
  const [thresholdInit, setThresholdInit] = useState("")
  const [thresholdEnd, setThresholdEnd] = useState("")
  const [visResults, setVisResults] =
    useState<OptionalBoolean>("")
  const [visReconstruction, setVisReconstruction] =
    useState<OptionalBoolean>("")
  const [visInterval, setVisInterval] = useState("")

  const [submitted, setSubmitted] =
    useState<SubmittedTest | null>(null)

  const readiness = useReadiness(getDefectReadiness, datasetType)

  const run = useTaskPolling<TestingParams, TestResult>(
    runTesting,
    getTestingStatus,
    stopTesting,
  )

  useEffect(() => {
    const allowed = VALID_SUBTESTS[datasetType]

    if (!allowed.includes(subtestFolder)) {
      const firstAllowed = allowed[0]

      if (firstAllowed) {
        setSubtestFolder(firstAllowed)
      }
    }
  }, [datasetType, subtestFolder])

  const selectedNetwork = NETWORK_TYPE_MAP[aeType][modelSize]
  const result = run.result
  const badge = RUN_BADGE[run.state]

  const hasSelectedModel =
    readiness.data?.trained_networks.includes(selectedNetwork) ?? false

  const thresholdInitValue = toNumber(thresholdInit)
  const thresholdEndValue = toNumber(thresholdEnd)

  let configurationError: string | null = null

  if (
    visResults === "true" &&
    visReconstruction === "true"
  ) {
    configurationError =
      "Result visualizations and reconstruction-only mode cannot both be enabled."
  } else if (
    thresholdInitValue !== undefined &&
    thresholdEndValue !== undefined &&
    thresholdInitValue >= thresholdEndValue
  ) {
    configurationError =
      "Threshold start must be smaller than threshold end."
  }

  const prerequisiteMessage =
    readiness.data && !hasSelectedModel
      ? `No trained ${selectedNetwork} model is available for ${datasetType}.`
      : null

  const startDisabled =
    readiness.loading ||
    readiness.data === null ||
    readiness.error !== null ||
    prerequisiteMessage !== null ||
    configurationError !== null

  const output =
    result ??
    (run.taskId
      ? {
          task_id: run.taskId,
          status: run.taskState ?? "QUEUED",
        }
      : null)

  const statusItems: StatusItem[] = [
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
      value:
        result?.network_type ??
        submitted?.networkType ??
        undefined,
    },
  ]

  if (isMetricsResult(result)) {
    statusItems.push(
      {
        label: "ROC AUC",
        value: formatMetric(result.auc_roc, 4),
      },
      {
        label: "Average SSIM",
        value: formatMetric(result.avg_ssim, 4),
      },
      {
        label: "Average MSE",
        value: formatMetric(result.mse_avg, 2),
      },
    )
  }

  if (isReconstructionResult(result)) {
    statusItems.push({
      label: "Reconstructed images",
      value: result.reconstructed_images,
    })
  }

  let previewDataset = datasetType
  let previewSubtest = subtestFolder
  let previewNetwork = selectedNetwork
  let previewType: DefectPreviewType = "test"
  let previewTitle = "Test images"

  if (result && submitted) {
    previewDataset = result.dataset_type
    previewNetwork = result.network_type

    if (isReconstructionResult(result)) {
      previewType = "reconstruction"
      previewTitle = "Reconstructed images"
      previewSubtest = submitted.subtestFolder
    } else {
      previewSubtest = result.subtest_folder

      if (submitted.visResults) {
        previewType = "reconstruction_vis"
        previewTitle = "Result visualizations"
      } else {
        previewType = "roc_plot"
        previewTitle = "ROC plot"
      }
    }
  }

  const preview = useDefectPreview(
    previewDataset,
    previewType,
    previewType === "test"
      ? { subtestFolder: previewSubtest }
      : { networkType: previewNetwork },
  )

  function handleStart() {
    const params: TestingParams = {
      dataset_type: datasetType,
      ae_type: aeType,
      model_size: modelSize,
      subtest_folder: subtestFolder,
      stride,
      num_of_steps: toNumber(numOfSteps),
      threshold_init: thresholdInitValue,
      threshold_end: thresholdEndValue,
      vis_results: toBoolean(visResults),
      vis_reconstruction: toBoolean(visReconstruction),
      vis_interval: toNumber(visInterval),
    }

    setSubmitted({
      datasetType,
      networkType: selectedNetwork,
      subtestFolder,
      visResults: toBoolean(visResults) ?? false,
    })

    void run.start(params)
  }

  return (
    <div className={styles.page}>
      <Breadcrumb step="testing" />

      <header className={styles.intro}>
        <h1 className={styles.title}>Testing</h1>
        <p className={styles.subtitle}>
          Evaluates a trained autoencoder and visualizes detected defects.
        </p>
      </header>

      <div className={styles.columns}>
        <Panel title="Parameters" className={styles.params}>
          <div className={styles.fields}>
            <ParamField
              label="Dataset"
              tooltip="The dataset whose test images and trained weights will be used."
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
              tooltip="Must match the type of the trained model."
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
              tooltip="Must match the architecture of the trained model."
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

            <ParamField
              label="Test subset"
              tooltip="Texture datasets use defective. CPU uses added, contamination, or missing."
            >
              <select
                value={subtestFolder}
                onChange={(event) =>
                  setSubtestFolder(
                    event.target.value as SubtestFolder,
                  )
                }
              >
                {VALID_SUBTESTS[datasetType].map((value) => (
                  <option key={value} value={value}>
                    {optionLabel(value)}
                  </option>
                ))}
              </select>
            </ParamField>

            <ParamField
              label="Stride"
              tooltip="Sliding-window stride. Smaller values create more overlapping patches and take longer."
            >
              <select
                value={stride}
                onChange={(event) =>
                  setStride(Number(event.target.value) as Stride)
                }
              >
                {STRIDES.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </ParamField>
          </div>

          <p className={styles.inherited}>
            Image size, crop size, grayscale mode, and latent dimension are
            inherited from the selected trained model.
          </p>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Optional overrides</h2>
            <p className={styles.sectionDescription}>
              Empty fields use the server defaults.
            </p>

            <div className={styles.fieldGrid}>
              <ParamField
                label="Threshold steps"
                tooltip="Number of thresholds evaluated between start and end. Minimum: 1."
              >
                <input
                  type="number"
                  min={1}
                  placeholder="Server default"
                  value={numOfSteps}
                  onChange={(event) =>
                    setNumOfSteps(event.target.value)
                  }
                />
              </ParamField>

              <ParamField
                label="Threshold start"
                tooltip="Beginning of the SSIM threshold range. Minimum: 0."
              >
                <input
                  type="number"
                  min={0}
                  step="any"
                  placeholder="Server default"
                  value={thresholdInit}
                  onChange={(event) =>
                    setThresholdInit(event.target.value)
                  }
                />
              </ParamField>

              <ParamField
                label="Threshold end"
                tooltip={`End of the SSIM threshold range. Maximum: ${LIMITS.thresholdEndMax}.`}
              >
                <input
                  type="number"
                  max={LIMITS.thresholdEndMax}
                  step="any"
                  placeholder="Server default"
                  value={thresholdEnd}
                  onChange={(event) =>
                    setThresholdEnd(event.target.value)
                  }
                />
              </ParamField>

              <ParamField
                label="Result visualizations"
                tooltip="Saves per-image defect masks and visualizations during metric evaluation."
              >
                <select
                  value={visResults}
                  onChange={(event) =>
                    setVisResults(
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
                label="Reconstruction-only mode"
                tooltip="Saves reconstructed images without calculating ROC, SSIM, or MSE metrics."
              >
                <select
                  value={visReconstruction}
                  onChange={(event) =>
                    setVisReconstruction(
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
                tooltip={`Save a visualization at every Nth threshold. Allowed range: 1-${LIMITS.testVisIntervalMax}.`}
              >
                <input
                  type="number"
                  min={1}
                  max={LIMITS.testVisIntervalMax}
                  placeholder="Server default"
                  value={visInterval}
                  onChange={(event) =>
                    setVisInterval(event.target.value)
                  }
                />
              </ParamField>
            </div>
          </section>

          {readiness.loading ? (
            <p className={styles.readiness}>
              Checking trained models...
            </p>
          ) : null}

          {readiness.error ? (
            <p className={styles.error}>{readiness.error}</p>
          ) : null}

          {configurationError ? (
            <p className={styles.error}>{configurationError}</p>
          ) : null}

          {prerequisiteMessage ? (
            <div className={styles.prerequisite} role="alert">
              <p className={styles.prerequisiteText}>
                {prerequisiteMessage}
              </p>

              <Button
                icon={ArrowRight}
                iconPosition="right"
                onClick={() => navigate("/training")}
              >
                Go to training
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
            startLabel="Start testing"
          />
        </Panel>

        <div className={styles.results}>
          <Panel title="Status">
            <StatusGrid items={statusItems} />

            {run.error ? (
              <p className={styles.error}>{run.error}</p>
            ) : null}
          </Panel>

          <JsonPanel value={output} />

          <PreviewGrid
            title={previewTitle}
            images={preview.images}
            loading={preview.loading}
            error={preview.error}
          />
        </div>
      </div>

      <PageNav step="testing" />
    </div>
  )
}