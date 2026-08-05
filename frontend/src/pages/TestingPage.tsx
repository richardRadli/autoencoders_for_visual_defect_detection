import { useState } from "react"
import { Link } from "react-router-dom"

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
  TestResult,
  TestingParams,
} from "../api/types"
import { Badge } from "../components/Badge/Badge"
import type { BadgeVariant } from "../components/Badge/Badge"
import { Breadcrumb } from "../components/Breadcrumb/Breadcrumb"
import { JsonPanel } from "../components/JsonPanel/JsonPanel"
import { PageNav } from "../components/PageNav/PageNav"
import { Panel } from "../components/Panel/Panel"
import { ParamField } from "../components/ParamField/ParamField"
import { PreviewGrid } from "../components/PreviewGrid/PreviewGrid"
import { RunControls } from "../components/RunControls/RunControls"
import { RunningNotice } from "../components/RunningNotice/RunningNotice"
import { StatusGrid } from "../components/StatusGrid/StatusGrid"
import { useDefectPreview } from "../hooks/usePreview"
import type { TaskRunState } from "../hooks/useTaskPolling"
import { useTaskPolling } from "../hooks/useTaskPolling"
import { useReadiness } from "../hooks/useReadiness"
import { formatElapsedTime } from "../utils/format"
import styles from "./TestingPage.module.css"

const TESTING_TASK_STORAGE_KEY = "defect-detection:testing-task"

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

function isMetricsResult(
  result: TestResult | null,
): result is TestMetricsResult {
  return result !== null && "auc_roc" in result
}

function resolvePreviewType(
  taskState: TaskRunState,
  result: TestResult | null,
  submittedParams: TestingParams | null,
): DefectPreviewType {
  if (taskState !== "done" || result === null) {
    return "test"
  }

  if ("reconstructed_images" in result) {
    return "reconstruction"
  }

  if (submittedParams?.vis_results === true) {
    return "reconstruction_vis"
  }

  return "roc_plot"
}

export function TestingPage() {
  const task = useTaskPolling<TestingParams, TestResult>(
    runTesting,
    getTestingStatus,
    stopTesting,
    {
      storageKey: TESTING_TASK_STORAGE_KEY,
      intervalMs: 2000,
    },
  )

  const restored = task.submittedParams
  const initialDatasetType =
    restored?.dataset_type ?? "texture_1"

  const [datasetType, setDatasetType] =
    useState<DatasetType>(initialDatasetType)

  const [aeType, setAeType] = useState<AEType>(
    restored?.ae_type ?? "plain",
  )

  const [modelSize, setModelSize] = useState<ModelSize>(
    restored?.model_size ?? "base",
  )

  const [subtestFolder, setSubtestFolder] =
    useState<SubtestFolder>(
      restored?.subtest_folder ??
        VALID_SUBTESTS[initialDatasetType][0],
    )

  const [stride, setStride] = useState<Stride>(
    restored?.stride ?? 32,
  )

  const [numOfSteps, setNumOfSteps] = useState(
    toInputValue(restored?.num_of_steps),
  )

  const [thresholdInit, setThresholdInit] = useState(
    toInputValue(restored?.threshold_init),
  )

  const [thresholdEnd, setThresholdEnd] = useState(
    toInputValue(restored?.threshold_end),
  )

  const [visResults, setVisResults] =
    useState<OptionalBoolean>(
      toBooleanField(restored?.vis_results),
    )

  const [visReconstruction, setVisReconstruction] =
    useState<OptionalBoolean>(
      toBooleanField(restored?.vis_reconstruction),
    )

  const [visInterval, setVisInterval] = useState(
    toInputValue(restored?.vis_interval),
  )

  const readiness = useReadiness(
    getDefectReadiness,
    datasetType,
  )

  const selectedNetworkType = resolveNetworkType(
    aeType,
    modelSize,
  )

  const modelReady =
    readiness.data?.trained_networks.includes(
      selectedNetworkType,
    ) === true

  const visualizationConflict =
    visResults === "true" &&
    visReconstruction === "true"

  const parsedThresholdInit = toNumber(thresholdInit)
  const parsedThresholdEnd = toNumber(thresholdEnd)

  const thresholdOrderInvalid =
    parsedThresholdInit !== undefined &&
    parsedThresholdEnd !== undefined &&
    parsedThresholdInit >= parsedThresholdEnd

  const startDisabled =
    readiness.loading ||
    readiness.error !== null ||
    !modelReady ||
    visualizationConflict ||
    thresholdOrderInvalid

  function handleDatasetChange(nextDataset: DatasetType) {
    setDatasetType(nextDataset)

    if (
      !VALID_SUBTESTS[nextDataset].includes(
        subtestFolder,
      )
    ) {
      setSubtestFolder(
        VALID_SUBTESTS[nextDataset][0],
      )
    }
  }

  function handleStart() {
    const params: TestingParams = {
      dataset_type: datasetType,
      ae_type: aeType,
      model_size: modelSize,
      subtest_folder: subtestFolder,
      stride,
      num_of_steps: toNumber(numOfSteps),
      threshold_init: toNumber(thresholdInit),
      threshold_end: toNumber(thresholdEnd),
      vis_results: toOptionalBoolean(visResults),
      vis_reconstruction:
        toOptionalBoolean(visReconstruction),
      vis_interval: toNumber(visInterval),
    }

    void task.start(params)
  }

  const activeParams =
    task.state === "idle"
      ? null
      : task.submittedParams

  const previewDatasetType =
    activeParams?.dataset_type ?? datasetType

  const previewAeType =
    activeParams?.ae_type ?? aeType

  const previewModelSize =
    activeParams?.model_size ?? modelSize

  const previewNetworkType = resolveNetworkType(
    previewAeType,
    previewModelSize,
  )

  const previewSubtestFolder =
    activeParams?.subtest_folder ?? subtestFolder

  const previewType = resolvePreviewType(
    task.state,
    task.result,
    task.submittedParams,
  )

  const preview = useDefectPreview(
    previewDatasetType,
    previewType,
    previewType === "test"
      ? {
          subtestFolder: previewSubtestFolder,
          limit: 5,
        }
      : {
          networkType:
            task.result?.network_type ??
            previewNetworkType,
          limit: 5,
        },
  )

  const previewTitle = `Preview: ${previewType}`
  const badge = RUN_BADGE[task.state]

  const submittedNetworkType = task.submittedParams
    ? resolveNetworkType(
        task.submittedParams.ae_type ?? "plain",
        task.submittedParams.model_size ?? "base",
      )
    : selectedNetworkType

  const metricsResult = isMetricsResult(task.result)
    ? task.result
    : null

  const reconstructedImages =
    task.result &&
    "reconstructed_images" in task.result
      ? task.result.reconstructed_images
      : undefined

  const outputLabel = task.result
    ? reconstructedImages !== undefined
      ? "reconstruction"
      : "metrics"
    : undefined

  const statusOutput = task.taskId
    ? {
        task_id: task.taskId,
        status: task.taskState,
        info: task.result ?? task.error,
      }
    : null

  return (
    <div className={styles.page}>
      <RunningNotice running={task.state === "running"} label="Testing" />

      <Breadcrumb step="testing" />

      <header className={styles.intro}>
        <h1 className={styles.title}>Testing</h1>
        <p className={styles.subtitle}>
          Evaluates a trained model and creates anomaly metrics or
          reconstruction outputs. Image size, crop size and grayscale
          are inherited from the selected trained model.
        </p>
      </header>

      <div className={styles.columns}>
        <Panel
          title="Parameters"
          className={styles.params}
        >
          <div className={styles.fields}>
            <ParamField
              label="dataset_type"
              tooltip="Dataset to evaluate. Allowed values: texture_1, texture_2, cpu."
            >
              <select
                value={datasetType}
                onChange={(event) =>
                  handleDatasetChange(
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
              tooltip="plain selects AE/AEE. denoising selects DAE/DAEE."
            >
              <select
                value={aeType}
                onChange={(event) =>
                  setAeType(
                    event.target.value as AEType,
                  )
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
              tooltip="base selects the standard model. extended selects the deeper model."
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

            <ParamField
              label="subtest_folder"
              tooltip="texture datasets use defective. cpu uses added, contamination or missing."
            >
              <select
                value={subtestFolder}
                onChange={(event) =>
                  setSubtestFolder(
                    event.target.value as SubtestFolder,
                  )
                }
              >
                {VALID_SUBTESTS[datasetType].map(
                  (value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ),
                )}
              </select>
            </ParamField>

            <ParamField
              label="stride"
              tooltip="Sliding-window stride. Allowed values: 4, 8, 16, 32, 64. Smaller values create more overlapping patches."
            >
              <select
                value={stride}
                onChange={(event) =>
                  setStride(
                    Number(
                      event.target.value,
                    ) as Stride,
                  )
                }
              >
                {STRIDES.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </ParamField>

            <h3>Optional overrides</h3>
            <p>
              Empty fields use the values from
              testing_config.json.
            </p>

            <ParamField
              label="num_of_steps"
              hint="Server default when empty."
              tooltip="Number of thresholds evaluated between threshold_init and threshold_end. Minimum: 1."
            >
              <input
                type="number"
                min={1}
                step={1}
                placeholder="e.g. 50"
                value={numOfSteps}
                onChange={(event) =>
                  setNumOfSteps(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="threshold_init"
              hint="Server default when empty."
              tooltip="Start of the SSIM residual threshold range. Minimum: 0. Must be smaller than threshold_end."
            >
              <input
                type="number"
                min={0}
                step="any"
                placeholder="e.g. 0.01"
                value={thresholdInit}
                onChange={(event) =>
                  setThresholdInit(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="threshold_end"
              hint="Server default when empty."
              tooltip="End of the SSIM residual threshold range. Maximum: 2. Must be greater than threshold_init."
            >
              <input
                type="number"
                max={LIMITS.thresholdEndMax}
                step="any"
                placeholder="e.g. 1.01"
                value={thresholdEnd}
                onChange={(event) =>
                  setThresholdEnd(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="vis_results"
              tooltip="true saves per-image anomaly-mask visualizations. false does not save them."
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
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </ParamField>

            <ParamField
              label="vis_reconstruction"
              tooltip="true runs reconstruction-only mode without ROC, SSIM or MSE metrics. It cannot be true together with vis_results."
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
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </ParamField>

            <ParamField
              label="vis_interval"
              hint="Server default when empty."
              tooltip={`Saves a visualization at every Nth threshold. Minimum: 1. Maximum: ${LIMITS.testVisIntervalMax}.`}
            >
              <input
                type="number"
                min={1}
                max={LIMITS.testVisIntervalMax}
                step={1}
                placeholder="e.g. 10"
                value={visInterval}
                onChange={(event) =>
                  setVisInterval(event.target.value)
                }
              />
            </ParamField>
          </div>

          {readiness.loading ? (
            <p>Checking trained models...</p>
          ) : null}

          {readiness.error ? (
            <p className={styles.error}>
              Readiness check failed: {readiness.error}
            </p>
          ) : null}

          {!readiness.loading &&
          readiness.data &&
          !modelReady ? (
            <p className={styles.error}>
              No trained {selectedNetworkType} model is
              available for {datasetType}.{" "}
              <Link to="/training">Open training</Link>
            </p>
          ) : null}

          {visualizationConflict ? (
            <p className={styles.error}>
              vis_results and vis_reconstruction cannot
              both be true.
            </p>
          ) : null}

          {thresholdOrderInvalid ? (
            <p className={styles.error}>
              threshold_init must be smaller than
              threshold_end.
            </p>
          ) : null}

          <RunControls
            className={styles.controls}
            running={task.state === "running"}
            stopping={task.stopping}
            onStart={handleStart}
            onStop={() => void task.stop()}
            startLabel={`Start testing (${selectedNetworkType})`}
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
                  value: formatElapsedTime(
                    task.elapsedMs,
                  ),
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
                  label: "subtest_folder",
                  value:
                    metricsResult?.subtest_folder ??
                    task.submittedParams?.subtest_folder ??
                    subtestFolder,
                },
                {
                  label: "output",
                  value: outputLabel,
                },
                ...(metricsResult
                  ? [
                      {
                        label: "auc_roc",
                        value:
                          metricsResult.auc_roc.toFixed(6),
                      },
                      {
                        label: "avg_ssim",
                        value:
                          metricsResult.avg_ssim.toFixed(6),
                      },
                      {
                        label: "mse_avg",
                        value:
                          metricsResult.mse_avg.toFixed(6),
                      },
                    ]
                  : []),
                ...(reconstructedImages !== undefined
                  ? [
                      {
                        label: "reconstructed_images",
                        value: reconstructedImages,
                      },
                    ]
                  : []),
              ]}
            />

            {task.taskId ? (
              <p title={task.taskId}>
                task_id: <code>{task.taskId}</code>
              </p>
            ) : null}

            {task.error ? (
              <p className={styles.error}>
                {task.error}
              </p>
            ) : null}
          </Panel>

          <JsonPanel value={statusOutput} />

          <PreviewGrid
            className={
              previewType === "roc_plot"
                ? styles.rocPreview
                : undefined
            }
            title={previewTitle}
            images={preview.images}
            loading={preview.loading}
            error={preview.error}
            emptyMessage={`No ${previewType} images are available for this selection.`}
          />
        </div>
      </div>

      <PageNav step="testing" />
    </div>
  )
}