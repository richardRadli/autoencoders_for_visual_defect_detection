import { useEffect, useState } from "react"

import { runDrawRectangles, stopDrawRectangles } from "../api/dataOperations"
import { DATASET_TYPES, LIMITS, SOURCE_IMAGES } from "../api/types"
import type {
  DatasetType,
  DrawRectanglesParams,
  SourceImage,
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
import { StatusGrid } from "../components/StatusGrid/StatusGrid"
import type { RunState } from "../hooks/useBlockingRun"
import { useBlockingRun } from "../hooks/useBlockingRun"
import { useOpsPreview } from "../hooks/usePreview"
import { formatElapsedTime } from "../utils/format"
import styles from "./DrawRectanglesPage.module.css"

const RUN_BADGE: Record<RunState, { variant: BadgeVariant; label: string }> = {
  idle: { variant: "neutral", label: "Idle" },
  running: { variant: "accent", label: "Running" },
  done: { variant: "success", label: "Done" },
  stopped: { variant: "warning", label: "Stopped" },
  error: { variant: "danger", label: "Failed" },
}

/* An empty field means "let the server decide" - buildQuery drops undefined,
   so the parameter is left out of the request entirely. */
function toNumber(value: string): number | undefined {
  const trimmed = value.trim()

  if (trimmed === "") {
    return undefined
  }

  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : undefined
}

export function DrawRectanglesPage() {
  const [datasetType, setDatasetType] = useState<DatasetType>("texture_1")
  const [source, setSource] = useState<SourceImage | "">("")
  const [sizeOfCover, setSizeOfCover] = useState("")

  const run = useBlockingRun(runDrawRectangles, stopDrawRectangles)
  const preview = useOpsPreview(datasetType, "noise")
  const result = run.result

  useEffect(() => {
    if (run.state === "done" || run.state === "stopped") {
      preview.reload()
    }
  }, [run.state, preview.reload])

  function handleStart() {
    const params: DrawRectanglesParams = {
      dataset_type: datasetType,
      source: source === "" ? undefined : source,
      size_of_cover: toNumber(sizeOfCover),
    }

    void run.start(params)
  }

  const badge = RUN_BADGE[run.state]

  const previewTitle =
    run.state === "done" || run.state === "stopped"
      ? "Current covered images"
      : "Previous covered images"

  return (
    <div className={styles.page}>
      <Breadcrumb step="draw-rectangles" />

      <header className={styles.intro}>
        <h1 className={styles.title}>Draw rectangles</h1>
        <p className={styles.subtitle}>
          Covers parts of the images with grey rectangles. The network learns to
          rebuild what is hidden, which is what makes it notice defects later.
        </p>
      </header>

      <div className={styles.columns}>
        <Panel title="Parameters" className={styles.params}>
          <div className={styles.fields}>
            <ParamField
              label="Dataset"
              tooltip="Which dataset folder to read the images from."
            >
              <select
                value={datasetType}
                onChange={(event) =>
                  setDatasetType(event.target.value as DatasetType)
                }
              >
                {DATASET_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </ParamField>

            <ParamField
              label="Source"
              hint="Leave on the default to let the server choose."
              tooltip="Which images to cover: the original good ones, or the augmented ones."
            >
              <select
                value={source}
                onChange={(event) =>
                  setSource(event.target.value as SourceImage | "")
                }
              >
                <option value="">Server default</option>

                {SOURCE_IMAGES.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </ParamField>

            <ParamField
              label="Size of cover"
              hint="Leave empty for the server default."
              tooltip="The edge length of each grey rectangle, in pixels."
            >
              <input
                type="number"
                min={LIMITS.sizeOfCoverMin}
                max={LIMITS.sizeOfCoverMax}
                placeholder="e.g. 20"
                value={sizeOfCover}
                onChange={(event) => setSizeOfCover(event.target.value)}
              />
            </ParamField>
          </div>

          <p className={styles.limits}>
            The server accepts a cover size between {LIMITS.sizeOfCoverMin} and{" "}
            {LIMITS.sizeOfCoverMax} pixels.
          </p>

          <RunControls
            className={styles.controls}
            running={run.state === "running"}
            stopping={run.stopping}
            onStart={handleStart}
            onStop={() => void run.stop()}
            startLabel="Start drawing"
          />
        </Panel>

        <div className={styles.results}>
          <Panel title="Status">
            <StatusGrid
              items={[
                {
                  label: "State",
                  value: <Badge variant={badge.variant}>{badge.label}</Badge>,
                },
                {
                  label: "Elapsed",
                  value: formatElapsedTime(run.elapsedMs),
                },
                {
                  label: "Source",
                  value: result ? (
                    <span className={styles.pathValue}>{`${datasetType}/${result.source}`}</span>
                  ) : undefined,
                  title: result?.source_dir,
                },
                {
                  label: "Target",
                  value: result ? (
                    <span className={styles.pathValue}>{`${datasetType}/noise`}</span>
                  ) : undefined,
                  title: result?.target_dir,
                },
                {
                  label: "Source used",
                  value: result?.source,
                },
                {
                  label: "Processed images",
                  value: result?.processed_images,
                },
              ]}
            />

            {run.error ? <p className={styles.error}>{run.error}</p> : null}
          </Panel>

          <JsonPanel value={result} />

          <PreviewGrid
            title={previewTitle}
            images={preview.images}
            loading={preview.loading}
            error={preview.error}
          />
        </div>
      </div>

      <PageNav step="draw-rectangles" />
    </div>
  )
}