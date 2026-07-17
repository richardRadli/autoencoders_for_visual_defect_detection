import { useEffect, useState } from "react"

import { runAugmentation, stopAugmentation } from "../api/dataOperations"
import { CROP_SIZES, DATASET_TYPES, IMG_SIZES, LIMITS } from "../api/types"
import type {
  AugmentationParams,
  CropSize,
  DatasetType,
  ImgSize,
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
import styles from "./AugmentationPage.module.css"

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

export function AugmentationPage() {
  const [datasetType, setDatasetType] = useState<DatasetType>("texture_1")
  const [imgSize, setImgSize] = useState<ImgSize>(512)
  const [cropSize, setCropSize] = useState<CropSize>(256)
  const [rotateCount, setRotateCount] = useState("")
  const [horizontalFlipCount, setHorizontalFlipCount] = useState("")
  const [verticalFlipCount, setVerticalFlipCount] = useState("")

  const run = useBlockingRun(runAugmentation, stopAugmentation)
  const preview = useOpsPreview(datasetType, "aug")
  const result = run.result

  useEffect(() => {
    if (run.state === "done" || run.state === "stopped") {
      preview.reload()
    }
  }, [run.state, preview.reload])

  function handleStart() {
    const params: AugmentationParams = {
      dataset_type: datasetType,
      img_size: imgSize,
      crop_size: cropSize,
      rotate_count: toNumber(rotateCount),
      horizontal_flip_count: toNumber(horizontalFlipCount),
      vertical_flip_count: toNumber(verticalFlipCount),
    }

    void run.start(params)
  }

  const badge = RUN_BADGE[run.state]

  return (
    <div className={styles.page}>
      <Breadcrumb step="augmentation" />

      <header className={styles.intro}>
        <h1 className={styles.title}>Augmentation</h1>
        <p className={styles.subtitle}>
          Rotates and flips the good images, then crops them to the size the
          network expects.
        </p>
      </header>

      <div className={styles.columns}>
        <Panel title="Parameters" className={styles.params}>
          <div className={styles.fields}>
            <ParamField
              label="Dataset"
              tooltip="Which dataset folder to read the good images from."
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
              label="Image size"
              tooltip="The source images are resized to this square size before cropping."
            >
              <select
                value={imgSize}
                onChange={(event) =>
                  setImgSize(Number(event.target.value) as ImgSize)
                }
              >
                {IMG_SIZES.map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </ParamField>

            <ParamField
              label="Crop size"
              tooltip="Each resized image is cut into crops of this size. Smaller crops mean more of them."
            >
              <select
                value={cropSize}
                onChange={(event) =>
                  setCropSize(Number(event.target.value) as CropSize)
                }
              >
                {CROP_SIZES.map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </ParamField>

            <ParamField
              label="Rotate count"
              hint="Leave empty for the server default."
              tooltip="How many rotated copies to make of each image."
            >
              <input
                type="number"
                min={0}
                value={rotateCount}
                onChange={(event) => setRotateCount(event.target.value)}
              />
            </ParamField>

            <ParamField
              label="Horizontal flip count"
              hint="Leave empty for the server default."
              tooltip="How many horizontally flipped copies to make of each image."
            >
              <input
                type="number"
                min={0}
                value={horizontalFlipCount}
                onChange={(event) =>
                  setHorizontalFlipCount(event.target.value)
                }
              />
            </ParamField>

            <ParamField
              label="Vertical flip count"
              hint="Leave empty for the server default."
              tooltip="How many vertically flipped copies to make of each image."
            >
              <input
                type="number"
                min={0}
                value={verticalFlipCount}
                onChange={(event) => setVerticalFlipCount(event.target.value)}
              />
            </ParamField>
          </div>

          <p className={styles.limits}>
            The server rejects runs that would produce fewer than{" "}
            {LIMITS.augmentedTotalMin.toLocaleString("en-US")} or more than{" "}
            {LIMITS.augmentedTotalMax.toLocaleString("en-US")} images.
          </p>

          <RunControls
            className={styles.controls}
            running={run.state === "running"}
            stopping={run.stopping}
            onStart={handleStart}
            onStop={() => void run.stop()}
            startLabel="Start augmentation"
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
                  label: "Source images",
                  value: result?.source_images,
                },
                {
                  label: "Augmented images",
                  value: result?.augmented_images,
                },
                {
                  label: "Processed images",
                  value: result?.processed_images,
                },
              ]}
            />

            {run.error ? <p className={styles.error}>{run.error}</p> : null}

            {result?.warning ? (
              <p className={styles.warning}>{result.warning}</p>
            ) : null}
          </Panel>

          <JsonPanel value={result} />

          <PreviewGrid
            title="Augmented images"
            images={preview.images}
            loading={preview.loading}
            error={preview.error}
          />
        </div>
      </div>

      <PageNav step="augmentation" />
    </div>
  )
}