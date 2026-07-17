import {
  ChartLine,
  ChevronRight,
  Cpu,
  ImagePlus,
  Play,
  SquarePlus,
} from "lucide-react"
import type { LucideIcon } from "lucide-react"
import { useState } from "react"
import { Link } from "react-router-dom"

import { getOpsReadiness } from "../api/dataOperations"
import { getDefectReadiness } from "../api/defectDetection"
import { DATASET_TYPES } from "../api/types"
import type { DatasetType, DefectReadiness, OpsReadiness } from "../api/types"
import { Button } from "../components/Button/Button"
import { GpuBar } from "../components/GpuBar/GpuBar"
import { Panel } from "../components/Panel/Panel"
import { Spinner } from "../components/Spinner/Spinner"
import { WorkflowStrip } from "../components/WorkflowStrip/WorkflowStrip"
import type { StepInfo } from "../components/WorkflowStrip/WorkflowStrip"
import { SERVICES, getServiceSteps } from "../config/workflow"
import type { ServiceId, StepId } from "../config/workflow"
import { useReadiness } from "../hooks/useReadiness"
import styles from "./MainMenu.module.css"

const SERVICE_ICON: Record<ServiceId, LucideIcon> = {
  data_operations: ImagePlus,
  defect_detection: Cpu,
}

const STEP_ICON: Record<StepId, LucideIcon> = {
  augmentation: SquarePlus,
  "draw-rectangles": SquarePlus,
  training: Play,
  testing: ChartLine,
}

/* Pinned to en-US so the grouping is "12,500" everywhere — the UI is English,
   and the machine locale would otherwise decide. */
function formatCount(value: number): string {
  return value.toLocaleString("en-US")
}

/* The two services report readiness in their own shapes; this is the only
   place that knows how they map onto the four workflow steps. */
function buildSteps(
  ops: OpsReadiness,
  defect: DefectReadiness,
): Record<StepId, StepInfo> {
  const trained = defect.trained_networks

  return {
    augmentation: ops.augmentation.ready
      ? {
          state: "complete",
          detail: `${formatCount(ops.augmentation.images)} images`,
        }
      : { state: "ready", detail: "Not run yet" },

    "draw-rectangles": ops.draw_rectangles.ready
      ? {
          state: "complete",
          detail: `${formatCount(ops.draw_rectangles.images)} images`,
        }
      : ops.augmentation.ready
        ? { state: "ready", detail: "Ready to run" }
        : { state: "missing", detail: "Augment first" },

    training: trained.length > 0
      ? { state: "complete", detail: `Trained: ${trained.join(", ")}` }
      : defect.aug.ready
        ? { state: "ready", detail: "Ready to train" }
        : { state: "missing", detail: "Augment first, then train" },

    // No readiness fact reports finished test runs, so testing never claims
    // to be complete — it only says whether it can start.
    testing: trained.length > 0
      ? { state: "ready", detail: "Ready to test" }
      : { state: "missing", detail: "Train first, then test" },
  }
}

export function MainMenu() {
  const [datasetType, setDatasetType] = useState<DatasetType>("texture_1")

  const ops = useReadiness(getOpsReadiness, datasetType)
  const defect = useReadiness(getDefectReadiness, datasetType)

  const loading = ops.loading || defect.loading
  const error = ops.error ?? defect.error

  function reload() {
    ops.reload()
    defect.reload()
  }

  const datasetSelect = (
    <label className={styles.dataset}>
      <span className={styles.datasetLabel}>Dataset</span>
      <select
        className={styles.datasetSelect}
        value={datasetType}
        onChange={(event) => setDatasetType(event.target.value as DatasetType)}
      >
        {DATASET_TYPES.map((type) => (
          <option key={type} value={type}>
            {type}
          </option>
        ))}
      </select>
    </label>
  )

  return (
    <div className={styles.page}>
      <div className={styles.services}>
        {Object.values(SERVICES).map((service) => {
          const Icon = SERVICE_ICON[service.id]

          return (
            <section className={styles.service} key={service.id}>
              <header className={styles.serviceHead}>
                <span className={styles.serviceIcon} aria-hidden="true">
                  <Icon className={styles.serviceGlyph} />
                </span>

                <div className={styles.serviceText}>
                  <h2 className={styles.serviceTitle}>{service.label}</h2>
                  <p className={styles.serviceDescription}>
                    {service.description}
                  </p>
                </div>

                <span className={styles.port}>:{service.port}</span>
              </header>

              <ul className={styles.stepLinks}>
                {getServiceSteps(service.id).map((step) => {
                  const StepIcon = STEP_ICON[step.id]

                  return (
                    <li key={step.id}>
                      <Link className={styles.stepLink} to={step.path}>
                        <StepIcon className={styles.stepIcon} aria-hidden="true" />
                        <span className={styles.stepLabel}>{step.label}</span>
                        <ChevronRight
                          className={styles.chevron}
                          aria-hidden="true"
                        />
                      </Link>
                    </li>
                  )
                })}
              </ul>
            </section>
          )
        })}
      </div>

      <Panel title="Workflow" actions={datasetSelect}>
        {loading ? (
          <div className={styles.state}>
            <Spinner size="large" label="Checking dataset readiness" />
          </div>
        ) : error ? (
          <div className={styles.state}>
            <p className={styles.error}>{error}</p>
            <Button onClick={reload}>Retry</Button>
          </div>
        ) : ops.data && defect.data ? (
          <WorkflowStrip steps={buildSteps(ops.data, defect.data)} />
        ) : null}
      </Panel>

      <GpuBar className={styles.gpu} />
    </div>
  )
}