import { useState } from "react"
import { runAugmentation, stopAugmentation } from "../api/dataOperations"
import type { AugmentationResult } from "../api/dataOperations"
import { ParamField } from "../components/ParamField"
import { RunControls } from "../components/RunControls"
import { OutputPanel } from "../components/OutputPanel"
import { BackToMenu } from "../components/BackToMenu"

export function AugmentationPage() {
  const [datasetType, setDatasetType] = useState("texture_1")
  const [imgSize, setImgSize] = useState(256)
  const [cropSize, setCropSize] = useState(128)
  const [rotate, setRotate] = useState("")
  const [hflip, setHflip] = useState("")
  const [vflip, setVflip] = useState("")

  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<AugmentationResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleStart = async () => {
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      const res = await runAugmentation({
        dataset_type: datasetType,
        img_size: imgSize,
        crop_size: cropSize,
        rotate_count: rotate === "" ? undefined : Number(rotate),
        horizontal_flip_count: hflip === "" ? undefined : Number(hflip),
        vertical_flip_count: vflip === "" ? undefined : Number(vflip),
      })
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }

  const handleStop = async () => {
    try {
      await stopAugmentation()
    } catch {
      // ignore stop errors
    }
  }

  return (
    <div style={{ maxWidth: 720 }}>
      <h1 style={{ fontSize: 22, fontWeight: 500, marginBottom: 6 }}>Augmentation</h1>
      <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5, margin: "0 0 18px", maxWidth: 620 }}>
        Augmentation expands your clean images into thousands of training samples by cropping,
        rotating and flipping them — the raw material for training.
      </p>

      <div className="card">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px 16px" }}>
          <ParamField label="Dataset" hint="Which image set to process.">
            <select className="field-input" value={datasetType} onChange={(e) => setDatasetType(e.target.value)}>
              <option value="texture_1">texture_1</option>
              <option value="texture_2">texture_2</option>
              <option value="cpu">cpu</option>
            </select>
          </ParamField>

          <ParamField label="Image size" hint="Source image size in pixels.">
            <select className="field-input" value={imgSize} onChange={(e) => setImgSize(Number(e.target.value))}>
              <option value={256}>256</option>
              <option value={512}>512</option>
            </select>
          </ParamField>

          <ParamField label="Crop size" hint="Patch size in pixels. Must be smaller than the image size and divide it evenly.">
            <select className="field-input" value={cropSize} onChange={(e) => setCropSize(Number(e.target.value))}>
              <option value={128}>128</option>
              <option value={256}>256</option>
              <option value={64}>64</option>
            </select>
          </ParamField>

          <ParamField label="Rotate count" hint="How many rotated variants to make. Leave empty for the server default.">
            <input className="field-input" type="number" min={0} placeholder="default" value={rotate} onChange={(e) => setRotate(e.target.value)} />
          </ParamField>

          <ParamField label="Horizontal flip count" hint="How many horizontally flipped variants. Leave empty for the server default.">
            <input className="field-input" type="number" min={0} placeholder="default" value={hflip} onChange={(e) => setHflip(e.target.value)} />
          </ParamField>

          <ParamField label="Vertical flip count" hint="How many vertically flipped variants. Leave empty for the server default.">
            <input className="field-input" type="number" min={0} placeholder="default" value={vflip} onChange={(e) => setVflip(e.target.value)} />
          </ParamField>
        </div>

        <p style={{ fontSize: 12, color: "var(--text-muted)", margin: "12px 0 16px" }}>
          Empty fields use the server default. The total augmented count must be 5000–20000, or exactly 0.
        </p>

        <RunControls running={running} onStart={handleStart} onStop={handleStop} startLabel="Start augmentation" />
      </div>

      <OutputPanel running={running} error={error} result={result} />

      <div style={{ marginTop: 20 }}>
        <BackToMenu />
      </div>
    </div>
  )
}