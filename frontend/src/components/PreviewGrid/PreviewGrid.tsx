import { useState } from "react"
import type { ReactNode } from "react"

import { Lightbox } from "../Lightbox/Lightbox"
import { Panel } from "../Panel/Panel"
import { Spinner } from "../Spinner/Spinner"
import type { PreviewImage } from "../../hooks/usePreview"
import styles from "./PreviewGrid.module.css"

type PreviewGridProps = {
  title?: string
  images: PreviewImage[]
  loading?: boolean
  error?: string | null
  emptyMessage?: string
  className?: string
}

function fileNameOf(name: string): string {
  return name.split("/").pop() ?? name
}

export function PreviewGrid({
  title = "Preview",
  images,
  loading = false,
  error = null,
  emptyMessage = "No images yet — run this step to generate a preview.",
  className,
}: PreviewGridProps) {
  const [selected, setSelected] = useState<PreviewImage | null>(null)

  function renderBody(): ReactNode {
    if (loading) {
      return (
        <div className={styles.state}>
          <Spinner label="Loading preview" />
        </div>
      )
    }

    if (error) {
      return <p className={styles.error}>{error}</p>
    }

    if (images.length === 0) {
      return <p className={styles.empty}>{emptyMessage}</p>
    }

    return (
      <ul className={styles.grid}>
        {images.map((image) => (
          <li key={image.name}>
            <button
              type="button"
              className={styles.thumb}
              onClick={() => setSelected(image)}
            >
              <img
                className={styles.image}
                src={image.url}
                alt={fileNameOf(image.name)}
                loading="lazy"
              />
              <span className={styles.caption}>{fileNameOf(image.name)}</span>
            </button>
          </li>
        ))}
      </ul>
    )
  }

  return (
    <Panel title={title} className={className}>
      {renderBody()}

      {selected ? (
        <Lightbox
          src={selected.url}
          alt={fileNameOf(selected.name)}
          onClose={() => setSelected(null)}
        />
      ) : null}
    </Panel>
  )
}