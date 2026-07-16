import { useCallback, useEffect, useRef, useState } from "react"

import { isAbortError } from "../api/client"
import { getOpsPreview, opsPreviewImageUrl } from "../api/dataOperations"
import { defectPreviewImageUrl, getDefectPreview } from "../api/defectDetection"
import type {
  DatasetType,
  DefectPreviewType,
  NetworkType,
  OpsPreviewType,
  SubtestFolder,
} from "../api/types"

export type PreviewImage = {
  name: string
  url: string
}

export type Preview = {
  images: PreviewImage[]
  loading: boolean
  error: string | null
  reload: () => void
}

/*
 * The backend returns image names only; the browser loads each one from the
 * image endpoint. There are at most five — sample_evenly caps the list and
 * returns fewer when the run holds fewer images, so an empty result is a
 * normal state, not a failure.
 *
 * The closures are read through refs and the effect keys off a plain string,
 * because fetchList/toUrl are rebuilt on every render and would otherwise
 * restart the request forever.
 */
function usePreviewList(
  fetchList: (signal: AbortSignal) => Promise<{ images: string[] }>,
  toUrl: (name: string) => string,
  key: string,
): Preview {
  const [images, setImages] = useState<PreviewImage[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)

  const fetchRef = useRef(fetchList)
  const urlRef = useRef(toUrl)
  fetchRef.current = fetchList
  urlRef.current = toUrl

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false

    setImages([])
    setError(null)
    setLoading(true)

    fetchRef
      .current(controller.signal)
      .then((res) => {
        if (cancelled) {
          return
        }
        setImages(res.images.map((name) => ({ name, url: urlRef.current(name) })))
        setLoading(false)
      })
      .catch((e) => {
        if (cancelled || isAbortError(e)) {
          return
        }
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      })

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [key, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])

  return { images, loading, error, reload }
}

export function useOpsPreview(
  datasetType: DatasetType,
  previewType: OpsPreviewType,
  limit = 5,
): Preview {
  return usePreviewList(
    (signal) => getOpsPreview(datasetType, previewType, limit, signal),
    (name) => opsPreviewImageUrl(datasetType, previewType, name),
    `${datasetType}|${previewType}|${limit}`,
  )
}

export type DefectPreviewOpts = {
  subtestFolder?: SubtestFolder
  networkType?: NetworkType
  limit?: number
}

export function useDefectPreview(
  datasetType: DatasetType,
  previewType: DefectPreviewType,
  options: DefectPreviewOpts = {},
): Preview {
  const { subtestFolder, networkType, limit = 5 } = options

  return usePreviewList(
    (signal) =>
      getDefectPreview(
        datasetType,
        previewType,
        { subtest_folder: subtestFolder, network_type: networkType, limit },
        signal,
      ),
    (name) =>
      defectPreviewImageUrl(datasetType, previewType, name, {
        subtest_folder: subtestFolder,
        network_type: networkType,
      }),
    `${datasetType}|${previewType}|${subtestFolder ?? ""}|${networkType ?? ""}|${limit}`,
  )
}