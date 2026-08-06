import { useEffect, useState } from "react"

import { isAbortError } from "../api/client"
import { getWeightsList } from "../api/defectDetection"
import type {
  DatasetType,
  NetworkType,
  WeightsList,
} from "../api/types"

export type WeightsListState = {
  data: WeightsList | null
  loading: boolean
  error: string | null
}

export function useWeightsList(
  datasetType: DatasetType,
  networkType: NetworkType,
): WeightsListState {
  const [data, setData] = useState<WeightsList | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false

    setData(null)
    setError(null)
    setLoading(true)

    getWeightsList(
      datasetType,
      networkType,
      controller.signal,
    )
      .then((response) => {
        if (cancelled) {
          return
        }

        setData(response)
        setLoading(false)
      })
      .catch((reason) => {
        if (cancelled || isAbortError(reason)) {
          return
        }

        setError(
          reason instanceof Error
            ? reason.message
            : String(reason),
        )
        setLoading(false)
      })

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [datasetType, networkType])

  return { data, loading, error }
}