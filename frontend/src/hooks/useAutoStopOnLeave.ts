import { useEffect, useRef } from "react"

/* Best-effort auto-stop for the blocking Aug/Draw runs: while `active`, leaving
   the page (SPA nav, F5, tab close) hits the existing /stop endpoint. stopSentRef
   keeps it to a single call per leave. */
export function useAutoStopOnLeave(
  active: boolean,
  stopPath: string,
) {
  const activeRef = useRef(active)
  const stopSentRef = useRef(false)

  activeRef.current = active

  useEffect(() => {
    const sendStop = (preferBeacon: boolean) => {
      if (!activeRef.current || stopSentRef.current) {
        return
      }

      stopSentRef.current = true

      if (preferBeacon && navigator.sendBeacon(stopPath)) {
        return
      }

      void fetch(stopPath, {
        method: "POST",
        keepalive: true,
      }).catch(() => undefined)
    }

    const handlePageHide = () => sendStop(true)

    window.addEventListener("pagehide", handlePageHide)

    return () => {
      window.removeEventListener("pagehide", handlePageHide)
      sendStop(false)
    }
  }, [stopPath])
}