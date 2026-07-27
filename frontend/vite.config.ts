import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

const dataOperationsTarget =
  process.env.DATA_OPS_PROXY_TARGET ?? "http://localhost:8000"

const defectDetectionTarget =
  process.env.DEFECT_PROXY_TARGET ?? "http://localhost:8001"

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    watch: {
      usePolling: true,
    },
    proxy: {
      "/data-ops": {
        target: dataOperationsTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/data-ops/, ""),
      },
      "/defect": {
        target: defectDetectionTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/defect/, ""),
      },
    },
  },
})