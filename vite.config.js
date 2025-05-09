import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['worker-insight-coaches-refinance.trycloudflare.com'], // ✅ Add your Cloudflare tunnel domain here
  },
})

