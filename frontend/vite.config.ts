import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    proxy: {
      '/admin/metrics': 'http://localhost:8000',
      '/admin/health': 'http://localhost:8000',
      '/admin/config': 'http://localhost:8000',
      '/admin/restart': 'http://localhost:8000',
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  }
})
