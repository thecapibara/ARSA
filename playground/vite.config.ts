import { defineConfig } from 'vite';

export default defineConfig({
  assetsInclude: ['**/*.wasm'],
  server: {
    port: 5173,
    fs: {
      allow: ['..'],
    },
  },
  optimizeDeps: {
    exclude: ['./src/wasm'],
  },
  build: {
    target: 'esnext',
  },
});
