import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig } from 'vite';

export default defineConfig({
  css: { postcss: { plugins: [tailwindcss()] } },
  plugins: [vinext()],
  optimizeDeps: {
    noDiscovery: true,
    include: [],
  },
  environments: {
    client: { optimizeDeps: { noDiscovery: true, include: [] } },
    rsc: { optimizeDeps: { noDiscovery: true, include: [] } },
    ssr: { optimizeDeps: { noDiscovery: true, include: [] } },
  },
  server: {
    host: '127.0.0.1',
    port: 3100,
    strictPort: true,
    watch: {
      ignored: ['**/.venv/**', '**/.runtime/**', '**/feedback/**', '**/models/**'],
    },
  },
});
