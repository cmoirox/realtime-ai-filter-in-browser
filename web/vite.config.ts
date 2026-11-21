import { defineConfig } from "vite";

// Cross-origin isolation (COOP/COEP) enables WebAssembly threads for the fallback path.
const headers = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  // relative base: works on GitHub Pages (/<repo>/), Vercel, Netlify or any static host
  base: "./",
  server: { host: true, headers },
  preview: { host: true, headers },
  optimizeDeps: { exclude: ["@litertjs/core", "@mediapipe/tasks-vision"] },
  build: { target: "es2022" },
});
