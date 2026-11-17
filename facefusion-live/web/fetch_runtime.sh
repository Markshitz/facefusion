#!/usr/bin/env bash
# Fetch the ONNX Runtime Web files index.html loads from its own folder:
# CPU runtime (always) and the WebGPU (JSEP) runtime, used when the browser has a GPU.
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
(cd "$tmp" && npm pack onnxruntime-web@1.30.0 --silent >/dev/null && tar xzf onnxruntime-web-1.30.0.tgz)
cp "$tmp"/package/dist/ort-wasm-simd-threaded.{mjs,wasm} "$tmp"/package/dist/ort-wasm-simd-threaded.asyncify.{mjs,wasm} .
echo "Runtime ready. Serve this folder (python -m http.server) and open index.html."
