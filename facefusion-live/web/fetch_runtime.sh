#!/usr/bin/env bash
# Fetch the ONNX Runtime Web WASM runtime that index.html loads from its own folder.
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
(cd "$tmp" && npm pack onnxruntime-web@1.20.1 --silent >/dev/null && tar xzf onnxruntime-web-1.20.1.tgz)
cp "$tmp/package/dist/ort-wasm-simd-threaded.mjs" "$tmp/package/dist/ort-wasm-simd-threaded.wasm" .
echo "Runtime ready. Serve this folder (python -m http.server) and open index.html."
