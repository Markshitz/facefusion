#!/usr/bin/env bash
# Fetch the ONNX Runtime Web files index.html loads from its own folder:
# CPU runtime (always) and the WebGPU (JSEP) runtime, used when the browser has a GPU.
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
(cd "$tmp" && npm pack onnxruntime-web@1.30.0 --silent >/dev/null && tar xzf onnxruntime-web-1.30.0.tgz)
cp "$tmp"/package/dist/ort-wasm-simd-threaded.{mjs,wasm} "$tmp"/package/dist/ort-wasm-simd-threaded.asyncify.{mjs,wasm} .
# Artifacts serve files up to 15 MB: also publish the WebGPU runtime as parts (index.html joins them).
python3 - <<'PY'
import hashlib, json, os
data = open("ort-wasm-simd-threaded.asyncify.wasm", "rb").read(); os.makedirs("runtime", exist_ok=True); P = 14 << 20; parts = []
for i in range(0, len(data), P):
    n = f"gpu.part{i // P:02d}.wasm"; open("runtime/" + n, "wb").write(data[i:i + P]); parts.append(n)
json.dump({"parts": parts, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, open("runtime/gpu.json", "w"), indent=1)
PY
echo "Runtime ready. Serve this folder (python -m http.server) and open index.html."
