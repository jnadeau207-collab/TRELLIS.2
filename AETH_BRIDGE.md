# Aeth bridge

`aeth_bridge` is a small, generic, newline-delimited JSON service around TRELLIS.2 generation and deterministic mesh evidence. It contains no Aethalgard source, private schemas, private prompts, or private datasets.

## Commands

Each stdin line is a versioned request. Each stdout line is a versioned event. Progress and library output go to stderr.

```json
{"version":1,"requestId":"p1","command":"probe","payload":{}}
{"version":1,"requestId":"g1","command":"generate","payload":{"imagePath":"input.png","profile":"shape-512","seed":42}}
{"version":1,"requestId":"a1","command":"analyze","payload":{"path":"shape-0.npz"}}
{"version":1,"requestId":"c1","command":"compare","payload":{"referencePath":"reference.npz","candidatePath":"candidate.npz"}}
{"version":1,"requestId":"x1","command":"cancel","payload":{"targetRequestId":"g1"}}
{"version":1,"requestId":"s1","command":"shutdown","payload":{}}
```

Large geometry and latent data are written as immutable, content-addressed artifacts; JSON carries paths, hashes, compact reports, and diagnostics.

## Windows

The supported Windows execution path is WSL2. The Windows desktop process launches the bridge through `wsl.exe`, so no HTTP server or open port is required and the same stdio protocol is used on every platform. Windows drive paths in requests are translated to their `/mnt/<drive>/...` WSL equivalents at the bridge boundary.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_aeth_bridge_windows.ps1 -InstallInference
powershell -ExecutionPolicy Bypass -File scripts/run_aeth_bridge_windows.ps1
```

Analysis and comparison are CPU-portable and are tested directly on Windows. TRELLIS.2 CUDA inference runs inside WSL2, where its Linux CUDA extensions are supported.

## Shape-only and 12 GiB-class GPUs

`shape-512` is the first execution mode for a 12 GiB-class GPU. It forces TRELLIS.2's low-VRAM one-model-at-a-time residency and loads only sparse-structure and shape models; texture flow and texture decoding are absent. `shape-1024` uses the same strategy with a bounded cascade.

The bridge does not convert total VRAM into a promise. `probe` verifies software, CUDA, driver, and visible-device prerequisites and reports the measured total/free memory. The actual generation run qualifies the machine. A CUDA allocation failure returns `insufficient_vram` with before/failure memory telemetry, while a successful manifest records peak allocation and reservation. No profile is silently substituted.

## Local checks

```bash
python -m pip install -r requirements-aeth-bridge.txt pytest
python -m pytest tests/test_aeth_bridge.py -q
python -m aeth_bridge --probe
```
