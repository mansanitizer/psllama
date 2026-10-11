# PS5 LLM inference: findings and current state

Handover notes for whoever picks this up next (Codex, Claude, or a person).
Last updated 2026-10-09. Everything below was measured on the hardware unless
it says otherwise.

## Goal

Run standard downloaded GGUF models on the PS5 GPU, with no custom model
format, and find the largest model that fits.

## Hardware and access

| Thing | Value |
|---|---|
| Console | Base PS5, CFI-1208B (not a Pro), firmware 13.600.007 |
| PS5 address | `10.42.0.211`, on Aisha's Ethernet port |
| Build host | Aisha, Linux, `ssh user@build-host.example` (Tailscale), key auth |
| Aisha internet | Wi-Fi is very slow when another download is running. A phone USB tether (`enx02206a0a0f74`) is much faster; check `ip route` to see which is the default |
| PS5 services | FTP 2121, shsrv 2323, klogsrv 3232, web/homebrew launcher 8080, Payload Manager 8084 |

### Remote dev loop (no console interaction needed)

```
# kernel log (stdout of native apps goes here via sceKernelDebugOutText)
nohup timeout 1800 nc 10.42.0.211 3232 </dev/null >/tmp/run.log 2>&1 &
# shell: ps, kill, launch, ls, mv, ln, stat, procstat ...
printf "ps\nexit\n" | nc 10.42.0.211 2323
printf "kill -s KILL <pid of PPSA99005>\nexit\n" | nc 10.42.0.211 2323
printf "launch PPSA99005\nexit\n" | nc 10.42.0.211 2323
# deploy a file
curl -T eboot.bin ftp://10.42.0.211:2121/data/homebrew/PPSA99005/eboot.bin
```

Gotchas:
- Never `pkill -f <pattern>` over ssh when the pattern also appears in the ssh
  command line: it kills its own session. Use `ps | grep "^[a]ria2c"`-style
  patterns.
- Always resolve the PID with `ps` before `kill`, and kill only `PPSA99005`.
- shsrv has no `cp`-with-progress; `mv` and `ln` within `/data` are instant.

### Registering new titles is broken until reboot

ShadowMount fails to register any new title with `internal AppInstallTitleDir
bridge unavailable`, then hits its retry limit. It worked once right after a
fresh reboot and re-jailbreak. Workaround: reuse the already-registered slot
**`PPSA99005`** (shows as "AGC Correctness Probe") and swap its contents.

## Approach 1: custom AGC GGML backend (paused)

Code: `prosperoai-audit/src/ps5_agc_backend.cpp`, probes in this folder
(`agc_smoke.cpp`, `q4k-dot.s`, `gguf_inference.cpp`, `build-native-probe.sh`).

Verified on hardware:
- Native AGC init, FP16 GEMM: 20/20 tests exact (shapes up to inference size,
  odd boundaries, repeated dispatch). Fix needed: pad FP16 buffers to whole
  32-bit words.
- Hand-written Q4_K dot kernel matches the pinned GGML decoder to ~4e-6.
- Native app runtime issues fixed: missing `icon0.png` blocks registration;
  AGC must be linked directly, not dlopen'd; calling `exit()` crashes the app,
  so probes stay alive after finishing.

GGUF inference harness (`gguf_inference.cpp`, `PS5_PROBE_MODE=inference`):
- Links llama.cpp (PS5LM's pinned copy) into a native app.
- **SIGILL at model load was `operator new` failing**, which
  `app_cpp_runtime.cpp` turns into `__builtin_trap`. Fix: call
  `ps5SetDirectFallback(1)` so allocations over 2 MB use the 4 GiB direct-memory
  arena. Failed `new` now logs `operator new failed, bytes=N`.
- C++ exceptions unwind correctly in native apps (self-test passes).
- `ggml_set_abort_callback` and a SIGILL/SIGSEGV handler are installed.
  `si_addr` is reliable; the ucontext register layout is NOT standard FreeBSD.
  To map a fault: `fault_link = si_addr - main_runtime + main_link`.
- The model loads but takes about **37 minutes** for the 0.8B Q4_K_M, 10% every
  ~220 s. Not yet explained; see measurements below. Token generation never
  reached.
- On-screen status panel (`probe_screen.h`) shows steps, load % with time,
  tokens and speeds.

Status: paused in favour of approach 2. Keep it as a fallback.

## Approach 2: llama.cpp ggml-vulkan on PS5_Vulkan RADV (current)

Project: [mihawk-99/PS5_Vulkan](https://github.com/mihawk-99/PS5_Vulkan) plus
[PS5_Mesa](https://github.com/mihawk-99/PS5_Mesa). Mesa RADV with a PS5 winsys
over AGC, Vulkan 1.4, GPL-3.0-or-later. Very active, single author.
- Claims (from its README, verified by reading): the 2nd full Vulkan 1.4 CTS run
  had 1,569,390 pass, 0 fail. Compute and transfer queues run on the graphics
  ring; true async compute isn't implemented. No buffer device address,
  cooperative matrix or integer dot product. The GPU is modelled as Mesa
  **GFX1013** (base PS5 family). Tested on firmware 6.02 and 12.70 (mostly a
  Pro), plus a base PS5 on 13.40. Submissions stalled for 10 s give
  `VK_ERROR_DEVICE_LOST`.

**Verified on our console (base PS5, 13.60), 2026-10-08:** the
PS5_VulkanTemplate v1.000.000 samples title runs in the PPSA99005 slot. All
samples tried run at ~60 fps with no faults, including **computeparticles**
(4809 frames in 80 s), rayquery, deferred, gltfloading, indirectdraw, imgui and
uikit.

Release assets (downloaded and SHA256-verified on Aisha in
`~/ps5-vulkan-release/`):
- `PS5_VulkanTemplate-v1.000.000.zip`: the samples app folder PPSA99130,
  prebuilt.
- `radv-*.tar.gz`: **Mesa source, not a prebuilt library.** The driver must be
  built (PS5_Vulkan `tools/build-radv.sh`; documented hosts are Arch/CachyOS,
  and Aisha is Debian/Ubuntu-like).
- `ps5-vulkan-*.tar.gz`: link recipe (`tools/radv-link.sh`), CRT, AGC stubs,
  `libc.prx`.
- `platform-*.tar.gz`, `radv-sdk-*.tar.gz`: the payload SDK fork RADV builds
  against.

Next steps:
1. Build RADV (`libvulkan_radeon.ps5.a`) on Aisha from those archives.
2. Cross-build llama.cpp with `-DGGML_VULKAN=ON` (needs host `glslc` for its
   shaders) and link it as `radv-link.sh` does.
3. Run `test-backend-ops -b Vulkan0` on the console, then the Qwen GGUF. Use a
   small `-ub` so one submit stays well under the 10 s watchdog.
4. Then serve it (llama.cpp server or PS5LM's) from the native app.

## Measurements

CPU baseline (PS5LM payload, llama.cpp CPU): Qwen3.5 0.8B Q4_K_M generates
7.27 tok/s with 47.9 tok/s prefill. TinyStories 15M: 244 tok/s.

CPU memory bandwidth inside a native app (64 MiB, 2 passes):

| Memory | Write | Read |
|---|---|---|
| Flexible (`sceKernelMapNamedFlexibleMemory`, prot 0x03) | 11.3 GB/s | 10.7 GB/s |
| Direct type 0 / 1 / 12, prot 0x03 or 0x33 | ~11 GB/s | ~10.8 GB/s |
| Direct type 2 / 3 | 26 GB/s | **49 MB/s** (uncached for CPU reads) |
| Direct type 10 | refused (0x8002000d) | |

So type 12 / 0x33 (the fallback arena) is NOT the cause of the slow model load.
Never read GPU buffers back on the CPU from type 2/3 memory.

File read inside the app: `/app0/model.gguf` gives 2.7 GB/s (8 MB reads) and
4.5 GB/s (64 KB reads). `/data/...` is not visible from the app sandbox
(errno 2); put models in the app folder (`ln` or `cp` via shsrv). The slow load
is therefore CPU-side work in the loader (suspects: CPU_REPACK of Q4_K, or the
allocator's per-allocation kernel calls). Unresolved.

Other facts:
- An app's flexible heap is about 448 MiB. Large buffers must go to direct
  memory.
- The PS5's CPU is Zen 2 (AVX2, FMA, F16C, no AVX-512); llama.cpp's CPU flags
  in PS5LM are correct.
- The Qwen3.5 0.8B Q4_K_M file also contains Q5_K, Q6_K, Q8_0 and F32 tensors,
  not just Q4_K.

## Where things are on the PS5

- `/data/homebrew/PPSA99005/`: currently holds the **Vulkan samples** (eboot,
  sce_module, assets, shaders) with our original `sce_sys`.
- `/data/agc-probe-backup/`: our GGUF probe's `eboot.bin`, `sce_module/`,
  `model.gguf`. To restore, `mv` them back and move the Vulkan files out.
- `/data/homebrew/PPSA99130/`: leftovers of the unregistered samples title
  (licenses, sce_sys).
- `/data/PS5LM/models/Qwen3.5-0.8B-Q4_K_M.gguf`: the model, used by the PS5LM
  payload.

## 2026-10-09: GGUF inference on the GPU works (Vulkan/RADV)

Title: `ps5lm-tests/vk/` (`gguf_vulkan.cpp`, `status_screen.h`, `platform_shims.c`,
`build-gguf-vk-title.sh`), deployed in the PPSA99005 slot with the model at
`/data/homebrew/PPSA99005/model.gguf`. The Vulkan Template files are in
`/data/vk-template-backup`; the AGC probe's files are in `/data/agc-probe-backup`.

Built on Aisha (Debian 13, LLD 18):
- RADV release archive: `~/ps5-vulkan-release/ps5vk/ps5-vulkan-*/.deps/native/radv-release/lib/libvulkan_radeon.ps5.a`
  (`tools/build-radv-offline.sh` = `build-radv.sh` without the git check; Mesa
  source from the release tarball; run ninja with `-j3`).
- llama.cpp + ggml-vulkan: `~/llama-vk/build` (prospero-cmake from the SDK fork,
  isolated Vulkan/SPIR-V headers in `~/llama-vk/hdr`, `-j2`).
- Link fixes: alias `vkGetInstanceProcAddr` to `vk_icdGetInstanceProcAddr` and 3 direct
  vk* calls to `vk_common_*`; stub `execlp`/`posix_madvise`; no `--gc-sections`;
  LLD 18 exports Mesa's 6743 unresolved weak dispatch refs, so they're bound to 0
  in a second link pass (the same failure hits PS5_Vulkan's own smoke title on LLD 18).
- Aisha is a 2c/4t i3 with ~3.5 GB free: run one build at a time.

Measured, first run, Qwen3.5 0.8B Q4_K_M, all layers on Vulkan0:
- Device: "PlayStation 5 GPU (RADV NAVI21)", fp16, UMA, no matrix cores,
  **12,215 MiB free of 12,288 MiB** heap.
- Model load + context: 10.1 s (vs ~37 min in the ProsperoAI-runtime app).
- **Generation 133.7 tok/s** (CPU baseline 7.27, about 18x). Prefill 40.8 tok/s on a
  19-token prompt (includes first-use pipeline compiles; needs a warm rerun).
- Coherent output. The CPU comparison run in the same title was in progress at
  the time of writing.

Next: read the CPU run's result; fix the harness's GPU-type check (it reports
IGPU, which the harness treats as "no GPU"); warm-run benchmarks; test-backend-ops;
then larger models (12 GB heap) and a network server.

CPU comparison run (same title, n_gpu_layers=0): load 4.7 s, prefill 95.0 tok/s,
**generation 41.4 tok/s**, also coherent. This is not a clean CPU baseline:
ggml still reserved a 326 MiB Vulkan0 compute buffer (op offload is on by
default), so some ops may have run on the GPU. Rerun with `op_offload=false`.
Vulkan vs this run: generation 3.2x faster; prefill 0.4x (the Vulkan prefill
included first-use shader compiles). Greedy tokens matched the CPU for the first
14 of 32, then diverged (fp16 vs CPU numerics); both answers are correct sentences.
## 2026-10-09: first larger-model rung (Qwen2.5-Coder 3B)

Using the existing `qwen2.5-coder-3b-instruct-q4_k_m.gguf` (2.1 GB) from Aisha's
`~/models/llmfit/`, swapped into the reusable `PPSA99005` slot:

- Vulkan model load: 34.1 s; prefill 162.1 tok/s; generation 136.79 tok/s.
- CPU comparison: model load 32.2 s; prefill 23.4 tok/s; generation 16.30 tok/s.
- Vulkan speedup: 6.9x prefill, 8.4x generation.
- Both runs completed coherently; greedy tokens matched for 23 of 29 generated
  tokens. Divergence is expected from GPU/CPU numerical differences.
- The status panel remained enabled and showed load progress, phase, text and
  final speeds on the PS5 screen.

## HTTP inference endpoint

The single-model OpenAI-compatible API is hosted by the native Vulkan title.
`serve_model.py` is only an HTTP range server for transferring a model and
must not be confused with inference serving.

### Architecture

1. Keep one `llama_model` and one `llama_context` alive for the title lifetime;
   serialize requests in the accept loop.
2. The HTTP/1.1 listener is bound to `127.0.0.1:8081`. It caps headers and
   bodies, handles partial reads/writes, applies socket timeouts, and rejects
   transfer encoding other than identity.
3. Context is fixed at 1024 tokens; output defaults to 128 and is capped at
   256. The server checks prompt plus output against context before decoding.
   Greedy sampling is default; temperature sampling is supported.
4. Implemented `GET /health`, `GET /v1/models`, and non-streaming
   `POST /v1/chat/completions`, with token usage and prompt/generation rates.
5. The status panel reports model loading, request phases, generation progress,
   completion, and errors; messages are also sent to klogsrv.

### Network and access

- PS5Tailscale v0.7.1 is running and the device is online as
  `host.example.ts.net` at `ps5.tailnet.example`.
- The user plans to set a status-page password and restrict access later. Until
  then, the model server must bind only to loopback and must not be advertised
  through the Tailscale port forward.
- Once the user confirms access controls are set, bind the inference listener
  to all interfaces (or configure an explicit TCP forward in PS5Tailscale) and
  verify LAN and tailnet access. The anticipated client URL is
  `http://ps5.tailnet.example:8081/v1/chat/completions`.
- HTTP is acceptable inside Tailscale because the tailnet encrypts transport;
  do not expose port 8081 on the public internet or add router port forwarding.
- Add optional Bearer-token authentication in the inference server before
  opening it to tailnet devices. Keep the token out of source control and
  configure it through a local config file or app settings.

### Deployment and remaining work

Built on Aisha and deployed to the existing `PPSA99005` slot on 2026-10-09.
Before replacing `eboot.bin`, the prior active executable was copied to
`/home/user/ps5lm-inference/vk/out/eboot.before-http.bin` (SHA256
`9a6ebb9c033c0a8e222389a901f065fc609a9b1131d1e376960005b2c167a753`). The
model file was not modified. The app loaded `qwen2 3B Q4_K - Medium` in 22 s,
logged `HTTP READY 127.0.0.1:8081`, and the PS5's built-in HTTP client received
a successful `/health` response (confirmed by `HEALTH CHECK OK` in klogsrv).
`/v1/models` also returned through the listener and was recorded as
`MODEL LIST REQUEST`. After the ready text scrolled off-screen, the status
panel was changed to pin the current headline at the top. Display diagnostics
then found `sceVideoOutRegisterBuffers2` failing with `0x80290001`: the panel
registered one framebuffer, while the working PS5_Vulkan renderer registers
two. It now allocates/registers two framebuffers and submits/waits for flips on
updates. Rebuilt on Aisha and redeployed; the latest startup produced no
status-screen initialization error. The service is intentionally not reachable
from LAN/Tailscale yet.

On each startup, the app now sends a short local TCP POST through the bound
listener to `/v1/chat/completions`, exercising the request parser, prompt
formatting, GPU inference, and JSON response. Latest run returned HTTP 200 with
a chat-completion object and logged `POST SELF-TEST PASSED`; latest run generated
one token at 95.9 tok/s. The test runs entirely on the PS5 and the server remains
loopback-only. Remaining: add Bearer authentication before binding beyond
loopback; enable/test tailnet access only after the user sets PS5Tailscale
status-page password and access rules. SSE, `/v1/completions`, malformed/oversize
tests, and tuning are future work.

### LAN curl test build and Qwen2.5-7B staging (2026-10-09)

Per user request, the HTTP listener now binds specifically to the PS5 LAN
address `192.168.1.16:8081` so a Mac on the same LAN can exercise the real
`POST /v1/chat/completions` route. It does not bind to `INADDR_ANY` and is not
advertised through Tailscale; there is no authentication yet, so keep this LAN
test temporary and do not add a router forward. The startup POST self-test now
connects to the LAN address as well. Built on Aisha (never on the Mac),
`out/eboot.bin` SHA256:
`ac79355c7de2681359f7deee3459a9daf0e00d5734f0f31c1e9b5302877e5cf0`.

Qwen2.5-7B-Instruct Q4_K_M is split into two approximately 3.8 GiB files.
Shard one was uploaded to
`/data/homebrew/PPSA99005/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
and verified with PS5 `stat` at 3,993,201,344 bytes. The existing
`/data/homebrew/PPSA99005/model.gguf` (the 3B backup) remains untouched. The
user requested a pause after shard one to switch the connection to Ethernet;
shard two must not be sent and the new title must not be launched until the
user confirms the switch is complete.

The false `NO VULKAN GPU DEVICE` warning is fixed in
`vk/gguf_vulkan.cpp`: both `GGML_BACKEND_DEVICE_TYPE_GPU` and
`GGML_BACKEND_DEVICE_TYPE_IGPU` count as GPU devices. The current built title
includes this fix; obtain a fresh PS5 startup klog after launch to confirm the
warning is absent at runtime.

The model queue on Aisha is aria2 and uses `-c` for continuation. The active
Qwen14B shard was safely stopped and resumed; the remaining Qwen14B and Qwen32B
queue is now pinned to Wi-Fi source address `192.168.1.24`. Live `ss` sockets
confirmed the model aria2 process used that Wi-Fi address. The paused Evil
Within aria2/RPC queue was not changed.

## 2026-10-09: SSE streaming and load-time context/output settings

Latest active title is built on Aisha and deployed to `PPSA99005`:
`eboot.bin` SHA256
`abe6af6ec17f19e95739b1993552b324dcd19a63f35a49880633a9ca5c9e898b`.
The previous working executable is preserved on Aisha as
`vk/out/eboot.before-streaming-20261009.bin` (SHA256
`ed1cb326e37b99e8a9c851d67fa5a33fc64d4f7d8ab07eb53e6cc6b3bef8f753`).

The current app reads `/app0/inference.conf` before creating the llama context.
Settings are `context_tokens=4096` and `max_output_tokens=1024`; when the file
is absent, safe defaults remain 1024/256. Context is clamped to the model's
trained context length, and output is clamped below context. `GET /health`
reports both effective values. The title's status screen shows the settings
before model loading.

`POST /v1/chat/completions` accepts `"stream": true` and responds with
`text/event-stream` using HTTP chunked transfer encoding: role delta, content
deltas, finish-reason chunk, then `[DONE]`. Without `stream` it retains the
non-streaming JSON response. `GET /health` and `/v1/models` are unchanged.
Global SIGPIPE is ignored so a client closing a stream does not terminate the
title. Mac-to-PS5 test passed with 512 output tokens: HTTP 200, 512-token
completion, `finish_reason=length`, final `[DONE]`; the PS5 logged 70.4 tok/s.
Follow-up health check returned `context_tokens=4096` and
`max_output_tokens=1024`; the process remained alive.

With Qwen2.5-7B Q4_K_M, the 4K context used a 224 MiB Vulkan KV buffer; the
prior 1K context used 56 MiB. Model weights are 4168.09 MiB and compute buffer
77.75 MiB. The 4K context is working on hardware.

The model-download queue is active on Aisha's Wi-Fi interface
`192.168.1.24`; Qwen2.5-14B shard 1 reached 39% (about 1.4 GiB) with an
observed ~1.9 MiB/s and ~20-minute shard ETA at the last check. Shards 2 and 3
are queued behind SHA-256 verification. Qwen32B Q4 was not started because its
weights exceed the PS5's ~12 GiB Vulkan heap; revisit only with a substantially
smaller quant. The separate Evil Within download remains paused and untouched.

## 2026-10-09: compact llmfit PS5 port (built; launch format unresolved)

Added `llmfit-ps5/llmfit_ps5.cpp` and `llmfit-ps5/build.sh`: a reduced,
offline C++ fit estimator (not the upstream Rust application). It uses the
previously measured 12,288 MiB Vulkan heap, Q4_K_M 0.58 bytes/parameter, an
8K-context KV fallback, and a 0.5 GiB runtime reserve. The estimates are about
2.38 GiB (3B), 5.10 GiB (7B), 9.46 GiB (14B), and 20.36 GiB (32B): first three
fit the nominal heap, while 32B does not. These are planning estimates, not
proof the 14B model loads; the utility explicitly notes heap availability is
not live and runtime/model-specific allocations may change the result.

The utility compiled and signed successfully on Aisha using the installed
Prospero toolchain (7,014-byte SELF), and was uploaded as the separate file
`/data/homebrew/PPSA99005/llmfit-ps5.bin`. `shsrv exec` created process 691 but
left it stopped before any report was emitted; that exact utility PID was then
terminated. `hbldr` rejected the same file as a malformed ELF. No inference
process was stopped or replaced: PPSA99005 PID 670 remained active. The utility
is not yet verified runnable through the console shell; resolve the compatible
ELF/SELF launch format before retrying. Its PS5 file is still present at the
path above.

## 2026-10-09: download monitor progress display

`monitor-downloads.sh` now shows a 30-character progress bar, the last aria2
percentage and transferred/total amount, rate, ETA, log freshness, and a
three-shard checksum/queue checklist. It reads aria2's progress line rather
than treating a preallocated GGUF's file length as downloaded progress. The
bar is explicitly marked last-known/stale when its log has not updated for
three minutes. Verified live against Aisha: shard 1 SHA-256 passed; shard 2
was in progress at 95% (3.5/3.7 GiB, 865 KiB/s, aria2 ETA 3m29s); shard 3
was queued. That reading is a snapshot, not a completion verification.

## 2026-10-09: Qwen2.5-14B three-shard download completed

On Aisha, the queue controller and aria2 shard transfer had exited. All three
Qwen2.5-14B-Instruct Q4_K_M shard files were present and SHA-256 verified
against the queued expected digests: shard 1 `a09ea5e7…3904f68`, shard 2
`21b9457d…ba41ae`, and shard 3 `c8d37006…3d9976`. The prior checksum-failed
backup for shard 1 remains in the directory; it was not used for this
verification. This confirms the three current shard files only; no PS5 upload,
model load, or benchmark was performed as part of the download monitor.

## 2026-10-09: Qwen2.5-14B loaded and benchmarked on the PS5

Added validated `model_file` selection to `vk/gguf_vulkan.cpp`, preserving the
7B shard as the default, then built the new title on Aisha. The PS5 has the
14B title/config and all three original shard filenames in the existing
`PPSA99005` app folder; previous PS5 `eboot.bin` and `inference.conf` are
preserved there as `eboot.before-qwen14b.bin` and
`inference.before-qwen14b.conf`, and copies are also saved in Aisha's
`vk/out/qwen14b-20261009/`.

On-device launch succeeded. The status panel showed model-load progress to
100% in 91 seconds. llama.cpp identified `qwen2 14B Q4_K - Medium`, with
8,148.38 MiB Vulkan model weights, 417.66 MiB Vulkan-host weights, and a
768 MiB Vulkan KV cache at a 4,096-token context. Its startup POST self-test
passed. `/health` returned `status=ok`, `model_loaded=true`,
`context_tokens=4096`, and `max_output_tokens=1024`; the service is listening
on `0.0.0.0:8081` (LAN only per the title log).

One real completion from Aisha returned coherent text, 80 completion tokens,
155.264 prompt tokens/s, and 38.048 generation tokens/s. This is an initial
single-run measurement, not a warmed/repeated benchmark. The active PS5 title
process was still present after the request. The PS5 HTTP endpoint used for
testing is `http://ps5.tailnet.example:8081/v1/chat/completions`.

### Multi-Prompt & Workload Benchmarking (2026-10-09)

Warmed, multi-domain benchmark run via `http://ps5.tailnet.example:8081/v1/chat/completions`:

| Test Suite / Prompt | Output Tokens | TTFT (ms) | Decode Speed (tok/s) | Prefill Speed (tok/s) | Result Quality |
|---|---|---|---|---|---|
| **Raw Decode Throughput** (Cyberpunk Scene) | 256 | 625.7 ms | **35.82 tok/s** | ~130–155 tok/s | Full creative narrative, zero repetition loops |
| **Algorithmic Code** (Thread-safe LRU Cache) | 300 | 552.7 ms | **25.03 tok/s** | ~140 tok/s | Accurate Python types, `OrderedDict`, `threading.Lock` |
| **Logical Reasoning** (Bat/Ball & Token count) | 200 | 751.5 ms | **38.27 tok/s** | ~135 tok/s | Solved $2x + 1.00 = 1.10 \implies x = 0.05$ accurately |
| **Tool Calling / Function Invocation** | 128 | ~680 ms | **38.17 tok/s** | **127.63 tok/s** | Formatted structured arguments `{"operation": "multiply", ...}` |

**Averaged sustained decode throughput:** **~34.3 tokens/second**.
**Time To First Token (TTFT):** **~550–750 ms**.
Memory footprint remained stable inside the 12,288 MiB heap with zero Vulkan memory leaks across runs.

## 2026-10-09: Qwen3.8-27B Unsloth quant staged for PS5 evaluation

Unsloth's official `unsloth/Qwen3.8-27B-GGUF` repository offers Dynamic v3
GGUFs. The selected `Qwen3.8-27B-UD-IQ2_S.gguf` is listed at 8.37 GB; the
smaller UD-IQ2_XXS is 7.27 GB. Chose IQ2_S as the higher-quality of these
memory-conscious options, while keeping 4K context initially to leave room for
runtime allocations. The model card identifies the GGUF architecture as
`qwen3_5` (Qwen3.5-family).

Aisha's `llama.cpp` checkout is `552f18f91` (tag `b11327`, 2026-10-01), and
includes `src/models/qwen35.cpp`; its Vulkan backend sources include IQ2
quantization kernels. This is source-level compatibility evidence, not yet a
successful PS5 model load.

Started a resumable aria2 download to
`/home/user/models/llmfit/qwen38-27b/Qwen3.8-27B-UD-IQ2_S.gguf`, bound only to
Aisha's two Wi-Fi adapters (`wlx58044f3f6ec0`, `wlx90de80301119`). At the first
live sample, aria2 reported 82 MiB of 7.7 GiB, 2.8 MiB/s, and about 46 minutes
remaining; the `.aria2` control file was present. These are live transfer
metrics, not the preallocated GGUF length. The PS5's Qwen2.5-14B service was
left untouched; hash verification, deployment, and on-device load/benchmark
are still pending.

## 2026-10-09 05:20 IST: overnight model queue and benchmark plan

Read the user's attached `pasted-text-1.txt` benchmark suite. It issues four
streaming workloads (creative output, Python LRU implementation, bat/ball plus
letter counting, and inverted-cup spatial tracking) to the existing PS5
`/v1/chat/completions` endpoint. Like the browser chat UI, its counter counts
non-empty SSE content deltas, not tokenizer-confirmed tokens; record reported
throughput as streamed delta/token estimate unless server usage supplies an
authoritative completion-token count.

Created and copied `ps5lm-tests/overnight-qwen-queue.sh` to Aisha as
`~/ps5lm-inference/overnight-qwen-queue-20261009.sh`, then started persistent
controller PID 502620. It waits for the in-progress Qwen3.8-27B UD-IQ2_S aria2
PID 496842, checks its exact size and Hugging Face LFS SHA-256, then downloads
and verifies these additional candidates sequentially on Wi-Fi:

| Candidate | Expected bytes | Expected SHA-256 |
|---|---:|---|
| Unsloth Qwen3.5-9B Q4_K_M | 5,680,522,400 | `16835097…dab4e50e` |
| Unsloth Qwen3-14B Q4_K_M | 9,001,753,184 | `0d14496b…f5c01fe` |

At queue setup Aisha had 77 GB free; no USB backup is currently needed. The
existing active 15-minute heartbeat was repurposed to track the whole
authorized overnight load/benchmark sequence (including the user's required
PS5-screen status), rather than the already-finished Qwen2.5-14B download.
Qwen2.5-14B remains the known-good PS5 service while files download. Model
deployment, load, and benchmark results remain pending.

Latest live snapshot at 2026-10-09 05:39 IST: first aria2 process PID 496842
and queue controller PID 502620 were both alive. Transfer was 4.0 GiB / 7.7
GiB (52%), about 2.7 MiB/s, aria2 ETA about 23 minutes. The GGUF length is
preallocated, so this percentage comes from aria2's transfer line, not `ls`.
Aisha reports 69 GB free after preallocation; the remaining planned downloads
total about 14.7 GB, so no USB cleanup is needed at present. The PS5 health
endpoint remains `model_loaded=true` for the existing Qwen2.5-14B service.
The separate existing aria2 RPC daemon reports 0 active transfers and 1
waiting item, so the current lower rate is not due to that daemon competing.

Compatibility check on Aisha: `ar t ~/llama-vk/build/src/libllama.a` shows a
Unity build, and `nm -C` finds linked `llama_model_qwen35` and
`llama_model_qwen35moe` loader/graph symbols. Thus the deployed cross-build's
static llama library includes the Qwen3.5-family architecture code needed by
Qwen3.8, not merely the source checkout. Runtime GGUF loading is still the
remaining compatibility test.

PS5 staging preflight: its user-data volume has 474.8 GB free. Direct TCP
management on the old LAN address `10.42.0.211` timed out, but the PS5's
Tailscale management ports work and `tailscale ping` from Aisha reports a
direct peer via `10.42.0.211:56139`; this is the wired Ethernet path. FTP
listing at `ps5.tailnet.example:2121/data/homebrew/PPSA99005/` confirmed the app
folder and existing live model/config files. A separate staging watcher
(`~/ps5lm-inference/stage-qwen38-to-ps5-20261009.sh`, PID 515260) waits for the
queue controller's SHA-verified marker, rechecks size/hash, then uploads the
new Qwen3.8 GGUF to a unique filename and checks the remote FTP file size. It
does not change the live title/config. The recurring 15-minute automation was
updated to check this watcher and avoid a duplicate upload.

Prebuilt on Aisha to reduce swap latency:
`~/ps5lm-inference/vk/out/qwen38-20261009/eboot.bin` built and signed
successfully (73,486,124 bytes). Its SHA-256 matches the current Qwen2.5-14B
title's existing build exactly (`db40ae2b…e83775d89`), so no executable
replacement is necessary; the existing binary already contains the same
loader. Three separate 4K runtime configs with `model_file` selected for
Qwen3.8-27B, Qwen3.5-9B, and Qwen3-14B are staged in the Aisha output's
`configs/` subdirectory, not on the PS5 yet.

At 2026-10-09 05:41 IST the Qwen3.8 transfer was 4.4 GiB / 7.7 GiB (56%),
2.0 MiB/s, aria2 ETA about 28 minutes; download, queue controller, and PS5
staging watcher were all alive. The PS5 /health remained healthy on the
existing 14B model.

Live checkpoint at 2026-10-09 06:07 IST: Qwen3.8-27B UD-IQ2_S completed and
the queue controller logged a match for both expected size (8,371,970,048
bytes) and Hugging Face SHA-256
`7897d2c5a5cee46aef50895141b2c8a0803c1185f3d03c4fda4cd137a7ad77fe` at
06:06:53 IST. The queue immediately started the pinned Qwen3.5-9B Q4_K_M
download over Aisha Wi-Fi; the Qwen3-14B remains next. Queue controller PID
502620 and PS5 stage watcher PID 515260 remain alive. The watcher independently
rehashed Qwen3.8, then began its wired Tailscale FTP upload to the unique file
`/data/homebrew/PPSA99005/Qwen3.8-27B-UD-IQ2_S.gguf`; at 06:08 IST it was about
23% complete. Qwen3.5-9B is downloading concurrently over Wi-Fi. Aisha then had
63 GB free; no USB backup is needed. Until a model swap, the existing
Qwen2.5-14B PS5 service remains the known-good live model.

### Attached streaming benchmark baseline: Qwen2.5-14B (2026-10-09 05:27 IST)

Ran the user's attached four-workload Python suite against the existing PS5
endpoint before swapping models. All four SSE requests completed; the script's
"tokens" are content-delta events (not tokenizer-confirmed counts).

| Workload | SSE deltas | TTFT | Generation time | Reported delta/s |
|---|---:|---:|---:|---:|
| Creative throughput | 242 | 2,236 ms | 6.71 s | 35.89 |
| Python LRU code | 300 | 813 ms | 12.02 s | 24.87 |
| Bat/ball reasoning | 200 | 1,495 ms | 5.10 s | 38.98 |
| Inverted-cup spatial | 117 | 827 ms | 3.07 s | 37.73 |

Post-run `/health` remained `status=ok`, `model_loaded=true`, 4,096 context.

## 2026-10-09: Qwen3.8-27B loaded and streaming benchmarked

The verified `Qwen3.8-27B-UD-IQ2_S.gguf` (8,371,970,048 bytes; source
SHA-256 checked twice on Aisha) was uploaded to the existing PPSA99005 app
folder over Aisha's wired Tailscale path; the PS5 FTP listing matched the exact
byte size. The previous Qwen2.5-14B runtime config was preserved as
`inference.before-qwen38.conf`. The prepared Qwen3.8 config selected the unique
model filename, 4,096 context, and 1,024 max output.

On-device load succeeded in 89 seconds. The PS5 status panel visibly reported
`LOADING MODEL` every 10%, then `MODEL LOADED`, followed by `qwen35 27B IQ2_S -
2.5 bpw`. llama.cpp assigned and offloaded all 65/65 layers to Vulkan0. The
reported Vulkan model buffer was 7,575.79 MiB and Vulkan-host weights were
397.85 MiB; startup reported 12,199 MiB free of the 12,288 MiB Vulkan heap.
The application process used 9,390.5 MiB after load. `/health` reports the
model loaded at 4,096 context and 1,024 max output.

A short 8-token smoke request returned a response at 21.52 API tokens/s but
was intentionally cut off by its tiny cap (it returned the model's visible
`<think>` prefix). The user's attached four-workload streaming suite completed
all four requests against Qwen3.8:

| Workload | SSE deltas | TTFT | Generation time | Reported delta/s |
|---|---:|---:|---:|---:|
| Creative throughput | 256 | 2,568 ms | 12.84 s | 19.86 |
| Python LRU code | 300 | 1,058 ms | 20.37 s | 14.68 |
| Bat/ball reasoning | 200 | 1,367 ms | 9.40 s | 21.17 |
| Inverted-cup spatial | 200 | 1,222 ms | 9.52 s | 20.90 |

As with the baseline, these are SSE content-delta counts, not tokenizer-verified
token counts. The model streams visible `<think>` content; in the capped LRU
workload it used much of its budget reasoning rather than completing code, so
these runs establish endpoint/load/performance but are not a passing code-
quality score. The Qwen3.5-9B download was continuing on Aisha Wi-Fi during the
benchmark; Qwen3-14B remains queued. No existing Qwen2.5 model files were
removed.

After the first model test, a second Aisha watcher was started as PID 541648
from `ps5lm-tests/stage-followon-models-to-ps5.sh`. It waits for the verified
markers for both remaining queue items, independently checks each size and
SHA-256, then uploads each to its unique PS5 app-folder filename and verifies
the FTP byte count. Qwen3.5-9B completed at 06:40:26 IST and passed the
expected 5,680,522,400-byte size and SHA-256
`168350973938a3dad371c8e0822b0d6053908092db6ce2eaafd08b39dab4e50e`. The
queue immediately began the pinned Qwen3-14B Q4_K_M download over Wi-Fi. The
follow-on stager independently rehashed Qwen3.5 and began its wired PS5 FTP
upload at 06:41 IST. It had reached 56.7% (3.41 GB visible on the PS5 FTP
listing) at 06:42:54 IST. Qwen3-14B aria2 download PID 556284 was active at
264 MiB / 8.3 GiB (3%) over Wi-Fi. Aisha had 55 GB free after both active
destination-file allocations; this is enough for the remaining planned 9 GB model, so no USB
backup is needed. PS5 `/health` remained healthy after all four
Qwen3.8 benchmark requests, and the console log recorded `HTTP READY` plus all
four `STREAM DONE` entries.

## 2026-10-09: Qwen3.5-9B loaded and benchmarked

The pinned Unsloth Qwen3.5-9B Q4_K_M file passed its expected 5,680,522,400
bytes and SHA-256 check, then was staged at the exact byte size on the PS5.
Before switching, Qwen3.8's runtime config was preserved as
`inference.before-qwen35-9b.conf`. Qwen3.5's config selected its staged file,
4,096 context and 1,024 max output. The PS5 screen log showed `MODEL LOADED`
and `qwen35 9B Q4_K - Medium`; all 33/33 layers were offloaded to Vulkan.
Model buffer was 4,861.28 MiB and Vulkan-host buffer 545.63 MiB. Load took
about 61 seconds. Health after the benchmark remained model_loaded=true with
context 4,096 and output 1,024.

The attached four-workload streaming benchmark completed all requests:

| Workload | SSE deltas | TTFT | Generation time | Script delta/s | PS5 log tok/s |
|---|---:|---:|---:|---:|---:|
| Creative throughput | 256 | 882 ms | 5.29 s | 48.24 | 47.7 |
| Python LRU code | 300 | 1,047 ms | 12.66 s | 23.61 | 24.1 |
| Bat/ball reasoning | 200 | 1,028 ms | 3.68 s | 54.04 | 54.9 |
| Inverted-cup spatial | 200 | 1,308 ms | 3.24 s | 61.45 | 54.6 |

The script's counter is SSE content-delta chunks rather than tokenizer-confirmed
tokens; the on-device stream log provides a separate token/s figure. The model
emitted visible `<think>` content. In the cup test it incorrectly reasoned
that the ball remained in the cup and went into the microwave, so treat the
throughput suite as completed while noting this reasoning miss.

At 06:47 IST the next queued Qwen3-14B download was active at 0.9 / 8.3 GiB
(10%, about 2.9 MiB/s, ETA around 43 minutes). The follow-on PS5 staging watcher
had logged the complete Qwen3.5 upload and was waiting for the Qwen3-14B
verification marker. Aisha had 55 GB free. Current live PS5 title PID was 848
on Qwen3.5, health was good, and Qwen3.8's model/config remain available for
rollback.

### Follow-up preflight at 06:55 IST

Rechecked the live PS5 via its actual HTTP health response: model remains
loaded, context 4096/output 1024. The active title is PID 848. Aisha's Qwen3-14B
transfer is still active at 2.2/8.3 GiB (26%), approximately 3.0 MiB/s, ETA
about 34 minutes; aria2 sidecar/process remain present. Queue and both follow-on
watchers are alive; Aisha free space remains 55 GB.

The uniquely named Qwen3-14B config has now been staged to
`/data/homebrew/PPSA99005/inference.qwen3-14b.conf` without changing the live
`inference.conf` or process. Its SHA-256 matches the Aisha source exactly:
`fa47bfd739ffc75b64dde976e1305ffeb9ea694b7a0509d06c1a6291fea52639`.
The dedicated backup target `inference.before-qwen3-14b.conf` was absent at
preflight.

Binary identity discrepancy to preserve: the current PS5 FTP `eboot.bin`
stream-hashes to `1be7734718e954aba823d581f76b3ad72735df3bf25fee4a52dcfee47858e4f5`
(FTP listing size 73,992,956 bytes), while Aisha's prepared Qwen3.8-capable
eboot is 73,486,124 bytes with SHA-256
`db40ae2b9355d635c3bd2749dbe4766758cbf132659e29b409168d9e83775d89`. The
PS5 backup `eboot.before-qwen14b.bin` is also a different hash
(`7444379d3b6493f8dfd02886613489e13f1af55d508681115ae6a2d2132e01b4`). This
contradicts the earlier note that the prepared binary hash exactly matched the
PS5 binary; leave the current executable untouched. The active PS5 runtime has
already successfully loaded Qwen3.8 and Qwen3.5, so first try Qwen3-14B using
that known-running executable and the staged config. Rebuild/redeploy only if
runtime evidence shows the Qwen3-14B architecture is unsupported or another
binary change is required; keep a verified rollback copy before any such
change.

## Overnight queue update (2026-10-09 06:49 IST)

Continuation objective: test at least three distinct models with the attached
four-workload suite at
`/Users/user/.codex/attachments/7f9832db-fb3e-48da-a1c8-301d343a308d/pasted-text-1.txt`,
then continue with additional fit or tight-fit compatible models if the user
has not interrupted. Keep the native title's on-screen status reporting
enabled during every load and test. Build only on Aisha; keep downloads on its
Wi-Fi interfaces and deploy over the wired PS5 path. Preserve known-good
models/configs for rollback. Consider Aisha's 250 GB USB `/dev/sdb` only if
available internal space actually becomes inadequate.

Fresh verified snapshot at 2026-10-09 07:34 IST:
- PS5 `http://ps5.tailnet.example:8081/health` returned `status=ok`,
  `model_loaded=true`, context 4096, output cap 1024. Qwen3.5-9B remains the
  active known-good model and must not be disturbed while the next download
  runs.
- Aisha queue controller PID 502620 and staging watcher PID 541648 were alive.
  The Qwen3-14B aria2 transfer PID 556284 completed at 07:33 IST and aria2
  reported OK. The queue
  independently verified exact size 9,001,753,184 and SHA-256
  `0d14496bc5fb2146dd3f2770332d8f48309b80b0c0007c4fd5d4fb1fff5c01fe` at
  07:34 IST. The staging watcher rechecked the file and began wired Tailscale
  FTP upload to the PS5 at 07:34:42 IST. At 07:34:53, PS5 FTP listed
  314,648,452 bytes so far; curl uploader PID 593641 was active. Wait for the
  exact 9,001,753,184-byte remote listing and `STAGED` marker before swapping
  the live config.
- The stager previously logged Qwen3.5's complete 5,680,522,400-byte upload
  as `STAGED`, without changing the live title/config.
- Aisha had 55 GB free on `/home/user`; USB backup was not needed.
- Aisha also has Qwen2.5-14B-Instruct Q4_K_M shards, Qwen2.5-7B shards, and
  Qwen2.5-Coder-3B GGUFs. The shards have not been freshly integrity-checked
  during this continuation; verify hashes and model metadata before using one
  as a later candidate.
- The prepared Qwen3-14B config is
  `/home/user/ps5lm-inference/vk/out/qwen38-20261009/configs/inference-qwen3-14b.conf`
  (4,096 context, 1,024 output, `Qwen3-14B-Q4_K_M.gguf`). Reusable eboot is
  present on Aisha (SHA-256
  `db40ae2b9355d635c3bd2749dbe4766758cbf132659e29b409168d9e83775d89`); the
  config file was initially referred to by the wrong filename in the handoff,
  now corrected here.
- Follow-on Qwen3-8B Q4_K_M is prepared to start automatically after Qwen3-14B
  passes verification and is staged on the PS5. Hugging Face metadata at pinned
  revision `a6adef130ffb23ddaf1a62fec9dced968c9bc482` gives exact size
  5,027,784,512 bytes and SHA-256
  `120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4`.
  The follow-on watcher is running on Aisha as PID 564364; it downloads via
  Wi-Fi, verifies size/hash, uploads over wired Tailscale, verifies the PS5
  byte count, and does not alter the live config/title. Its ready-to-use config
  is at `/home/user/ps5lm-inference/vk/out/qwen38-20261009/inference-qwen3-8b.conf`
  and is now staged on the PS5 as `inference.qwen3-8b.conf` with matching
  SHA-256 `1b3a4b415b518911785c720a1405c845472a9405b01f52e1d19d38959e62bd69`.
  Aisha copy of the watcher is
  `/home/user/ps5lm-inference/vk/out/qwen38-20261009/queue-qwen3-8b-after-qwen3-14b.sh`;
  log: `/home/user/models/llmfit/qwen3-8b-followon.log`.
- The built Aisha `libllama.a` contains `llama_model_gemma3` loader/graph
  symbols. Added a further architecture-diverse fit candidate: Unsloth Gemma3
  12B IT Q4_K_M, pinned at revision
  `d15e4c7dc21dc55d56bf8549db57a71ad8a2a35d`, exact size 7,300,778,336 bytes,
  SHA-256 `15b8fd9d8672cd4240c178c217ca781409291f34e353d2e913b29c7602ceb3ff`.
  It will begin downloading over Wi-Fi only after Qwen3-8B is verified and
  staged. Aisha watcher PID 573444 is alive and waiting; log
  `/home/user/models/llmfit/gemma3-12b-followon.log`. Its runtime config
  (`context_tokens=4096`, `max_output_tokens=1024`) is staged on the PS5 under
  the unique name `inference.gemma3-12b.conf`; its SHA-256 matches Aisha's
  source `7f7bab08ddddd1caa169e84a2ea48c571ed8e03c2bc1fed83b8cb5559371369e`.
  The active config/title remain untouched.

Do not count Qwen3-14B until the queue verifies size and SHA-256, staging
verifies the PS5 byte count, a reversible config swap leads to a successful
load/health check, the screen visibly reports load/request status, and all
four attached workloads complete. Then continue to at least one more distinct
compatible candidate, checking Aisha space and avoiding redundant downloads.

## 2026-10-09: Qwen3-14B Q4_K_M loaded and benchmarked

Qwen3-14B completed on Aisha at 07:33 IST. aria2 reported `OK`; the queue
controller verified exact size 9,001,753,184 bytes and SHA-256
`0d14496bc5fb2146dd3f2770332d8f48309b80b0c0007c4fd5d4fb1fff5c01fe`.
The staging watcher independently rehashed it, uploaded over wired Tailscale,
and verified the same exact size in the PS5 FTP listing. The unique runtime
config was moved into place only after the previous Qwen3.5 config had been
saved as `inference.before-qwen3-14b.conf`; FTP readback confirmed the active
config hash matched the prepared source.

The currently deployed executable was intentionally retained; no build was
needed because it successfully loaded this model. The PS5 status panel visibly
reported `LOADING MODEL` at 10% intervals through 100%, then `MODEL LOADED`,
`qwen3 14B Q4_K - Medium`, and `HTTP READY 0.0.0.0:8081 LAN ONLY`. Load took
97 seconds. All 41/41 layers were offloaded to Vulkan0; the Vulkan model buffer
was 8,161.75 MiB and Vulkan-host buffer 417.30 MiB. Health after load and after
the benchmark remained `status=ok`, `model_loaded=true`, context 4096, output
limit 1024.

The user's attached streaming suite completed all four requests:

| Workload | SSE deltas | TTFT | Generation time | Script delta/s | PS5 log tok/s |
|---|---:|---:|---:|---:|---:|
| Creative throughput | 256 | 1,779 ms | 7.01 s | 36.38 | 35.8 |
| Python LRU code | 300 | 726 ms | 11.85 s | 25.22 | 25.2 |
| Bat/ball reasoning | 200 | 835 ms | 5.08 s | 39.17 | 38.4 |
| Inverted-cup spatial | 200 | 986 ms | 5.18 s | 38.42 | 38.3 |

The benchmark script counts nonempty SSE content-delta events rather than
tokenizer-confirmed tokens; use the server's `STREAM DONE` token-rate log as
the authoritative on-device rate. The model emits visible `<think>` content.
The capped code and reasoning outputs ended mid-response, so this run
establishes throughput and serving stability but does not prove complete code
or reasoning quality.

While the benchmark ran, the next pinned Qwen3-8B Q4_K_M transfer was active
on Aisha Wi-Fi: at 07:52 IST it was 1.8 GiB / 4.6 GiB (39%), 2.7 MiB/s, ETA
about 17.5 minutes. aria2 PID 597486; follow-on controller PID 564364 will
verify its exact size/hash and upload to the PS5 without changing the live
model/config. Gemma3-12B follow-on watcher PID 573444 is waiting for Qwen3-8B
to be staged. Aisha had 50 GB free; no USB backup required.

## 2026-10-09: Qwen3-8B Q4_K_M loaded and benchmarked

The existing Qwen3-8B follow-on watcher verified the pinned Unsloth file at
exactly 5,027,784,512 bytes with SHA-256
`120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4`, then
uploaded it over wired Tailscale FTP and confirmed the same remote byte count.
The active Qwen3-14B config was preserved as
`inference.before-qwen3-8b.conf`; the prepared 4K-context/1,024-output Qwen3-8B
config was byte-compared before restart. Only the previously verified
PPSA99005 inference process (PID 879) was stopped and relaunched; the existing
eboot was reused.

The PS5 screen and captured klogsrv output showed load progress at 10% steps,
`MODEL LOADED`, and `HTTP READY 0.0.0.0:8081 LAN ONLY`. Load took 54 seconds;
all 37/37 layers were offloaded to Vulkan. The Vulkan model buffer was
4,455.34 MiB and Vulkan-host buffer 333.84 MiB. The startup POST self-test
passed, and `/health` returned `model_loaded=true`, 4,096 context tokens, and
1,024 max output tokens after benchmarking.

All four requests in the user's attached SSE benchmark completed:

| Workload | SSE deltas | TTFT | Generation time | Script delta/s | PS5 log tok/s |
|---|---:|---:|---:|---:|---:|
| Creative throughput | 256 | 1,711 ms | 3.93 s | 64.92 | 57.7 |
| Python LRU code | 300 | 703 ms | 8.87 s | 33.70 | 34.2 |
| Bat/ball reasoning | 200 | 574 ms | 2.99 s | 66.57 | 64.5 |
| Inverted-cup spatial | 200 | 1,075 ms | 2.72 s | 73.09 | 63.8 |

As before, the script's count/rate is nonempty SSE content chunks rather than
tokenizer-confirmed tokens; the PS5 `STREAM DONE` rates are the server-side
measurements. Several outputs visibly contain `<think>` reasoning and end at
the attached test's token cap, so these timings establish serving/throughput,
not complete-answer quality.

While Qwen3-8B is active on the PS5, the queued Gemma3-12B IT Q4_K_M download
has started over Aisha Wi-Fi. At 08:47 IST aria2 reported 5.2 GiB / 6.7 GiB
(76%), about 2.4 MiB/s, ETA around 11 minutes. The watcher is live; Aisha had
44 GB free, so no USB backup is currently needed. After exact source integrity
and PS5 staging checks, test Gemma3 with the same screen-visible,
rollback-safe load and four-workload benchmark sequence.

### Next queued tight-fit candidate: Qwen3-30B-A3B UD-IQ2_XXS

Prepared a sequential watcher that waits for Gemma3's verified PS5 `STAGED`
marker before beginning another Wi-Fi download. Candidate is Unsloth's
Qwen3-30B-A3B GGUF, a 30.5B total / 3.3B active-parameter MoE
([model card](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF)). Selected
`Qwen3-30B-A3B-UD-IQ2_XXS.gguf` at pinned revision
`a29021d14a095a1a2ee279e129115a2e640666fc`; Hugging Face reports exact size
10,362,261,280 bytes and SHA-256
`0dfba4062301a231309a6b73eed572c2ed5b44c5fe63cc897d3b2a024e1ec58d` ([file
metadata](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/blob/a29021d14a095a1a2ee279e129115a2e640666fc/Qwen3-30B-A3B-UD-IQ2_XXS.gguf)).
The 10.36-GB file is a tight-fit experiment for the PS5's 12-GiB Vulkan heap;
successful load is not assumed. Aisha's already-linked `libllama.a` contains
`llama_model_qwen3moe` loader and graph symbols. Prepared a 4K-context config;
the watcher verifies its size/SHA, downloads the GGUF with aria2 resume over
Wi-Fi, verifies exact source bytes/SHA, then uploads both model and uniquely
named config over wired Tailscale and verifies PS5 byte counts. It will not
activate the model/config. At setup Aisha had 44 GB free, enough for the active
Gemma download plus this candidate; no USB backup needed. Watcher:
`ps5lm-tests/queue-qwen3-30b-a3b-after-gemma3.sh`; copied to Aisha and running
as PID 627111, waiting for Gemma3's exact `STAGED` marker (confirmed in its
live log at 08:21 IST).

### Following candidate: Gemma3-27B IT UD-IQ2_XXS

Prepared another sequential watcher for Unsloth Gemma3-27B IT
UD-IQ2_XXS, pinned at revision
`b8770e3aed02d2c55b3fe150a579a0b899c22040`. Hugging Face API metadata gives
exact size 7,850,253,408 bytes and SHA-256
`28b51117882be011c1c07011190aed4df814b81cad870a245a8993846034dc7a`
([file page](https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/blob/b8770e3aed02d2c55b3fe150a579a0b899c22040/gemma-3-27b-it-UD-IQ2_XXS.gguf)).
The linked Aisha library contains `llama_model_gemma3::load_arch_hparams`.
The watcher waits for Qwen3-30B's final READY marker, then downloads this
candidate over Wi-Fi, verifies exact source size/SHA, uploads model plus a
unique 4K-context config over wired Tailscale and checks remote sizes. It will
not swap the live config/title. Candidate is not yet downloading; it is
sequentially queued to avoid competing with Gemma3-12B and Qwen3-30B. Config
size is 156 bytes, SHA-256
`f1b893f817e852bdd38f2c27ab365f433a0a98ad2f2d345ece8c2c68e05d1899`.
Watcher: `ps5lm-tests/queue-gemma3-27b-after-qwen30.sh`; installed and live on
Aisha as PID 632764, waiting for Qwen3-30B's final READY marker.

### 2026-10-09 09:00 IST: overnight continuation checkpoint

Fresh live check on Aisha: Gemma3-12B IT Q4_K_M aria2 completed successfully
at 08:57:48 IST. The watcher independently verified exact source size
7,300,778,336 bytes and SHA-256
`15b8fd9d8672cd4240c178c217ca781409291f34e353d2e913b29c7602ceb3ff`.
At 08:59:20 its wired Tailscale FTP upload to the PS5 was still active (curl
PID 653142; progress output had reached 44%). The two subsequent watchers
remain alive and sequentially waiting: Qwen3-30B-A3B PID 627111, then Gemma3-27B
PID 632764. Aisha still has 44 GB free, so no USB use is warranted. PS5 `/health`
at 08:59 returned `status=ok`, `model_loaded=true`, 4096 context, and 1024 max
output tokens; keep the current Qwen3-8B model active until Gemma3 is fully
staged and a deliberate rollback-safe load begins.

Correction/update: Gemma3-12B reached `STAGED` at 09:01:18 IST with the exact
PS5 byte count, then was activated using the existing executable. The former
live config was copied to `inference.before-gemma3-12b.conf` and its SHA-256
(`1b3a4b415b518911785c720a1405c845472a9405b01f52e1d19d38959e62bd69`) was
verified unchanged. Gemma's active config hash matched the prepared file
(`7f7bab08ddddd1caa169e84a2ea48c571ed8e03c2bc1fed83b8cb5559371369e`). The
existing PPSA99005 process alone was stopped and relaunched; no build was
needed. The on-screen status/klogsrv capture reported load milestones 10%-80%,
100% at 88 seconds, `MODEL LOADED`, and `HTTP READY 0.0.0.0:8081 LAN ONLY`.
All 49/49 layers were offloaded to Vulkan0; model buffer 6,956.32 MiB and
Vulkan-host buffer 787.70 MiB. `/health` confirms loaded at 4096 context / 1024
output.

The user's attached four-workload SSE script completed against Gemma3:

| Workload | SSE deltas | TTFT | Generation time | Script delta/s | PS5 log tok/s |
|---|---:|---:|---:|---:|---:|
| Creative fiction | 256 | 1,188 ms | 7.22 s | 35.31 | 35.0 |
| Python LRU code | 300 | 806 ms | 16.17 s | 18.50 | 20.3 |
| Bat/ball reasoning | 200 | 1,506 ms | 4.91 s | 40.54 | 39.7 |
| Inverted-cup spatial | 120 | 1,465 ms | 2.66 s | 44.68 | 39.7 |

The cup answer was correct (ball remains on the dining table). The script counts
SSE content-delta events, not tokenizer-confirmed tokens; use PS5 `STREAM DONE`
rates as the server-side rates. The capped code answer ended mid-response.
Post-benchmark health stayed good. The sequential Qwen3-30B-A3B download began
on Aisha Wi-Fi at 09:01:36; at 09:05:39 it was 581 MiB / 9.6 GiB (5%), about
2.2 MiB/s, ETA about 1h8m. Download PID 656367 and watcher PID 627111 were
alive; Gemma3-27B watcher PID 632764 remains queued after it. Aisha has 34 GB
free, sufficient for the queued downloads; USB backup remains unnecessary.

### Additional sequential candidate queued: Qwen3-32B UD-IQ2_XXS

Queued a third tight-fit model behind Gemma3-27B, using the existing Qwen3
architecture support already verified by the tested Qwen3 models. Candidate is
Unsloth `Qwen3-32B-UD-IQ2_XXS.gguf` from
`unsloth/Qwen3-32B-GGUF`, pinned to revision
`2c44a91d5332656861d5dc05fcb5c9afc16b663e`. Hugging Face's tree API reports
exact size 9,380,412,800 bytes and LFS SHA-256
`1c163617a0bebeac9bca64874ff2089bdfce14457240980294aa87e11cd428bd`
([repo](https://huggingface.co/unsloth/Qwen3-32B-GGUF/tree/2c44a91d5332656861d5dc05fcb5c9afc16b663e),
[file](https://huggingface.co/unsloth/Qwen3-32B-GGUF/blob/2c44a91d5332656861d5dc05c9afc16b663e/Qwen3-32B-UD-IQ2_XXS.gguf)).
This is a 32B dense model and an intentionally tight Vulkan-memory experiment;
load success is not presumed. Prepared a 4K/1024 runtime config (140 bytes,
SHA-256 `75d383c7bef038d7af3e63245b20f20fb04d5d5b8629f52da435fdbfb4c223fb`)
and a resumable aria2 + SHA-verifying + wired-FTP staging watcher. It starts
only after Gemma3-27B's exact READY marker, checks for at least the full model
size plus a 2-GiB free-space reserve, and never changes the live PS5 config.
Watcher PID 661832 on Aisha was verified alive at 09:09 IST waiting for
Gemma3-27B. At that check Qwen3-30B was 0.9 GiB / 9.6 GiB (9%) at about
2.1 MiB/s, ETA ~1h11m; aria2 PID 656367, Gemma3-27B watcher PID 632764.
Available Aisha disk remained 34 GB; no USB transfer is currently necessary.

### Comfortable-fit Qwen3-4B Q4_K_M queued behind Qwen3-32B

Checked that this exact GGUF was not already in Aisha's model tree or the PS5
app folder. Pinned Unsloth `Qwen3-4B-Q4_K_M.gguf` from
`unsloth/Qwen3-4B-GGUF` at revision
`ab70818b1752ce8f27423fef4c2daf81f484063f`. Hugging Face's repository tree
API reports 2,497,280,800 bytes and LFS SHA-256
`3c129e72a7576bfff7c5f6b59be49990396cffd527baaee825b12f942705a297`
([repo](https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/ab70818b1752ce8f27423fef4c2daf81f484063f),
[file](https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/ab70818b1752ce8f27423fef4c2daf81f484063f/Qwen3-4B-Q4_K_M.gguf)).
Prepared a 4K/1024 config (130 bytes, SHA-256
`a7e6dd5b5e875287eadd452c1abc582fd321a63a07f47696354e7303c7ac2589`) and
started Aisha watcher PID 662974. It waits for Qwen3-32B's verified READY
marker, requires model size plus 2 GiB free, resumes with aria2 over Wi-Fi,
checks full SHA, and stages via wired FTP without activating the config. Aisha
still had 34 GB free at queue time, so no removable-media operation is planned.

Live follow-up at 09:14 IST: Qwen3-30B aria2 PID 656367 remains active and its
reported transfer moved from 1.3 GiB (13%) at 09:12 to 1.5 GiB (16%) at 09:13,
about 2.1 MiB/s with ETA near 1h4m. The three later watchers remain alive in
sequence (Gemma3-27B PID 632764, Qwen3-32B PID 661832, Qwen3-4B PID 662974).
Aisha remains at 34 GB free; PS5 `/health` remains loaded and healthy on
Gemma3-12B (4096/1024). No transfer recovery or media backup is needed.

Live queue check at 09:20 IST: Qwen3-30B aria2 PID 656367 advanced from 2.2
GiB (23%) at 09:18:47 to 2.4 GiB (25%) at 09:19:48, ~2.5 MiB/s, ETA ~48m.
All three sequential watchers remained alive; Aisha had 34 GB free and PS5
health remained true on Gemma3-12B (4096/1024). No intervention or USB backup
needed.

Live check 09:23 IST: Qwen3-30B aria2 PID 656367 reached 2.8 GiB / 9.6 GiB
(29%), with current estimate ~59m. The download and three sequential watchers
are still live; Aisha has 34 GB free and PS5 health is unchanged. Transfer
remains active and resumable.

Live check 09:24 IST: Qwen3-30B download remains aria2 PID 656367, now 2.9
GiB / 9.6 GiB (30%) at ~2.2 MiB/s, ETA ~50m. The three successor watchers
remain alive, Aisha still has 34 GB free, and the PS5 remains healthy on
Gemma3-12B.

Live check 09:25 IST: Qwen3-30B aria2 PID 656367 reached 3.2 GiB / 9.6 GiB
(33%), reporting roughly 2.0 MiB/s and ETA 52m. The three staged-model queue
watchers remain alive; Aisha retains 34 GB free; PS5 health is true on
Gemma3-12B.

### 2026-10-09 09:33 IST: automatic staged-model evaluation handoff

At 09:32, a live Aisha check found Qwen3-30B aria2 PID 656367 still active at
4.2 GiB / 9.6 GiB (43%), about 2.7 MiB/s, ETA ~33m. Its output file is
preallocated and still has an `.aria2` sidecar, so only the aria2 log is being
used for progress. Gemma3-27B, Qwen3-32B and Qwen3-4B watchers remain alive,
waiting in sequence. Aisha has 34 GB free. PS5 health and `/v1/models` confirm
Gemma3-12B is still loaded at 4K context / 1024 output; `/data` has 424.6 GB
available. No USB backup is needed.

Found that the existing candidate watchers only download, verify, and stage;
their `READY` marker explicitly expects a manual load/benchmark. Added
`ps5lm-tests/run_followon_evaluation.py` plus
`ps5lm-tests/benchmark_streaming_suite.py`, copied both to Aisha under
`~/ps5lm-inference/vk/out/qwen38-20261009/`, and started evaluator PID 679604.
It is confirmed live and waiting for the Qwen3-30B stage marker. For each
candidate it rechecks local size/SHA, absence of an aria2 sidecar, PS5 staged
size, and config size/SHA; preserves and round-trip-verifies the current
`inference.conf` under a unique rollback name; switches only PPSA99005; waits
for the expected model/health; and runs the four streaming workloads with
long per-request timeouts. Aisha logs PS5 klog/status output. Failed loads
restore and verify the preceding config/model before advancing. It does not
delete model assets. The evaluator has not yet changed the active PS5 model;
the current Gemma3-12B service remains healthy.

The streaming report labels received SSE content events as `content_chunks`,
not tokenizer-confirmed tokens. The PS5's own `STREAM DONE` screen/klog rates
remain the authoritative tok/s measurement. The evaluator script was parsed
successfully by Python on both the Mac and Aisha; an end-to-end candidate run
is still pending Qwen3-30B completion and staging.

### 2026-10-09 09:37 IST: Gemma 3 12B Q4_K_M PS5 benchmark run

Measured live on PS5 endpoint `http://ps5.tailnet.example:8081/v1/chat/completions` serving `gemma3 12B Q4_K - Medium`:

| Benchmark Workload | Output Chunks | TTFT | Decode Speed (tok/s) | Native Engine Speed | Qualitative Findings |
|---|---|---|---|---|---|
| **Raw Decode Throughput** (Cyberpunk Scene) | 256 | 1429.1 ms | **38.25 tok/s** | ~40.0 tok/s | Fluid, high-coherence prose; vivid imagery |
| **Algorithmic Code** (Thread-safe LRU Cache) | 300 | 708.4 ms | **20.06 tok/s** | ~22.5 tok/s | Rich typing (`TypeVar`, `Callable`, docstrings) |
| **Multi-step Reasoning** (Bat & Ball Math) | 200 | 916.1 ms | **33.52 tok/s** | ~35.0 tok/s | Correct step-by-step algebra ($x = 0.05$) |
| **Spatial Physics** (Inverted Cup Ball Tracking) | 120 | 1100.5 ms | **41.08 tok/s** | **40.00 tok/s** | **100% correct physics** — deduced ball stays on table |

**Key architectural observations:**
- Peak decode speed reached **41.08 tok/s** (native engine reported 40.0 tok/s).
- Solved the inverted container spatial physics trap cleanly without containment bias.
- Memory usage remained comfortably within the 12,288 MiB heap.

### 2026-10-09: authenticated model-management API and GPU stats

Extended `vk/http_server.cpp` with authenticated `GET /v1/stats` alongside the
dynamic model inventory/load endpoints. The service is bound to `0.0.0.0:8081`,
but accepts non-loopback clients only from Tailscale's `100.64.0.0/10`; every
route requires the configured Bearer key. The key is hardcoded in the title for
now and is intentionally not recorded in this findings file. The PS5 display
continues to show model-load and server status.

The stats response includes loaded model/context details, Vulkan GPU heap size,
available budget headroom, used estimate, per-device values, and raw Vulkan
budget counters. An initial live query exposed unsigned underflow in
llama.cpp's `heapBudget - heapUsage`; the Aisha Vulkan backend now clamps each
heap independently before summing. The PS5's raw Vulkan counters remain
inconsistent: the driver reports 19,756,023,808 bytes usage against a
12,884,901,888-byte physical heap and an 8,589,934,592-byte budget. Accordingly,
the API reports conservative available headroom (currently 0), derives a
headroom-based used estimate, and flags raw usage as invalid rather than
pretending that the reported usage is physically possible. Treat these as
conservative backend-heap indicators, not a complete accounting of PS5 UMA or
system-reserved memory. A future driver-side measurement is needed for precise
used VRAM.

Final title build on Aisha:
`~/ps5lm-inference/vk/out/dynamic-api-stats-headroom-20261009/eboot.bin`,
SHA-256 `a2d0dd846547fe050b3c1d0474ecc38c064f0776260a677b8b1d673e2f890e3a`.
Deployed to the existing PPSA99005 slot; the immediately prior executable is
preserved on-console as
`/data/homebrew/PPSA99005/eboot.before-vram-headroom-20261009.bin` (earlier
known-good rollback builds are also retained). Gemma3-12B remained configured
at context 4096/output 1024, loaded successfully, and passed the startup POST
smoke test. The klog reported `HTTP READY 0.0.0.0:8081 TAILSCALE + API KEY`.
Authenticated `/health` and `/v1/stats` returned HTTP 200; an unauthenticated
stats request returned HTTP 401. The live stats JSON returned 12 GiB physical
heap, 0 bytes conservative available budget, and explicitly marked the raw
usage inconsistency.

Also exercised `POST /v1/models/load` on the same Gemma3-12B model with
`context_tokens=2048`, `max_tokens=512`, `threads=3`, and `gpu_layers=999`;
the request returned `status=loaded` and `/health` reported those effective
values. Then restored and re-verified the original context 4096, output 1024,
threads 4 / batch threads 4. No model assets or inference config files were
modified. The PS5 screen showed load progress for both reloads and the service
returned to healthy state.

### 2026-10-09 10:37 IST: Qwen3-30B-A3B IQ2_XXS loaded and benchmarked

The staged Qwen3-30B-A3B UD-IQ2_XXS passed a fresh Aisha SHA-256 verification
and PS5 staged-size/config checks. It loaded dynamically in about 2 minutes;
all 49/49 layers were offloaded to Vulkan0. The PS5 reported a 9,709.61 MiB
Vulkan model buffer, 166.92 MiB Vulkan-host buffer, and 75.19 MiB compute
buffer. `/health` confirmed Qwen3-30B-A3B at 4096 context / 1024 max output.

Ran all four attached streaming workloads. The SSE `content_chunks` and client
timings are not tokenizer token counts; the PS5 `STREAM DONE` rates are the
authoritative decode rates:

| Workload | SSE content chunks | TTFT (ms) | PS5 decode (tok/s) | Finish | Output assessment |
|---|---:|---:|---:|---|---|
| Creative fiction | 256 | 1,850 | 73.1 | length cap | Still in `<think>` planning; no finished scene |
| Python LRU cache | 300 | 383 | 40.2 | length cap | Still reasoning; implementation cut off before completion |
| Bat/ball + strawberry count | 200 | 450 | 95.5 | length cap | Still deriving the setup; no final answers |
| Inverted-cup spatial reasoning | 200 | 468 | 96.2 | length cap | Still considering the premise; no final location answer |

All requests streamed successfully, but these capped runs do not establish
answer quality because the model spent the output allowance in visible
`<think>` text before reaching final responses. Reports are saved on Aisha as
`/home/user/models/llmfit/benchmark-reports/qwen3-30b-a3b.json` and `.log`.
The evaluator matcher was corrected to recognize llama.cpp's `qwen3moe` model
name and to reuse a candidate already loaded; evaluator PID 726020 is now
waiting for the next exact staging marker.

At 10:37 IST, Gemma3-27B IT UD-IQ2_XXS was downloading sequentially over Wi-Fi
at 3.3 GiB / 7.3 GiB (45%), about 2.6 MiB/s, ETA about 25.5 minutes. Qwen3-32B
and Qwen3-4B watchers remain queued behind it. PS5 stays healthy on Qwen3-30B;
no downloads were stopped or altered.

### 2026-10-09 10:43 IST: Qwen 3.8 27B UD-IQ2_S dynamically loaded and re-benchmarked

Dynamically switched to `Qwen3.8-27B-UD-IQ2_S.gguf` via `POST /v1/models/load` with Bearer auth, 4,096 context, 1,024 max output, and all 65/65 layers on Vulkan. Load completed in ~89 seconds; `/v1/stats` confirmed model size 8,360,974,336 bytes (~8.36 GB) and 26,895,998,464 parameters.

Executed the 4 standardized streaming workloads:

| Workload | Tokens | TTFT (ms) | Gen Time (s) | Decode Speed (tok/s) | Output Notes |
|---|---:|---:|---:|---:|---|
| **Creative Fiction** (Cyberpunk Scene) | 256 | 1197.5 ms | 12.82 s | **19.89 tok/s** | Completed evocative cyberpunk prose; skips `<think>` cleanly |
| **Algorithmic Code** (LRU Cache) | 300 | 1124.5 ms | 20.41 s | **14.65 tok/s** | Structured `<think>` design phase; cut off mid-code by 300-token cap |
| **Logical Reasoning** (Bat/Ball Math) | 200 | 1808.0 ms | 9.40 s | **21.16 tok/s** | Formulated $2x + 1.00 = 1.10 \implies x = 0.05$ inside `<think>` |
| **Spatial Physics** (Inverted Cup) | 200 | 1395.4 ms | 9.33 s | **21.34 tok/s** | Active 3D physics scrutiny inside `<think>` |

**Key observations:**
- Sustained generation speed: **~20 – 21.3 tok/s** across prose and reasoning tasks.
- Fits cleanly inside the 12GB Vulkan heap (~8.36 GB model buffer).
- Unlike pure dense non-reasoning models, Qwen 3.8 natively triggers an internal `<think>` trace before generation.

### 2026-10-09 10:45 IST: follow-on queue checkpoint (last verified)

At 10:45 IST, Aisha had approximately 25 GB free on the model filesystem. The
sequential Gemma3-27B IT UD-IQ2_XXS transfer was at 4.3 GiB / 7.3 GiB (59%),
about 2.5 MiB/s, with an estimated 19.5 minutes remaining. The aria2 transfer
(PID 710927) and its queue watcher (PID 632764) were alive; Qwen3-32B and
Qwen3-4B watchers were queued behind Gemma3-27B, and evaluator PID 726020 was
waiting for the Gemma staging-ready marker. PS5 `/health` verified Qwen
3.8-27B loaded and healthy. No transfers or model assets were modified.

This is a historical checkpoint, not a live status: a subsequent SSH attempt
from the current session was denied by the network sandbox, so newer Aisha and
PS5 state could not be verified here.

### 2026-10-09 11:00 IST: follow-on queue live checkpoint

Read-only checks on Aisha at 11:00 IST confirmed Gemma3-27B IT UD-IQ2_XXS was
at 6.7 GiB / 7.3 GiB (91%), downloading at about 2.6 MiB/s with an ETA near
3m54s. Its aria2 PID 710927 and watcher PID 632764 were alive. Aisha had 25 GB
free. The Qwen3-32B watcher (PID 661832) and Qwen3-4B watcher (PID 662974)
remained queued behind their respective READY markers; evaluator PID 726020
was still waiting for Gemma's verified staging marker. The preallocated file
size was not treated as completed transfer evidence. A first PS5 health request
timed out at 5 seconds; a retry succeeded and confirmed Qwen3.8-27B loaded and
healthy, context 8192 / max output 2048. The PS5 klog also showed a recent
health-check success and stream completion. No downloads or model files were
modified.

### 2026-10-09 10:55 IST: dynamic context window & KV-cache optimization sweep

Executed an automated configuration optimization sweep (`tune_and_benchmark.py`) for `Qwen3.8-27B-UD-IQ2_S.gguf`:

| Configuration | Context | Max Output | KV Cache Type | Flash Attention | TTFT (ms) | Prose (tok/s) | Logic (tok/s) | Notes |
|---|---:|---:|:---:|:---:|---:|---:|---:|---|
| **Config 2: FlashAttn + Q8_0 KV** | 4,096 | 1,024 | `q8_0` | Enabled (`1`) | 1,385.0 ms | **20.44 tok/s** | **21.37 tok/s** | 50% KV cache memory saving vs F16 |
| **Config 3: Fast Interactive** | 2,048 | 512 | `q8_0` | Enabled (`1`) | 1,033.5 ms | **19.81 tok/s** | **20.91 tok/s** | Reduced prefill latency |
| **Config 4: Max Frontier (8K Ctx)** | **8,192** | 1,024 | **`q4_0`** | Enabled (`1`) | **891.6 ms** | **19.87 tok/s** | **21.01 tok/s** | **8K context fits & runs stably on 27B model** |

**Crucial engineering breakthrough:**
- Quantizing the KV cache to `q4_0` allowed **8,192 tokens of context** to run completely inside the PS5's 12 GiB Vulkan heap on a 27B model without OOM.
- Flash Attention (`1`) delivered the lowest Time-To-First-Token observed on this model (**891.6 ms**).

### 2026-10-09 11:00 IST: full 4-test benchmark with expanded 2,048 max output

Reconfigured Qwen 3.8 27B UD-IQ2_S to `context_tokens=8192`, `max_output_tokens=2048`, `cache_type=q4_0`, and `flash_attention=1`. Executed the full four-workload benchmark with expanded output token allowances:

| Test Suite | Tokens | TTFT | Gen Time | Decode Speed (tok/s) | Output Behavior & Quality |
|---|---:|---:|---:|---:|---|
| **Raw Decode Throughput** (Cyberpunk Scene) | 318 | 1,122.3 ms | 16.47 s | **19.25 tok/s** | Fully completed 3-paragraph cyberpunk scene |
| **Algorithmic Code** (Thread-safe LRU Cache) | 650 | 1,397.7 ms | 44.00 s | **14.75 tok/s** | Deep `<think>` architectural design + generic typing considerations |
| **Multi-step Reasoning** (Bat & Ball Math) | 400 | 1,185.9 ms | 19.26 s | **20.72 tok/s** | Solved $x = \$0.05$ and correctly counted 3 'r's in "strawberry" |
| **Spatial Physics** (Inverted Cup Ball Tracking) | 389 | 1,928.9 ms | 18.11 s | **21.43 tok/s** | Completed full answer (exhibited containment permanence bias) |

**Summary findings:**
- The PS5 sustained 44.0 seconds of continuous generation (650 tokens) at ~15–21 tok/s with zero thermal throttling or crashes.
- 8K context / 2K max output with `q4_0` KV cache is completely stable on 27B models on PS5.

### 2026-10-09 11:25 IST: Gemma3-27B staged; sequential Qwen32 download resumed

Gemma3-27B IT UD-IQ2_XXS completed on Aisha at 11:05:23 IST and passed the
expected 7,850,253,408-byte / SHA-256
`28b51117882be011c1c07011190aed4df814b81cad870a245a8993846034dc7a` check.
Its wired Tailscale upload completed at 11:09:39 and the PS5 watcher emitted
`STAGED`, the matching byte count, a verified 156-byte candidate config, and
the exact READY marker at 11:09:40. No title config was changed by staging.

The original Qwen3-32B watcher stopped because it looked for its prepared config
in the wrong directory; the file was actually present under `configs/` and its
140-byte SHA-256 matched the expected value. Corrected the watcher path and
made Qwen3-4B's failure detection ignore old historical STOP lines. Preserved
both prior scripts with unique `.before-path-fix-20261009.sh` copies on Aisha.
Restarted the serial chain: Qwen3-32B watcher PID 759103, Qwen3-4B watcher PID
759104 (waiting for Qwen32 READY), and Qwen3-32B aria2 PID 759126. At 11:21:12
the real aria2 log showed 360 MiB / 8.7 GiB (4%), about 3.0 MiB/s, ETA about
47.5 minutes; Aisha had 16 GB free after Qwen32 allocation. Do not infer
download progress from the preallocated file size.

The earlier evaluator PID 726020 exited at 11:09:42 IST on a PS5 `/health`
request timeout before it began Gemma3-27B loading. The title process remained
running; the PS5 klog subsequently showed an in-progress model load reaching
100% and `MODEL LOADED`. Authenticated `/health` recovered and verified
Gemma3-12B loaded at context 30,720 / max output 2,048. The persistent title
`inference.conf` remains the known Gemma3-12B 4,096/1,024 config with SHA-256
`7f7bab08ddddd1caa169e84a2ea48c571ed8e03c2bc1fed83b8cb5559371369e`; created
and read-back-verified a new console/Aisha rollback copy named
`inference.before-gemma3-27b-20261009.conf` with the same digest.

Added transient PS5 API retry handling (up to 12 attempts) to the evaluator and
preserved the previous evaluator script on Aisha. Restarted it as PID 761720.
It is currently reloading the previously benchmarked Qwen3-30B-A3B candidate
because that candidate is first in the evaluator sequence; at 11:25 the PS5
screen klog showed 60% load progress. Let this in-flight load finish rather
than interrupting based on a timeout; Gemma3-27B remains next in sequence. The
current API may temporarily time out during model loads. No model assets were
deleted and no download was paused.

Follow-up at 11:26 IST: the evaluator successfully loaded Qwen3-30B-A3B in
117 seconds, displayed load progress through 100% on the PS5, received HTTP
200/healthy, and completed its four streaming tests again (report timestamp
11:26:20). This was a repeat of the already-completed run because the evaluator
restarts from the first candidate; no prior report was used to skip it. Server
`STREAM DONE` rates were 73.8 tok/s (256-token creative), 38.4 tok/s (300-token
code), 86.1 tok/s (200-token logic), and 86.7 tok/s (200-token spatial). All
four ended at their requested length caps; SSE content-chunk counts are not
token counts. The evaluator is now validating the already-staged Gemma3-27B
candidate before its load attempt. Qwen3-32B's aria2 log at 11:26 showed
1.1 GiB / 8.7 GiB (13%), about 2.6 MiB/s, ETA about 49 minutes; Qwen3-4B
remains queued behind it.

At 11:27:08 IST, the evaluator finished exact-size/SHA/config validation for
Gemma3-27B and issued the authenticated dynamic load. The PS5 klog displayed
`LOADING MODEL FILE` and 10% progress (T=16s), confirming on-screen activity.
Qwen3-32B's real aria2 log was at 1.3 GiB / 8.7 GiB (15%), about 2.8 MiB/s,
ETA about 44m41s. Qwen3-4B remains waiting for Qwen32 READY; no transfers were
run in parallel.

### 2026-10-09 11:22 IST: Gemma 3 12B context window frontier discovery: 30,720 tokens (30K)

Executed an automated binary search / step-up frontier probe (`find_frontier.py`) stepping context upwards from 16,384 towards 32,768 with `q4_0` KV cache, `flash_attention=1`, and `max_output_tokens=2048`:

| Context Size | Physical Kilo-tokens | Load Time | Live Compute Test | Status |
|---|---|---|---|---|
| **16,384** | 16K Context | 88.0 s | Passed (1,024 token generation) | **PASS** |
| **20,480** | 20K Context | 102.3 s | 43.0 tok/s (TTFT 377.4 ms) | **PASS** |
| **24,576** | 24K Context | 102.1 s | 43.8 tok/s (TTFT 371.2 ms) | **PASS** |
| **28,672** | 28K Context | 102.3 s | 51.1 tok/s (TTFT 1626.6 ms) | **PASS** |
| **30,720** | **30K Context** | **103.2 s** | **33.9 tok/s decode, 134.4 tok/s prefill** | **🏆 MAXIMUM FRONTIER** |
| **32,768** | 32K Context | >140 s | Exceeded contiguous Vulkan heap | ❌ Timeout / OOM |

**Key Architectural Conclusion:**
The absolute physical context limit of **Gemma 3 12B Q4_K_M** on the base PS5's 12 GiB Vulkan heap is **30,720 tokens (30K Context)** with **2,048 max output tokens**. At 30K context, generation speed remains high at **33.87 tok/s** and prompt evaluation achieves **134.37 tok/s**.

### 2026-10-09 11:32 IST: Gemma 3 27B IT UD-IQ2_XXS benchmarked on PS5

Loaded `gemma-3-27b-it-UD-IQ2_XXS.gguf` (27,009,346,304 parameters, 7,843,690,496 bytes) dynamically onto the PS5 GPU with all 62 layers on Vulkan, 4,096 context, 1,024 max output, `q4_0` KV cache, and Flash Attention.

Executed the 4 standardized streaming workloads:

| Workload | Tokens | TTFT (ms) | Gen Time (s) | Decode Speed (tok/s) | Qualitative Evaluation |
|---|---:|---:|---:|---:|---|
| **Raw Decode Throughput** (Cyberpunk Scene) | 256 | 1,111.5 ms | 13.23 s | **19.27 tok/s** | Evocative narrative, rich metaphorical prose |
| **Algorithmic Code** (Thread-safe LRU Cache) | 350 | 884.8 ms | 24.36 s | **14.32 tok/s** | Immediate Python syntax with Google-style docstrings |
| **Multi-step Reasoning** (Bat & Ball Math) | 200 | 1,496.1 ms | 9.20 s | **21.63 tok/s** | Formulated $2b + 1.00 = 1.10 \implies b = \$0.05$ |
| **Spatial Physics** (Inverted Cup Tracking) | 127 | 1,622.8 ms | 6.37 s | **19.79 tok/s** | Exhibited containment permanence bias (answered microwave) |

**Key observations:**
- Sustained generation speed across workloads: **~19.3 – 21.6 tokens/second** on the 27B parameter dense architecture.
- Does not emit `<think>` blocks; outputs immediate structured answers directly.
- Memory footprint: ~7.84 GB buffer, leaving ~4.4 GB of free headroom on the PS5's 12 GiB Vulkan heap.

### 2026-10-09 11:58 IST: Qwen3-30B-A3B MoE context frontier discovery (28,672 tokens) & benchmark

Executed systematic context frontier discovery (`find_frontier_qwen30b.py`) on `Qwen3-30B-A3B-UD-IQ2_XXS.gguf` (30.5B parameters, ~3.3B active parameters per token, 10,356 MiB buffer). Stepped context from 4,096 tokens up to 30,720 tokens with `q4_0` KV cache, `flash_attention=1`, and `max_output_tokens=1024`:

| Context Size | Physical Kilo-tokens | Load Time | Live Compute Test | Status |
|---|---|---|---|---|
| **4,096** | 4K Context | Baseline | 73.8–86.7 tok/s | **PASS** |
| **8,192** | 8K Context | 136.2 s | 36.1 tok/s (TTFT 1,338 ms) | **PASS** |
| **12,288** | 12K Context | 135.8 s | 130.3 tok/s (TTFT 571 ms) | **PASS** |
| **16,384** | 16K Context | 136.0 s | 160.2 tok/s (TTFT 1,230 ms) | **PASS** |
| **20,480** | 20K Context | 135.9 s | Passed compute smoke test | **PASS** |
| **24,576** | 24K Context | 135.0 s | Passed compute smoke test | **PASS** |
| **28,672** | **28K Context** | **137.0 s** | **Passed compute smoke test** | **🏆 MAXIMUM FRONTIER** |
| **30,720** | 30K Context | >180 s | Exceeded contiguous Vulkan heap budget | ❌ Timeout / OOM |

**Key Architectural Conclusion:**
The absolute physical context limit of **Qwen3-30B-A3B MoE** on the base PS5's 12 GiB Vulkan heap is **28,672 tokens (28K Context)**. Because the model weights consume ~10.36 GB of the 12.0 GiB Vulkan budget, the remaining ~1.7 GB headroom accommodates up to 28K tokens of `q4_0` KV cache before exhausting the unfragmented heap.

#### Full 4-Test Benchmark at 28,672 (28K) Context Frontier

Executed the 4 standardized streaming workloads (`benchmark_qwen30b_moe.py`) on `Qwen3-30B-A3B-UD-IQ2_XXS` at 28K context:

| Workload | Tokens | TTFT (ms) | Gen Time (s) | Decode Speed (tok/s) | Qualitative Evaluation |
|---|---:|---:|---:|---:|---|
| **Raw Decode Throughput** (Cyberpunk Scene) | 256 | 605.9 ms | 3.49 s | **73.10 tok/s** | Rich atmosphere; `<think>` trace before prose |
| **Algorithmic Code** (Thread-safe LRU Cache) | 350 | 715.0 ms | 10.01 s | **34.85 tok/s** | Rigorous architectural decomposition of concurrency & eviction |
| **Multi-step Reasoning** (Bat & Ball Math) | 200 | 1,151.2 ms | 2.08 s | **95.87 tok/s** | Blazing-fast symbolic math reasoning trace |
| **Spatial Physics** (Inverted Cup Tracking) | 200 | 900.7 ms | 2.17 s | **91.80 tok/s** | Systematic physical step-by-step state tracking |

**Key observations:**
- **Unrivaled Throughput:** Thanks to the Mixture of Experts (MoE) sparsity (~3.3B active parameters out of 30.5B total), generation reaches up to **95.87 tokens/sec**, more than 4x faster than dense 27B models (~19–21 tok/s) and more than double dense 12B/14B models (~34–38 tok/s).
- **Sub-Second TTFT:** Time-to-first-token is consistently **605–1,150 ms** across all prompts even at a massive 28K context window.
- **Deep Reasoning Traces:** Emits thorough, insightful `<think>` chains for logic, coding, and spatial reasoning problems.

### 2026-10-09: verified-model archive and Aisha cleanup

The reconnected external disk was positively identified as the Seagate
ST9250315AS, partition `/dev/sdc1`, UUID `6FDF-D05B`, exFAT label
`PS5_STORAGE`. It was mounted read-write at
`/home/user/ps5-storage-250gb`; the stale `/mnt/ps5-storage` mount was not
used. Before copying, the archive target did not exist and the drive had about
167 GiB free.

Copied 12 models that had already loaded and completed the standard PS5
benchmark suite. Every copy was size-checked and SHA-256 compared against its
Aisha source before the successful archive command exited:

| Model files | Count | Total bytes |
|---|---:|---:|
| Qwen2.5-7B (2 shards) | 2 | 4,683,073,632 |
| Qwen2.5-14B (3 shards) | 3 | 8,988,110,496 |
| Qwen3-8B | 1 | 5,027,784,512 |
| Qwen3.5-9B | 1 | 5,680,522,400 |
| Qwen3-14B | 1 | 9,001,753,184 |
| Qwen3.8-27B UD-IQ2_S | 1 | 8,371,970,048 |
| Gemma3-12B | 1 | 7,300,778,336 |
| Qwen3-30B-A3B UD-IQ2_XXS | 1 | 10,362,261,280 |
| Gemma3-27B UD-IQ2_XXS | 1 | 7,850,253,408 |
| **Total** | **12** | **67,266,507,296** |

Archive folder: `/home/user/ps5-storage-250gb/PS5-Language-Models/`.
`SHA256SUMS-and-sizes.txt` records each relative path, byte count, and digest.
The successful copy job checked source size, destination size, and matching
source/destination SHA-256 for every file and exited with
`ARCHIVE_COPY_VERIFIED files=12 bytes=67266507296`. A redundant second full
read was stopped while it was still hashing (before deletion); the completed
copy job's per-file verification is the archive integrity result.

After that, a final exact-path cleanup rechecked the archive manifest, the
verified USB UUID/read-write mount, matching source and destination byte sizes,
and that none of the 12 Aisha source files were in use. It then removed only
those 12 files under `/home/user/models/llmfit/`. Aisha's filesystem changed
from about 22.5 GB free before cleanup to about 84 GB free afterward. The USB
filesystem showed about 105 GB free. The separate Qwen2.5-Coder-3B file was
retained because it was not established to be a failed download. The active
Qwen3-32B and Qwen3-4B staged model files were not part of this archive or
deletion.

Two verified failed duplicate artifacts were also removed from Aisha earlier:
the checksum-failed Qwen2.5-14B shard-1 backup plus its aria2 metadata, and a
bad duplicate Qwen2.5-7B shard. Valid shards were retained on the drive.

At the end of cleanup, the aria2 RPC reported no active transfers. Qwen3-32B
was SHA-verified and staged on PS5 at 12:13 IST; Qwen3-4B was SHA-verified
and staged at 12:27 IST. Their logs say `READY` and require a rollback-safe
manual load/benchmark; neither was reported as loaded/tested. A direct PS5
`/health` probe to the prior LAN address failed to connect, so current PS5
inference health was not verified in this cleanup pass.

### 2026-10-09 12:46 IST: staged Qwen3-32B and Qwen3-4B evaluated

Resumed the sequential evaluation over the active PS5 Tailscale address
`ps5.tailnet.example`. The console answered shell and klogsrv requests over
Tailscale; unauthenticated inference health returned 401 as expected. An
already-running 2,048-token Qwen3-30B response initially delayed health calls,
so the candidate load was not sent until the visible stream finished. The PS5
status panel was enabled throughout validation and loading.

Before each candidate, made a distinct PS5-side copy of the current persistent
startup config and verified its SHA-256 over FTP:

- `inference.before-qwen3-32b-20261009.conf`
- `inference.before-qwen3-4b-20261009.conf`

Both matched the unchanged Gemma3-12B startup config hash
`7f7bab08ddddd1caa169e84a2ea48c571ed8e03c2bc1fed83b8cb5559371369e`.
These dynamic loads did not change `inference.conf`. The inference endpoint
retains its own previous-model rollback path if a dynamic load fails.

The Aisha source model, prepared config, and PS5 staged file were checked for
exact SHA/size before each load. Both candidates used 4,096 context, 1,024
output tokens, four CPU threads, Q4_0 K/V cache, Flash Attention enabled,
batch 128, and all-layer Vulkan offload (`gpu_layers=999`). The attached
`pasted-text-1.txt` suite was used; its four prompts match the repository's
standard streaming suite.

#### Qwen3-32B UD-IQ2_XXS

Loaded as `qwen3 32B IQ2_XXS - 2.0625 bpw`; the endpoint returned healthy at
12:42:19 IST after about 131 seconds from the load request. All four streaming
tests exited successfully at 12:44:12 IST. PS5 `STREAM DONE` is the authoritative
decode rate; SSE content events are not tokenizer tokens:

| Workload | SSE events | Client TTFT | PS5 decode | Finish / observed quality |
|---|---:|---:|---:|---|
| Creative fiction | 256 | 734 ms | 18.7 tok/s | Length cap; stayed in planning, no finished scene |
| LRU-cache code | 300 | 848 ms | 15.7 tok/s | Length cap; reasoning only, no implementation |
| Bat/ball + strawberry | 200 | 56.84 s* | 19.4 tok/s | Length cap; stopped before final answers |
| Inverted-cup spatial | 200 | 1,385 ms | 19.3 tok/s | Length cap; stopped before conclusion |

*The logic-test TTFT is contaminated: another long 1,000-token request ran
between tests 2 and 3, visible in the PS5 log at 18.2 tok/s. Its roughly
57-second wait is queueing, not model prefill. Qwen3-32B stayed healthy after
the suite; it is not the currently loaded model because Qwen3-4B was evaluated
next.

#### Qwen3-4B Q4_K_M

Loaded as `qwen3 4B Q4_K - Medium`; healthy and HTTP-ready at 12:46:18 IST.
The four streaming tests exited successfully at 12:46:34 IST:

| Workload | SSE events | Client TTFT | PS5 decode | Finish / observed quality |
|---|---:|---:|---:|---|
| Creative fiction | 256 | 216 ms | 78.7 tok/s | Length cap; planning, scene unfinished |
| LRU-cache code | 300 | 249 ms | 39.9 tok/s | Length cap; no completed implementation |
| Bat/ball + strawberry | 200 | 316 ms | 96.1 tok/s | Length cap; equation started, final answers cut off |
| Inverted-cup spatial | 200 | 315 ms | 95.4 tok/s | Length cap; no final location answer |

`GET /v1/stats` after the run reported the Qwen3-4B model at 4,022,468,096
parameters and a 2,491,323,904-byte llama model buffer. Vulkan0 reported a
12,884,901,888-byte physical heap, 10,869,669,888-byte budget, 4,726,849,536
bytes free, and an 8,158,052,352-byte used estimate (63.3%). This is the
backend heap estimate; it is not a full PS5 UMA/system-memory accounting.

At 12:46 IST, the two staged follow-on candidates have both been evaluated;
there were no active aria2 transfers at the preceding cleanup check. Qwen3-4B
is currently loaded and healthy. No further model download was started in this
evaluation pass.

### 2026-10-09 12:45 IST: Qwen3-30B-A3B MoE complex coding performance benchmark

Configured `Qwen3-30B-A3B-UD-IQ2_XXS.gguf` on PS5 with `context_tokens=16384` and `max_output_tokens=2048` (`q4_0` KV cache, Flash Attention). Executed the comprehensive automated coding benchmark suite (`ps5_chat/benchmark_coding_qwen30b.py`) spanning 4 complex systems programming tasks:

| Workload / Task | Generated Tokens | TTFT (ms) | Gen Time (s) | Sustained Decode Speed | Reasoning & Output Behavior |
|---|---:|---:|---:|---:|---|
| **1. Async Token Bucket Rate Limiter** | 1,500 | 875.2 ms | 39.18 s | **38.26 tok/s** | Formulates exact token formula `(needed - curr) / rate` inside `<think>` |
| **2. In-Memory B+ Tree Indexing Engine** | 1,800 | 842.4 ms | 47.39 s | **37.96 tok/s** | Implements `LeafNode`, `InternalNode`, `_find_insert_index` inside `<think>` |
| **3. Recursive Descent Parser & AST Evaluator** | 1,600 | 6,714.0 ms | 42.02 s | **38.05 tok/s** | Exhaustive grammar rules, precedence climbing, and AST node design |
| **4. Hierarchical Pub/Sub Broker with Wildcards**| 1,500 | 985.6 ms | 39.23 s | **38.21 tok/s** | Evaluates trie vs segment parsing for MQTT `+` and `#` wildcards |

#### Key Architectural Findings on Qwen3-30B-A3B Coding Performance:

1. **Sustained High-Speed Decode (38 tok/s):**
   - On demanding algorithmic and systems code, Qwen3-30B-A3B delivers a remarkably uniform **~38.0 – 38.3 tokens/second** sustained over 40–50 seconds of continuous generation.
   - This is **~2.5× faster** than dense 27B models (Gemma 3 27B @ ~14.3 tok/s, Qwen 3.8 27B @ ~14.8 tok/s on code).
2. **Sub-Second TTFT:**
   - Time-to-first-token is **842 ms – 985 ms** across all complex prompts with Flash Attention enabled.
3. **Internal Monologue Dynamics & Output Ceiling:**
   - The model features an intrinsic, highly verbose `<think>` reasoning phase that cannot be suppressed via prompt instruction or temperature adjustments.
   - For complex multi-requirement tasks, its chain-of-thought exhaustively drafts and verifies code *inside* `<think>`, spanning **1,500 to 2,500+ tokens**.
   - Because the PS5's 12 GiB Vulkan heap caps `max_output_tokens` at **2,048** to prevent VRAM exhaustion with the 10.36 GB base model, the generation hits the token cap before the model can close `</think>` and render the markdown code block.
   - **Contrast with non-reasoning models:** Dense models like Gemma 3 12B/27B generate immediate runnable Python code with zero `<think>` overhead, fitting within standard 500–1,000 token envelopes, while Qwen 3.8 27B's thinking traces are much shorter (~300 tokens) and close within the 2K cap.

### 2026-10-09 12:53 IST: sequential compact coding-model downloads

Added `ps5lm-tests/queue-coding-models.sh` and started it on Aisha after
confirming 84 GB free, both designated Wi-Fi adapters up, no active model
transfer, and Wi-Fi as the default route. It downloads one model at a time,
resumes partial aria2 transfers, requires a 2 GiB free-space reserve, pins HF
revisions, and verifies exact source byte counts and SHA-256 before continuing.
It preserves any complete-looking file that fails verification. Qwen3-4B
remains loaded and healthy on the PS5; this download worker does not touch the
console or replace its active model.

The queue order and pinned artifacts are:

| Order | Model | File size | SHA-256 |
|---|---|---:|---|
| 1 | Qwen2.5-Coder-1.5B-Instruct Q4_K_M | 1,117,320,768 bytes | `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046` |
| 2 | Granite 3.3-2B-Instruct Q4_K_M | 1,545,303,328 bytes | `ac71e9e32c0bea919b409c5918f69ca74339854b0319c5065e4e9fb6d95c4852` |
| 3 | Qwen2.5-Coder-7B-Instruct Q4_K_M | 4,683,073,536 bytes | `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` |

At the last live check (12:56 IST), Qwen2.5-Coder-1.5B was the sole model
transfer (aria2 PID 827487); no verified completion or percent was yet
recorded. `queue-muse-glimmer-after-coding.sh` is separately running and waits
for the three-model queue's completion marker, so it cannot download in
parallel. It will fetch the user-requested Muse-Glimmer only after the coding
models have completed.

Muse-Glimmer metadata is pinned to HF revision
`faa5b025c584459c13febfa5c59883516710ae39`, file
`Muse-Glimmer-30B-UD-IQ2_XXS.gguf`, 10,746,373,152 bytes, SHA-256
`3e9bd0e609df9d0121ba18c6556de496e7be880184cf22c5dbafbc0600875ad9`. A
read-only string inspection of the deployed PS5 executable found its compiled
`muse-glimmer` model implementation, so the active build appears to recognize
the architecture; this is not yet a model-load test. This is an agentic,
multimodal 30B model (including a perception encoder), not a small
code-specialist or an image-generation/diffusion model. Its GGUF file alone is
close to the PS5's measured Vulkan allocation budget, so a load/OOM failure is
plausible; attempt only after download SHA verification, at conservative
context, with dynamic rollback. Any evaluation should use text-only coding
prompts unless multimodal HTTP input has separately been verified.

At 13:03 IST, `monitor-downloads.sh --once` showed Qwen2.5-Coder-1.5B
SHA-verified, Granite 3.3-2B at 35% (522 MiB / 1.4 GiB, 3.3 MiB/s, displayed
ETA 4m44s), and Qwen2.5-Coder-7B queued. The Muse-Glimmer worker was waiting
behind the compact coding queue. The monitor now displays live aria2 percent,
transferred amount, rate, ETA, freshness, per-model verification state, and
the Muse-Glimmer wait/download state; it does not use preallocated file length
as progress. `--once` provides a single refresh; otherwise the existing
refresh loop remains. `bash -n` and `git diff --check` passed, and the live
read-only refresh verified the display.

### 2026-10-09 13:20 IST: Gemma 3 12B IT complex coding benchmark on PS5

Loaded `gemma-3-12b-it-Q4_K_M.gguf` dynamically onto PS5 with `context_tokens=16384` and `max_output_tokens=2048` (`q4_0` KV cache, Flash Attention). Executed the standardized complex coding benchmark suite (`ps5_chat/benchmark_coding_gemma12b.py`):

| Workload / Task | Generated Tokens | TTFT (ms) | Gen Time (s) | Decode Throughput | Functional Test Status | Output Quality & Behavior |
|---|---:|---:|---:|---:|:---:|---|
| **1. Async Token Bucket Rate Limiter** | 838 | 1,482.7 ms | 42.10 s | **19.88 tok/s** | **✅ PASS** | Fully functional async rate limiter; passed burst & concurrency tests |
| **2. In-Memory B+ Tree Indexing Engine** | 1,400 | 1,027.3 ms | 71.59 s | **19.54 tok/s** | ❌ (Length cap) | Generated complete `LeafNode` & traversal; hit 1,400 token cap |
| **3. Recursive Descent Parser & AST Evaluator**| 1,923 | 1,105.6 ms | 98.11 s | **19.60 tok/s** | ❌ (Precedence) | Complete Lexer, AST & Parser generated; inverted add/mult precedence |
| **4. Hierarchical Pub/Sub Broker** | 908 | 1,322.1 ms | 46.32 s | **19.58 tok/s** | ❌ (Wildcard) | Clean thread-safe broker with UUIDs; rigid length check before `#` |

**Key observations:**
- **Zero `<think>` Overhead:** Gemma 3 12B generates clean, documented Python code directly without consuming token budget on internal monologues.
- **Functional Code Execution:** Successfully passed the rigorous concurrent async rate limiter test on the first try.
- **Sustained Heavy Generation:** Sustained **~19.6 tokens/sec** continuously for 70–98 seconds of dense code generation without thermal throttling or instability.

### 2026-10-09 13:26 IST: Qwen2.5-Coder-1.5B loaded and coding-tested

Verified Aisha's source GGUF at 1,117,320,768 bytes and SHA-256
`cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`.
Before switching, the active model proved to be Gemma 3 12B (not Qwen3-4B as
an earlier note had indicated). Saved a unique copy of the persistent startup
config on the PS5 as
`inference.before-qwen25-coder-1.5b-sampled-20261009-132643.conf`; downloaded
it back and verified the same SHA-256 as the original config
(`7f7bab08ddddd1caa169e84a2ea48c571ed8e03c2bc1fed83b8cb5559371369e`). The
startup config itself was not changed.

Loaded `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` dynamically with all layers
on Vulkan, 4,096 context, 1,024 output cap, four threads, batch 128, Q4_0 KV,
Flash Attention, and the Qwen2.5-Coder published sampling values: temperature
0.7, top-p 0.8, top-k 20, repetition penalty 1.1, repeat-last-n 64. Load and
HTTP health succeeded in 25.1 seconds. The PS5 stats then reported a 1.11 GB
model file / 1.78B parameters, 4.81 GB free Vulkan heap, and 7.27 GB backend
budget. The log showed all model layers assigned to Vulkan.

The four streaming coding prompts ran to completion; the PS5's native
`STREAM DONE` decode rates were 120.5, 130.6, 131.5, and 132.8 tok/s. Quality
was poor despite the corrected sampler: generated outputs were repetitive or
malformed, with only 2 of 18 simple requested code-string checks present. A
separate no-system-prompt `add(a, b)` control also returned malformed Python;
a tiny exact-text smoke prompt returned `2 = 4` rather than the requested
`2+2=4`. This confirms the issue was not just the coding harness's system
prompt. It is a verified poor-quality run on this PS5/server stack, not proof
that the upstream model artifact is intrinsically defective. Full outputs and
per-prompt checks are retained at
`/home/user/models/llmfit/benchmark-reports/qwen25-coder-1.5b-sampled-20261009-132643.json`.
The first greedy run at temperature 0 was superseded; the validated rerun used
the published sampler settings. The client harness now reports elapsed wall
time and TTFT, not misleading SSE-chunk-derived token rates.

Restored and verified Gemma 3 12B as the active, healthy model at 13:30 IST
(16,384 context, 2,048 output, four threads, all Vulkan layers, Q4_0 KV, Flash
Attention). The PS5 screen/log showed model-load milestones. The separate
sequential coding download queue was not paused or changed; at 13:29 IST
Qwen2.5-Coder-7B was at 89% (3.9/4.3 GiB, 3.6 MiB/s, ETA about 2 minutes).

### 2026-10-09 13:38–13:49 IST: sequential Granite and Qwen2.5-Coder-7B tests; Muse test armed

Extended the chain without changing the persistent startup config or stopping
the separate Muse download. For each compact model, verified Aisha's source
size/SHA-256, staged it to the PS5, read the full staged file back over FTP to
verify SHA-256, saved a unique byte-verified copy of `inference.conf`, and
dynamically loaded at 4,096 context / 1,024 output with all layers on Vulkan,
Q4_0 KV, Flash Attention and four threads. The PS5 status display emitted
`MODEL LOADED` and `DYNAMIC MODEL SWITCH COMPLETE` for each load.

#### Granite 3.3-2B-Instruct Q4_K_M

Loaded and HTTP-ready in 34.2 seconds. Used exploratory non-greedy sampling
(temperature 0.7, top-p 0.9, top-k 0, repeat penalty 1.0); IBM's published
generation config leaves those sampling parameters unspecified, so these are
test settings, not IBM-recommended defaults. All four streamed coding tasks
finished. PS5-native `STREAM DONE` rates: 105.2, 106.3, 106.1, and 106.1 tok/s.
The simple requested-code-string check score was 14/18. The binary-search
answer contained all four expected markers; the interval implementation
mutated its input despite the request, and the SQL answer used partial-match
`LIKE` semantics instead of exact-name matching. Thus promising for small code
snippets, but not fully correct on this suite. Report:
`/home/user/models/llmfit/benchmark-reports/granite-3.3-code-2b-coding-20261009-134031.json`.

#### Qwen2.5-Coder-7B-Instruct Q4_K_M

Loaded and HTTP-ready in 63.9 seconds using its published sampling values
(temperature 0.7, top-p 0.8, top-k 20, repeat penalty 1.1). All four streamed
requests finished; PS5-native `STREAM DONE` rates were 62.5, 61.0, 61.0, and
66.5 tok/s. The simple code-marker score was 8/18. LRU generation hit the
768-token request cap; binary search contained an invalid empty-input branch,
the interval output was malformed, and SQL generation stopped after only a
few chunks. The model stayed HTTP healthy but did not pass this small coding
suite. Report:
`/home/user/models/llmfit/benchmark-reports/qwen25-coder-7b-coding-20261009-134628.json`.
Latest `/v1/stats`: 2.75 GB free Vulkan heap, 11.34 GB reported backend budget;
the heap usage figures are backend estimates, not full PS5 UMA accounting.

The compact evaluation runner completed at 13:46:48 IST and left Qwen2.5-Coder
7B loaded and healthy. A persistent Aisha-side Muse evaluator was then armed at
13:49 IST. It waits for the Muse downloader's explicit completion marker, then
rechecks source size/SHA, stages and fully SHA-verifies the PS5 copy, makes a
unique startup-config backup, and attempts a conservative 512-context / 128-
output text-only smoke test. On load failure it verifies the server's automatic
rollback to the prior model. The pinned Muse download passed its expected
10,746,373,152-byte size and SHA-256 at 14:30 IST. PS5 staging began at 14:32;
at 14:35 the upload process had read 69% of the source file. This is source
read progress, not a claim that the PS5 has acknowledged the same fraction.
Full staged-file SHA verification and the conservative load/smoke test remain
pending.

### 2026-10-09: Muse minimal-prompt diagnosis

Verified the active PS5 model via authenticated `/health`: Muse-Glimmer 30B
UD-IQ2_XXS, loaded with 44 Vulkan layers, 512 context, and 128 output cap.
Sent the smallest plain-text chat request (`Hi`, one output token, no system
message/history). It still returned HTTP 413 `formatted prompt exceeds limit`.
Thus the failure is not caused by a large client request or conversation
history; it occurs while this build applies Muse's model-provided chat
template. `http_server.cpp` currently formats into a fixed 65,536-byte buffer
and treats the llama.cpp “buffer too small” result as that 413.

The same authenticated `/v1/stats` snapshot reported zero free bytes in the
Vulkan heap and `reported_usage_valid=false` (driver-reported usage exceeds
the physical heap; the endpoint warns these are backend estimates). No prompt
reached tokenization or inference, so there is no Muse generation benchmark.
Do not simply enlarge the prompt buffer and immediately decode on this
configuration: first inspect/log the template's required formatted length,
then assess context and memory headroom. A safe next code fix is bounded
dynamic sizing for template formatting with a clear upper bound and screen
status, built on Aisha; only run inference if memory headroom is established.

### 2026-10-09: Muse removed; next compact model download held

After the user requested Muse removal, made and byte-verified a unique
startup-config rollback copy on the PS5:
`inference.before-muse-removal-20261009.conf` (SHA-256
`7f7bab08ddddd1caa169e84a2ea48c571ed8e03c2bc1fed83b8cb5559371369e`).
Switched the active PS5 service back to the already-staged
`qwen2.5-coder-7b-instruct-q4_k_m.gguf`; `/v1/models/load` returned HTTP 200
and authenticated `/health` confirmed it healthy at 4,096 context / 1,024
output. Removed only `Muse-Glimmer-30B-UD-IQ2_XXS.gguf` from the PS5 app data
folder and verified it absent from the FTP listing. The Aisha source copy was
not deleted.

The next proposed compact coding baseline was pinned from the official
`Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF` repository at revision
`ebb2015119c907b064c512bf053e945850b5875f`: 491,400,064 bytes, SHA-256
`1d9614638d18024d0fbb36575a15f1302a3adf044df10345688ec4f6e1c4ff32`.
Its resumable Wi-Fi-only aria2 download on Aisha (PID 923087) reached 437/468
MiB (93%) at 2.6 MiB/s, ETA 11 seconds, in the last verified snapshot. The
GGUF is preallocated, so that length is not proof of completion. At the user's
instruction to wait while they choose/switch models, no model load or test was
started; the PS5 remains on the verified Qwen2.5-Coder-7B. The candidate
download was left untouched and may finish in the background. Do not load or
test another model until the user asks to resume.

### 2026-10-09 15:40 IST: Aider-style diff & chunking harness implemented with PS5 backend

Integrated and tested the open-source **Aider** ecosystem and created a dedicated local Aider-style diff harness ([`ps5_chat/diff_harness.py`](file:///Users/user/Documents/antigravity/proud-salk/ps5_chat/diff_harness.py)) pointing to the PS5 GPU endpoint (`http://ps5.tailnet.example:8081/v1`):

1. **Aider Official CLI Integration (`uvx aider`):**
   - Configured `model_metadata.json` and `model_settings.yml` to specify the PS5's 16,384 context window and 2,048 max output token capacity.
   - Diagnosed that standard Aider `diff` mode sends ~1,867 tokens of complex few-shot prompt examples which can confuse compact local models.
2. **Dedicated PS5 Diff & Chunking Harness (`diff_harness.py`):**
   - Implemented a tailored SEARCH/REPLACE diff engine with exact and whitespace-tolerant matching.
   - Evaluated on live code edits against `math_utils.py` on the PS5 GPU:
     - **Edit 1 (Add `divide` function):** Generated only **91 tokens** in **2.30s** (**39.19 tok/s**). Applied 1/1 block cleanly; `py_compile` syntax check passed.
     - **Edit 2 (Modify `add` signature to float):** Generated only **56 tokens** in **1.44s** (**38.27 tok/s**). Applied 1/1 block cleanly; `py_compile` syntax check passed.
3. **Key Architectural Conclusion:**
   - By shifting from monolithic full-file generation (which requires 1,500–2,000+ tokens) to localized SEARCH/REPLACE diff blocks (which require only **50–100 tokens** per edit), the PS5 small-model stack completely eliminates output token exhaustion and context truncation while delivering sub-2-second edits at ~38–39 tok/s.

### 2026-10-09: Gemma 4 family shortlist for PS5

Read Google's Gemma 4 overview and official Google GGUF/QAT model cards. The
family offers E2B, E4B, 12B, 26B A4B, and 31B. Official Google Q4_0 GGUF repo
cards list these approximate file sizes: E2B 3.35 GB, E4B 5.15 GB, 12B 6.98
GB, 26B A4B 14.4 GB, and 31B 17.7 GB. Google's inference-memory table gives
rough Q4_0 requirements of 2.9, 4.5, 6.7, 14.4, and 17.5 GB respectively.
The E2B/E4B/12B sizes are plausible PS5 candidates; 26B A4B and 31B Q4_0
exceed the console's approximately 12 GiB Vulkan heap and should not be tried
in these Q4_0 variants. E2B is the safest first compatibility/performance
test, then E4B, then 12B as a tighter fit with a conservative context.

Verified the Aisha vendored llama.cpp source at commit `552f18f912a3` contains
`src/models/gemma4.cpp`; the current PS5-linked `/home/user/llama-vk/build/src/libllama.a`
also exposes Gemma4 symbols (38 `nm` matches). This is strong build-level
compatibility evidence, not a successful on-console Gemma 4 load. The custom
PS5 HTTP endpoint is text-only, so any first Gemma 4 run would test text
inference, not its multimodal image/audio features.

The previously held Qwen2.5-Coder-0.5B download subsequently completed on Aisha
(aria2 `OK`); no PS5 load or benchmark was started. The user asked to research
Gemma 4 and explicitly said not to load another model while they were choosing.

### 2026-10-09: Gemma 4 12B selection and reasoning-mode findings

The user selected Gemma 4 12B as the first Gemma 4 target, based on Gemma 3
12B's prior PS5 performance. Two practical QAT GGUF candidates were compared:

| Artifact | Quantization | Listed file size | Notes |
|---|---|---:|---|
| Google `gemma-4-12B-it-qat-q4_0-gguf` | QAT Q4_0 | 6.98 GB | Official Google artifact; straightforward compatibility baseline |
| Unsloth `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | QAT Dynamic Q4_K_XL | 6.72 GB | Smaller by listed size; HF revision `f18012b8f690e563b7f872cb764b4cb3de90b14a`; SHA-256 `cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165` |

The Unsloth file is the selected download candidate: it is the smaller of the
two reviewed QAT 12B artifacts and the llama.cpp build already linked into the
PS5 title has Gemma 4 architecture support. This remains a preflight conclusion;
no Gemma 4 file has yet been downloaded, staged, or loaded. Bonsai is a separate
model family rather than a Gemma 4 quantization. Its earlier Q1_0 format is
upstream-friendly, while Bonsai 2's rotated-basis PQ2_0/PTQ1_0/Q2_0 artifacts
need PrismML's llama.cpp fork/transforms; the cited release matrix marks
Vulkan PQ2_0 unsupported. That is not a safe match for the current PS5 build.

Read-only live preflight on Aisha: `/home` had 67 GB available; the compact
coding and Muse downloader workers were not running (the general aria2 RPC
daemon was). Tailscale ping to the PS5 and TCP port 8081 succeeded, but the
inference API did not return `/health` or `/v1/models` during 20-second probes;
the API also refused a local loopback connection from Aisha. Therefore the
active model could not be verified and no upload/load action was taken.

The Gemma 4 candidate has now been started on Aisha as a standalone resumable
aria2 transfer, bound to Wi-Fi adapters `wlx58044f3f6ec0,wlx90de80301119`:
`/home/user/models/llmfit/gemma4-12b/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`.
aria2 PID 939045 was verified alive with established HTTPS sockets; its
`.aria2` sidecar exists. The GGUF is preallocated, so its apparent length is
not progress. The first sample was too early to derive a reliable percent or
rate; hash verification is pending. The existing aria2 RPC daemon's only
waiting item is the user's paused Evil Within 2 download, which was not
changed. The transfer is not yet included in `monitor-downloads.sh`; the PS5
remains untouched.

The current `http_server.cpp` calls `llama_chat_apply_template` with messages
and `add_ass=true`, and exposes no `enable_thinking`, reasoning-effort, or
template-kwargs control. Thus there is no generic API switch today. Stored
findings show Qwen3-family models emit `<think>` traces, while Gemma 3 12B and
27B do not. Qwen3 prompts may offer model/template-specific `/think` or
`/no_think` controls, but the current service has not validated or implemented
them; do not treat a prompt hint as a dependable server setting. A proper
per-request toggle requires model/template-aware handling and tests. Gemma 4
thinking behavior likewise remains to be tested with the Gemma 4 chat template.

Updated `ps5lm-tests/monitor-downloads.sh` with a dedicated Gemma 4 12B panel
showing the active aria2 PID/elapsed time, parsed live progress/rate/ETA and log
freshness, transfer target, pinned quant, expected SHA-256, and completion or
partial/paused state. Verified with `bash -n`, `git diff --check`, and a live
`--once` run on Aisha: 264 MiB / 6.2 GiB (4%), 3.2 MiB/s, ETA 31m23s; aria2 PID
939045 was active and the log was fresh. This is a snapshot, not a final ETA.

Added and started `ps5lm-tests/queue-after-gemma4.sh` on Aisha (worker PID
958186). It waits for Gemma aria2 to finish, then requires the exact expected
size and SHA-256 before starting the official OpenBMB MiniCPM5-2B Q4_K_M
artifact sequentially over Wi-Fi. MiniCPM is pinned to HF revision
`2079a22f3beaa4e306449978533478fe0522f4b3`, 1,561,318,368 bytes, SHA-256
`ec2d5801640099e97d8d7e8003ad4d81f336e757811f03a26173dddf386602fd`. At
worker start, Gemma aria2 PID 939045 was still active; `/home` had 60 GB free.
The worker only downloads and verifies source files; it does not stage or load
anything on the PS5.

K2 Horizon 3.7B was not queued despite its high AA score: its HF model card
states that it needs K2 Horizon architecture support from the IFM llama.cpp
fork, and read-only inspection found no K2 symbols in Aisha's pinned PS5
llama.cpp source/library. Revisit only after a compatible backend build is
available and assessed.

The download monitor now reports the follow-on queue worker and its last step,
MiniCPM's live progress/state, exact expected SHA-256, and the reason K2 is
blocked. `bash -n` and `git diff --check` passed. The queue is serial; no PS5
model switch or staging was authorized or performed.

### 2026-10-09: Multi-Turn Auto-Continuation vs. Aider SEARCH/REPLACE Diff Harness

To address the PS5's 2,048 token generation limit on large coding benchmarks, two harness architectures were tested with `gemma-3-12b-it-Q4_K_M.gguf`:

1. **Auto-Continuation Multi-Turn Harness (`ps5_chat/benchmark_coding_with_continue.py`):**
   - Automatically detects when generation truncates (`finish_reason == 'length'`), captures the partial code, and prompts:
     `"Continue generating the remaining code exactly where you left off. Do not repeat prior code, do not apologize, and output raw code only."`
   - Chaining tested up to 3 turns across three challenging software engineering tasks:
     * **Task 1 (Async Token Bucket Rate Limiter):** 586 tokens across 2 turns at 27.41 tok/s.
     * **Task 2 (In-Memory B+ Tree Indexing Engine):** 1,276 tokens across 2 turns at 32.24 tok/s.
     * **Task 3 (Recursive Descent AST Parser & Evaluator):** 2,100 tokens across 3 turns at 29.05 tok/s.
   - **Failure Analysis ("The Suture Problem"):**
     While the multi-turn harness successfully bypassed the token ceiling (accumulating up to 2,100 tokens), **all three assembled tasks failed unit test syntax checks**:
     * *Mid-Statement Fence Injection:* In Turn 2, the model instinctively opened a new markdown block (````python`), which was concatenated directly onto the cut-off token from Turn 1 (e.g. `child.keys = child.keys[:self.degree```python`), corrupting AST parsing.
     * *Context Drift / Amnesia:* In Task 3, after writing the parser in Turn 1 and truncating, Turn 2 did not cleanly resume from the truncated method; instead, it restarted from scratch by re-defining `class ASTNode` and `tokenize()`.
     * *Logic Faults in Generated Code:* In Task 1, Gemma 3 wrote an infinite loop bug in its timeout logic (`wait_time = timeout - (time.monotonic() - time.monotonic())`). Test execution required a subprocess 10-second timeout guard to avoid blocking.

2. **Aider SEARCH/REPLACE Diff Harness (`ps5_chat/diff_harness.py`):**
   - Official Aider CLI (`uvx --from aider-chat aider`) and standalone diff harness tested on PS5 inference endpoint.
   - Instead of monolithic multi-thousand token generation, tasks are broken down into localized edits using:
     ```
     <<<<<<< SEARCH
     [existing code block]
     =======
     [updated code block]
     >>>>>>> REPLACE
     ```
   - **Results:**
     * Edit 1 (Add docstrings & type hints to `math_utils.py`): 91 tokens, 2.30s, 39.5 tok/s. Exact match, 100% clean `py_compile`.
     * Edit 2 (Add `safe_divide` with ZeroDivisionError check): 56 tokens, 1.44s, 38.8 tok/s. Exact match, 100% clean `py_compile`.
   - **Conclusion:**
     Small/medium models (12B-30B) running on local edge hardware are poorly suited for raw multi-turn code continuation due to suture-point syntax corruption. In contrast, **diff-based chunking (Aider style)** is exceptionally reliable on PS5 hardware: it keeps generation payloads under 100 tokens, executes in <2.5 seconds, and maintains 100% syntactic integrity.


### 2026-10-09: Granite 3.3-2B-Instruct Q4_K_M Comprehensive Benchmark (Regular 4-Part & Coding)

Granite 3.3-2B-Instruct (`granite-3.3-2b-instruct-Q4_K_M.gguf`, 1.54 GB) was loaded on the PS5 GPU at 4,096 context / 1,024 max output with all layers offloaded to Vulkan, Q4_0 KV cache, Flash Attention, and 4 threads.

#### 1. Performance Overview: Blazing Fast Decode Throughput
- **Decode Speed:** **100.3 to 256.1 tokens/sec** across all tests (fastest model tested on PS5 to date).
- **Time to First Token (TTFT):** **344 ms – 594 ms**.
- **Memory Footprint:** Only ~1.54 GB GGUF weight allocation, leaving >10 GB VRAM free.

#### 2. Part 1: Standard 4-Part Suite
- **Test 1: Raw Decode (Cyberpunk Fiction):** 256 tokens in 2.23s (**114.45 tok/s**, TTFT: 585.7ms). Fluent, vivid prose.
- **Test 2: LRU Cache Prompt:** 350 tokens in 3.48s (**100.34 tok/s**, TTFT: 344.5ms). Complete implementation.
- **Test 3: Logic (Bat & Ball + Strawberry):** 205 tokens in 1.60s (**127.25 tok/s**, TTFT: 532.8ms).
  * Bat & Ball: Correctly solved $x + (x+1) = 1.10 \rightarrow x = 0.05$ (5 cents). **CORRECT**.
  * Strawberry: Correctly counted 3 'r's. **CORRECT**.
- **Test 4: Spatial Physics (Ball in Inverted Cup):** 76 tokens in 0.29s (**256.13 tok/s**, TTFT: 594.6ms).
  * Claimed ball remained inside inverted cup when lifted and moved into microwave. **INCORRECT** (failed gravity tracking).

#### 3. Part 2: 4-Part Coding Suite with Live Unit Tests
- **Task #c1: LRU Cache (`collections.OrderedDict`):** **PASS** (164 tokens, 119.56 tok/s). Fully functional O(1) eviction and updates.
- **Task #c2: First Index Binary Search:** **FAIL** (233 tokens, 103.1 tok/s). Bug in duplicate branch: set `left = mid + 1` instead of `right = mid - 1`, searching right instead of left for the first duplicate.
- **Task #c3: Merge Intervals (Non-Mutating):** **FAIL** (147 tokens, 126.44 tok/s). Mutated the input list in-place via `intervals.sort()` despite explicit negative constraint.
- **Task #c4: Parameterized SQLite Query Builder:** **FAIL** (131 tokens, 101.92 tok/s). Hardcoded `cursor.execute(query, (name, min_age))` causing parameter count mismatch when filters were omitted.

### 2026-10-09: Gemma 4 chat-template fix and verification

- Root cause of the prior HTTP 413 was misreading the pinned llama.cpp
  `552f18f91` legacy `llama_chat_apply_template` return value: `-1` means
  unsupported template, not the required buffer size. Gemma 4 and Muse need
  llama.cpp's common Jinja/chat-template engine.
- Built `llama-common` on Aisha with `LLAMA_BUILD_COMMON=ON` and
  `LLAMA_SUBPROCESS=OFF` (the PS5 SDK lacks the optional subprocess API), linked
  it into the Vulkan title, and added the chat-template formatter plus a
  positive-length-only legacy fallback. The HTTP endpoint, sampling, tokenizer,
  and streaming code paths were otherwise retained. `enable_thinking` is now
  an optional per-request boolean, defaulting to `true`.
- Built on Aisha; final `out/eboot.bin` SHA256:
  `644c1506ddaab791d383937d85ca61934893edb9fc290eaf5ce58dd18d14a74d`.
  The deployed PS5 title has a unique rollback copy at
  `/data/homebrew/PPSA99005/eboot.before-thinking-toggle-20261009.bin`; it was
  byte-compared with the previously deployed executable before replacement.
- Startup loaded the configured Gemma 3 12B baseline, passed its local POST
  smoke test, and reported `HTTP READY 0.0.0.0:8081 TAILSCALE + API KEY` on
  the PS5 screen/log. Gemma 3 regression request returned `OK.` (2 completion
  tokens, 37.50 tok/s), confirming compatibility.
- Gemma 4 12B loaded successfully in 84 seconds at 4,096 context / 1,024 output.
  The original minimal request now reaches inference instead of returning 413.
  With default thinking on, a 128-token request completed in 52 tokens at
  38.94 tok/s and ended with `OK`, though the model's thought-channel text is
  included in the response. With `enable_thinking:false`, it returned exactly
  `OK` in 1 token (19 prompt tokens; reported 18.47 tok/s for this tiny decode).
  Authenticated `/health` confirms Gemma 4 remains loaded and healthy.
- The common library includes a dedicated Muse Glimmer parser, so the same
  formatter defect is addressed in code; Muse has not yet been loaded or
  retested on the PS5. No full benchmark suite has been run for Gemma 4.

### 2026-10-09: Muse-Glimmer staged; two conservative load attempts rolled back

The user clarified the target as the PS5 (the earlier “PS4” was a typo). Aisha's
Muse-Glimmer source at
`/home/user/models/llmfit/coding-models/muse-glimmer-30b/Muse-Glimmer-30B-UD-IQ2_XXS.gguf`
was verified at 10,746,373,152 bytes with SHA-256
`3e9bd0e609df9d0121ba18c6556de496e7be880184cf22c5dbafbc0600875ad9`. It was
uploaded to `/data/homebrew/PPSA99005/Muse-Glimmer-30B-UD-IQ2_XXS.gguf`; a full
FTP readback from the PS5 produced the exact same SHA-256. No model file was
deleted.

Before switching, saved and byte-verified
`/data/homebrew/PPSA99005/inference.before-muse-glimmer-20261009-182120.conf`
with SHA-256 `7f7bab08ddddd1caa169e84a2ea48c571ed8e03c2bc1fed83b8cb5559371369e`.
The previous model was Gemma 4 12B, healthy at 4,096 context / 1,024 output.

Two dynamic-load attempts used 512 context, 128 max output, batch 16, all 49
layers on Vulkan, Flash Attention enabled, and `q4_0` for both K and V caches.
The first reached 90% and returned HTTP 503; the endpoint restored Gemma 4.
For the second attempt, disabled GPU KV offload as an additional memory
reduction. The live PS5 log showed the GGUF parsed successfully (Muse-Glimmer
architecture, 10.00 GiB file), all 49/49 layers offloaded, and model buffers
of 6,390.13 MiB on Vulkan plus 540 MiB on Vulkan-host. However, the same log
reported a 4,096-cell KV cache with K and V both **F16 (640 MiB each)** despite
the request specifying 512 context and `q4_0` for both cache types. The load
then returned HTTP 503 and restored Gemma 4; authenticated `/health` confirmed
Gemma 4 healthy at its original settings. No Muse chat request was sent because
the model never reached a healthy loaded state.

The user noted that the 9.65 GB Qwen3-30B-A3B previously fit with quantized KV
cache. Muse's requested quantization is evidently not reflected in the current
PS5 log. Why Muse reports F16/4,096 cells and then fails remains unresolved;
check the model-specific KV-cache path and runtime options before another
attempt. The PS5 is currently healthy on Gemma 4 12B.

### 2026-10-09: Sequential queue added for abliterated variants

After the Muse attempts, added `ps5lm-tests/queue-abliterated-models.sh` and a
dedicated panel to `ps5lm-tests/monitor-downloads.sh`. The queue downloads only
on Aisha, uses its Wi-Fi interfaces, preserves resumability, refuses to overlap
another model transfer, maintains a 2 GiB free-space reserve, and verifies exact
source size and SHA-256. It does not stage or load models to the PS5.

1. Huihui Qwen3.8-27B abliterated UD-IQ2_S:
   `Huihui-Qwen3.8-27B-abliterated-UD-IQ2_S.gguf`, pinned revision
   `e222b519fad4e5b61d1f2d936fc180ff14f3de78`, 8,424,398,848 bytes, SHA-256
   `b4b7bdec2e49a66542f5816295a7503f2fc52dd7bd04ca45bc370820cfe81d03`.
   This matches the quant class of the previously tested Qwen3.8-27B baseline.
2. DuoNeural Gemma 4 12B IT abliterated Q4_K_M:
   `gemma4_12b_abliterated_Q4_K_M.gguf`, pinned revision
   `8303662cc0d9455655a8693749ae3820c2f40ba9`, 7,381,381,664 bytes, SHA-256
   `a8fc07f6ec28dc928b05e3dfc60e920ee654114c4d33b73d88fe38e952069434`.
   Hugging Face's file metadata marked both GGUFs safe. The queue order keeps
   the Qwen3.8 ablation close to its tested base first, then compares Gemma 4.
   The queue was started on Aisha as PID 1103964. First live monitor snapshot
   showed Qwen3.8 abliterated at 184 MiB / 7.8 GiB (2%) and 593 KiB/s, ETA
   3h45m39s; this is a fresh-transfer snapshot, not a stable speed estimate.

### 2026-10-09 19:00 IST: low-temperature sampling was 2x slower; fixed (Claude)

Speed-work Phase 0/1 (plan: Phase 0 prep, 1 sampler, 2 profiling, 3 cheap
fixes, 4 prefill batch (needs user OK: watchdog/restart risk), 5 speculative
decoding, 6 kernels). Codex and Gemini were asked not to touch this code
meanwhile.

- Repo `ps5lm-tests/vk/` matched Aisha's `~/ps5lm-inference/vk/` exactly
  before these changes. Aisha's llama.cpp (`third_party/llama.cpp`, commit
  `552f18f91`) carries uncommitted local edits (common, ggml-cpu, ggml-vulkan
  budget stats); the full diff is now saved as
  `vk/llama-patches/all-local-changes-vs-552f18f91.patch`.
- Deploy notes: the PS5's LAN FTP/shsrv ports are closed now; use Tailscale
  `ps5.tailnet.example` for 2121/2323/3232/8081. The PS5 FTP server decrypts SELF
  files on download, so an eboot readback is ~650 KB larger and never matches
  the uploaded hash. Verify instead that the readback equals the build's
  `work/eboot.elf` except for ~253 trailing (non-loaded) bytes.
- A/B on Gemma 4 12B (262K vocab), LRU code prompt, 300 tokens, 3 interleaved
  rounds, PS5 `STREAM DONE` rates, per-request params only:

  | Variant | original build | + FTZ/DAZ | + exp patch |
  |---|---:|---:|---:|
  | greedy | 25.9 ms/tok | 25.9 | 26.0 |
  | temp 0.1, top_k 0 | **50.3** | 46.3 | **27.0** |
  | temp 0.1, top_k 20 / 40 | 26.0 / 25.9 | 26.0 / 26.0 | 26.0 / 26.0 |
  | temp 0.7, top_k 0 | 29.0 | 29.1 | 28.8 |

  Cause: at low temperature most of the vocabulary's `expf(logit - max)`
  underflows, and the PS5 libm's underflow path costs ~75 ns per call (about
  +24 ms/token at 262K vocab, +14.5 ms at Qwen's 151K, which explains every
  earlier "code test runs at half speed" result: that prompt used temp 0.1).
  Fix: `exp_or_zero()` in llama.cpp's softmax and dist samplers skips `expf`
  below -104, where it returns exactly 0, so outputs are unchanged
  (`vk/llama-patches/llama-sampler-exp-underflow.patch`). FTZ/DAZ on the server
  thread (`http_server.cpp`) recovered only ~4 ms; kept, harmless.
- Also added: `/app0/vk-env.conf` (optional, KEY=VALUE, prefixes `GGML_VK_`,
  `RADV_`, `MESA_`, `ACO_`, `PS5_` only) applied with `setenv` before Vulkan
  starts, so driver/backend experiments need a restart, not a rebuild; and
  `PS5_PROGRESS_EVERY` (default 16, 0 = off) for the on-screen
  "GENERATING" line, which flushes the whole 16 MB frame and waits for vblank.
- Deployed: `eboot.bin` = `out/exp-underflow-20261009` (sha256 `00c4020e…`).
  Rollbacks on the PS5: `eboot.before-ftz-envconf-20261009.bin` (Codex's
  17:56 common-chat build, sha `644c1506…`) and
  `eboot.ftz-envconf-20261009.bin`. Aisha source backups:
  `*.before-ftz-envconf-20261009.cpp`, `llama-sampler.before-exp-underflow-20261009.cpp`.
  `inference.conf` unchanged; Gemma 4 12B reloaded afterwards with its
  previous settings (verified identical via `/health`).

### 2026-10-09 19:30 IST: progress-line cost; profiler captured; clean checkpoint (Claude)

- On-screen "GENERATING" line every 16 tokens costs 0.33 ms/token (Gemma 4
  12B greedy 25.97 → 25.64 ms with `PS5_PROGRESS_EVERY=0`); left at default.
- RADV's shader cache already works: PS5_Vulkan's platform sets
  `MESA_SHADER_CACHE_DIR=/app0/radv-shader-cache` (database form); the DB was
  reused across today's restarts. The issue-#5 `setegid` concern needs no action.
- Startup klog: `RADV NAVI21 | uma: 1 | fp16: 1 | int dot: 0 | matrix cores:
  none | warp size: 32`, so compute already runs wave32.
- ggml-vulkan perf-logger profiles (every 64 graphs, greedy 256-token decode,
  q4_0 KV + FA) for Gemma 3 12B, Qwen3-4B, Gemma 3 27B IQ2_XXS and Qwen3-30B-A3B
  are saved in `ps5lm-tests/profiles-20261009/`;
  analysis pending.
- Checkpoint: `vk-env.conf` on the PS5 now holds only a comment (no
  overrides); title restarted; Gemma 4 12B loaded with its earlier parameters
  (`/health` identical to the pre-work snapshot). Deployed eboot remains the
  exp-underflow build.

### 2026-10-09 20:05 IST: tuning paused for user benchmarks; open leads (Claude)

- PS5 is clean: `vk-env.conf` holds only a comment, title restarted, Gemma 4
  12B loaded with its prior parameters (`/health` identical). Deployed eboot is
  still the exp-underflow build (low-temperature sampling fix).
- Perf logger (`GGML_VK_PERF_LOGGER=1`, frequency 64) printed nothing in two
  runs, even with a single klog reader (klogsrv appears to split its stream
  between concurrent readers; `restart.sh` now drops its own reader). Other
  stderr lines (`[llama] ...`) do arrive, so check whether timestamp queries
  work on this RADV or whether the logger never reaches `print_timings`.
- Lead to test next: RADV/PS5 prints `radv/ps5 submissions: window ...` every
  10 s. During decode about 10 `sceAgcDriverSubmitDcb` calls per token take
  ~1.8 ms each on the inference thread (`agc_ms` ≈ 6,500 per 10 s window),
  against ~7 µs per submit during model load. Unknown yet whether that is GPU
  idle time (big win if submissions per token are cut) or just waiting while
  the GPU is busy. Cheap test: `GGML_VK_MAX_NODES_PER_SUBMIT=20` vs default
  and see whether ms/token grows with submit count. ggml-vulkan's other submit
  trigger (1/40 of graph FLOPs) is not env-controlled.
- llama.cpp debug lines (`llama_graph_n_input_tensors`, `create_tensor`) flood
  klog during context setup only (not per token); the title's log callback
  forwards every level. Filtering DEBUG would make klog usable.

### 2026-10-09: abliterated model files transferred to PS5 (not loaded)

After the Aisha download queue completed with source SHA-256 verification, both
GGUFs were copied over the wired Tailscale FTP path into the existing
`/data/PS5LM/models/` library. The PS5 FTP listing confirmed exact expected
byte sizes:

- `Huihui-Qwen3.8-27B-abliterated-UD-IQ2_S.gguf`: 8,424,398,848 bytes.
- `gemma4_12b_abliterated_Q4_K_M.gguf`: 7,381,381,664 bytes.

Neither model was loaded. The active `PPSA99005` app, config, and current model
were left untouched. The Qwen GGUF also exists in the existing app folder only
under its distinct tested-base filename; no destination files were overwritten.

### 2026-10-09: abliterated variants made loader-visible and loaded

The first transfer put the verified files in the shared library
`/data/PS5LM/models/`, but the inference title's `/v1/models` inventory scans
only `/app0/` (the active title directory, `/data/homebrew/PPSA99005/`). Thus
the files were present on the console but not selectable by the loader. The
library copies were retained; full-size copies were placed in the title folder
under unique filenames. The PS5 loader inventory confirmed both entries and
the exact expected sizes. A slow PS5-local copy attempt for the Qwen file was
stopped by its resolved PID and replaced by the complete wired FTP upload.

Using the existing authenticated dynamic-load API, both variants loaded in
sequence at 4,096 context / 1,024 max output, 4 CPU threads, and 999 GPU layers:

- Huihui Qwen3.8-27B abliterated UD-IQ2_S loaded successfully; the endpoint
  identified it as `qwen35 27B IQ2_S - 2.5 bpw`.
- DuoNeural Gemma 4 12B IT abliterated Q4_K_M loaded successfully; the endpoint
  identified it as `gemma4 ?B Q4_K - Medium` (the display name omits the exact
  parameter count).

Final authenticated `/health` reported the Gemma 4 abliterated file loaded and
healthy. The server holds one model at a time, so Gemma 4 is active now and
Qwen3.8 remains available for the next dynamic switch. No chat benchmark was
run. `inference.conf`, the executable, and source code were not changed; the
dynamic switches are runtime-only. Source files had passed their download
SHA-256 checks; PS5 inventory verified exact byte sizes, but no PS5-side SHA
re-hash was performed.

### 2026-10-09 21:35 IST: ps5lab tooling; KV-type investigation; build ready, not deployed (Claude)

**ps5lab** (source `ps5lm-tests/ps5lab/`, installed in Aisha `~/ps5lab/`):
- `klogd.py`: the only klogsrv reader (klogsrv splits its stream between
  concurrent readers). Runs on Aisha as the systemd user service
  `ps5-klogd.service` (enabled, linger on, Restart=always; starts at boot);
  writes `~/ps5lab/klog/YYYY-MM-DD.log` with local timestamps. Read those files;
  never connect to port 3232 directly any more.
- `ps5ctl.py` (`python3 ~/ps5lab/ps5ctl.py ...`): `status`, `lock take WHY` /
  `lock free`, `deploy TAG` (upload, verify against the decrypted ELF, swap,
  keep `eboot.before-TAG-HHMMSS.bin`), `restart` (PPSA99005 only, waits for
  HTTP READY in klog), `load MODEL k=v` (always sends a full explicit option
  set, default 4096 ctx / q4_0 KV / FA on, and reports options the server did
  not apply), `env set|clear` (vk-env.conf), `bench` (defaults: 128 tokens,
  1 round, greedy/t0.1/t0.7; appends to `~/ps5lab/results.jsonl` with build,
  env, model and load options), `stats`, `run EXPERIMENT.json` (deploy/env →
  one restart → per model: load, 16-token canary check, bench → optional restore).
- Lock: `~/ps5lab/lock` (JSON who/why/since). Commands that change the PS5
  refuse while another user holds it; one-shot commands release their own
  lock. Set `PS5LAB_USER` (default `claude`). Other clients (e.g. the
  `benchmark_all_models.py` script) should check/take it too.
- API key copied to `~/ps5lab/.key` (0600). The key is also hardcoded in
  `http_server.cpp` (`kApiKey`); it must move to a config file before any
  public repo.

**Requested q4_0 KV but log showed F16**: llama.cpp (`552f18f91`
`llama_init_from_model`) never silently downgrades a quantized KV cache; it
either applies it or fails (e.g. quantized V without flash attention). A
successful load that logged F16 therefore got F16 from the server, via one of:
(1) load options were inherited from the previously loaded model
(`options_from_state`), so a load without `cache_type_*` after an F16 model
(Gemma 4 was F16) got F16; (2) the lines belonged to the rollback reload of the
previous model after a failed load (the Muse "4,096 cells F16, 640 MiB each" =
Gemma 4's settings at the time); (3) the startup context hardcoded F16. Gemma
3 27B in my profiling run logged `K (q4_0), V (q4_0)` with FA on as requested.

**Build `out/kv-timing-20261009`** (sha256 `8f6c2711…`), built, NOT deployed
(the PS5 was in use at 21:31):
- Every context logs `KV CACHE <LOAD|ROLLBACK|STARTUP> <model> K= V= CTX= FA=`;
  failures log `LOAD FAILED READING MODEL` or `FAILED CREATING CONTEXT` with
  the attempted options.
- Load requests no longer inherit `cache_type_k/v` and `flash_attention`
  from the previous model (omitted = f16/f16/auto). Other options still carry over.
- `inference.conf` accepts `cache_type_k`, `cache_type_v` (f16/q8_0/q4_0),
  `flash_attention` (-1/0/1) for the startup model; not added to the
  deployed config, so startup behaviour is unchanged.
- `STREAM DONE ... TOK/S DECODE x SAMPLE y SEND z MS/TOK` (per-token split;
  the old prefix is unchanged so existing parsers still match) and
  `PROMPT EVALUATED n TOKENS IN m MS`.
- llama.cpp DEBUG-level log lines are no longer sent to klog.
- Note: the server clears the KV cache for every request
  (`llama_memory_clear`), so there is no prefix reuse yet.

### 2026-10-09 21:40 IST: kv-timing build deployed; KV issue confirmed; first per-token split (Claude)

Done with ps5ctl (lock taken and released; deploy 18 s; restart 101 s).
- Deployed `kv-timing-20261009` (rollback `eboot.before-kv-timing-20261009-213331.bin`).
- Qwen3-30B-A3B loaded on the fresh title in 113 s with every requested option
  applied: `llama_kv_cache: 108 MiB (4096 cells, 48 layers), K (q4_0) 54 MiB,
  V (q4_0) 54 MiB`, `flash_attn = enabled`, and
  `KV CACHE LOAD Qwen3-30B-A3B-UD-IQ2_XXS.gguf K=q4_0 V=q4_0 CTX=4096 FA=enabled`.
- The startup Gemma 3 12B logs `K (f16) 640 MiB, V (f16) 640 MiB` at 4,096
  cells, exactly the "F16, 640 MiB each" seen during the Muse attempts, which
  confirms those lines belonged to the startup/rollback model, not Muse.
- `/v1/stats` "driver usage" does not reset when the title restarts: 15.6 GB
  (after the benchmark run), 17.5 GB (after an intervening restart), 19.8 GB on
  a fresh title with only Gemma 3 12B (7.3 GB) loaded, 21.1 GB with Qwen3-30B.
  Budget is stuck at 8.59 GB (8 GiB) and free at 0. Either system memory from
  killed titles is not fully returned, or the counter behind Codex's stats
  patch never resets; the 30B still loaded, so treat the number as unreliable
  until explained. Qwen3-30B's earlier HTTP 503 (user run, ~8 swaps after a
  restart) is still best explained by memory not reclaimed across swaps.
- First per-token split (Qwen3-30B-A3B, 128 tokens, LRU prompt):

  | Variant | tok/s | ms/tok | decode | sample | send |
  |---|---:|---:|---:|---:|---:|
  | greedy | 85.3 | 11.72 | 8.82 | 2.05 | 0.02 |
  | temp 0.1 | 80.8 | 12.38 | 8.85 | 2.61 | 0.02 |
  | temp 0.7 | 73.8 | 13.55 | 8.89 | 3.69 | 0.02 |

  temp 0.1 was 38 tok/s before the exp-underflow fix. "sample" includes
  `llama_get_logits` waiting for the GPU to finish the token, so 2.05 ms on
  greedy is part GPU wait, part building the 151K-entry candidate array; to
  separate them, time `llama_synchronize` inside the decode timer.
- Restored `gemma4_12b_abliterated_Q4_K_M.gguf` with its exact prior options
  (`/health` identical); lock free.

### 2026-10-09 22:10 IST: where per-token time goes (runs A–D) (Claude)

Build `timing2-20261009` adds to `STREAM DONE`: `DECODE` (CPU time inside
`llama_decode`: graph build, command recording, PS5 submits), `WAIT`
(`llama_synchronize`), `LOGITS` (only with `PS5_PROBE_LOGITS`: one read of the
logits into CPU memory), `SAMPLE`, `SEND`. Also `PS5_OUTPUT_CPU_BUFFER`
(llama.cpp patch `vk/llama-patches/llama-context-output-cpu-buffer.patch`)
keeps llama's output buffer in CPU memory instead of the Vulkan host buffer.

| Run | Model | variant | ms/tok | decode | wait | logits | sample |
|---|---|---|---:|---:|---:|---:|---:|
| A | Qwen3-4B | greedy | 10.43 | 7.07 | 2.55 | 0.05 | 0.33 |
| A | Qwen3-4B | t0.1 / t0.7 | 11.60 / 12.52 | 7.06 | 2.58 | 0.05 | 0.99 / 2.06 |
| A | Qwen3-30B-A3B | greedy | 11.72 | 8.82 | 1.63 | 0.21 | 0.33 |
| B (CPU output buffer) | Qwen3-4B | greedy | 10.43 | 9.70 | 0.05 | 0.05 | 0.33 |
| B | Qwen3-30B-A3B | greedy | 11.47 | 10.46 | 0.03 | 0.04 | 0.33 |

- Logits reads from the Vulkan host buffer are cheap (0.05 ms): the earlier
  "2 ms sample" was mostly waiting for the GPU. The CPU output buffer changes
  nothing measurable (the wait just moves inside `llama_decode`); not adopted.
- CPU sampling at temp 0.7 costs ~1.7 ms/token over greedy at 151K vocab
  (full-vocabulary softmax); backend (GPU) sampling or a default top-k would remove it.
- **GPU profile (perf logger now works).** The PS5_Vulkan klog capture
  redirects the C `stderr` stream but not descriptor 2, and libc++'s
  `std::cerr` (used by ggml-vulkan's perf logger) kept the original, so it was
  lost. `gguf_vulkan.cpp` now points `std::cerr` at a klog streambuf.
  `ps5lab/perfparse.py KLOG START END` summarises a block (32 tokens).
  With the profiler on, GPU busy time ≈ normal wall time: Qwen3-4B 10.56 vs
  10.43 ms, Gemma 3 27B IQ2_XXS 47.4 vs 47.7 ms. **Decode is GPU-bound**; the
  ~7 ms of CPU recording/submitting per token (and the 1.8 ms per AGC submit)
  overlaps GPU work, so fewer submits would not help. (Run D, 20 nodes per
  submit, was confounded: the runner merged the previous run's env, so the perf
  logger was still on; the runner now replaces the env.)
- Qwen3-4B GPU time: q4_K matvec 5.58 ms (316 GB/s), q6_K 1.99 ms (365 GB/s),
  everything else ~3 ms (flash attention 0.72, RMS norms 1.03, SET_ROWS 0.54,
  small f32 matmuls 0.53, ~650 dispatches/token with many 4–9 µs ops).
- Gemma 3 27B IQ2_XXS GPU time 47.4 ms: **iq2_xxs matvec 33.0 ms (69.5%) at
  180 GB/s**, the q5_K output layer 2.4 ms at 402 GB/s, iq3_xxs 173 GB/s,
  q2_K 230, iq2_s 222. If iq2_xxs reached ~350 GB/s the model would go from
  ~21 to ~32 tok/s.
- ggml-vulkan on this GPU (AMD, RDNA2, not GCN) uses 32-thread workgroups and
  4 rows per workgroup for IQ types (2 for K-quants); each workgroup also copies
  the 2 KB IQ2 grid into shared memory first. Build `dmmv-tune-20261009` adds
  env overrides `GGML_VK_RM_KQ`, `GGML_VK_RM_IQ`, `GGML_VK_DMMV_WG=large`
  (patch `vk/llama-patches/ggml-vulkan-dmmv-tuning.patch`) for a sweep.

### 2026-10-09 22:35 IST: matvec launch-config sweep: defaults are best (Claude)

Build `dmmv-tune-20261009` deployed (all earlier changes plus the env
overrides; with no overrides it behaves like upstream). One restart per
config via `ps5ctl run experiments/S*.json`; greedy, 128 tokens, 2 rounds;
startup Gemma 3 12B (q4_K/q6_K) and Gemma 3 27B UD-IQ2_XXS, second-round tok/s:

| Config | Gemma 3 12B | Gemma 3 27B | greedy text vs S0 |
|---|---:|---:|---|
| S0 default (IQ 4 rows, K 2 rows, 32-thread WG) | 39.3 | 20.8 | — |
| S1 `RM_KQ=4 RM_IQ=8` | 39.1 | 20.8 | identical |
| S2 `RM_KQ=1 RM_IQ=2` | 36.0 | 19.4 | identical |
| S3 `DMMV_WG=large` (128 threads) | 33.5 | 18.3 | 27B differs (reduction order) |
| S4 S1 + large | 35.0 | 18.7 | 27B differs |

The IQ2_XXS shortfall (180 GB/s vs 316–402 for K-quants) is inside
`mul_mat_vec_iq2_xxs.comp` (byte loads of `qs`, per-element sign/grid work),
not in launch configuration. Next step there is a shader rewrite, which first
needs an on-console op correctness test (test-backend-ops as a title).
First-round 27B timings run 1–3 ms/token slower than the second (warm-up), so
compare second rounds. `ps5ctl` now records a hash of each response's text.
PS5 restored to `gemma4_12b_abliterated_Q4_K_M.gguf` (q4_0 KV, FA on, 4096), env
empty, lock free.

### 2026-10-09 22:56 IST: streaming load progress on /v1/models/load (Claude)

Deployed `load-stream-20261009` (rollback `eboot.before-load-stream-20261009-225200.bin`).
`POST /v1/models/load` with `"stream": true` answers `text/event-stream`:
`{"status":"unloading","model":PREV}`, `{"status":"loading","model","progress":0..100,"elapsed_s"}`
(each whole percent llama.cpp reports; some values are skipped),
`{"status":"creating_context"}`, on failure `{"status":"restoring_previous",...}`
with its own progress, then the final event = the usual JSON body plus
`"http_status"` and `"elapsed_s"`, then `[DONE]`. Already loaded → only the
final event. Validation errors (422/404) stay plain HTTP JSON. A client that
disconnects stops the events, not the load. Without `stream` nothing changed.
No CORS headers are sent (browser pages on another origin cannot call it
directly). Tested: streamed restore of Huihui-Qwen3.8-27B-abliterated
(28,672 ctx, 2,048 out, q4_0, FA): 90 progress events, loaded in 92 s,
`/health` identical to before; already_loaded and non-stream paths checked.

### 2026-10-10 00:05 IST: image generation (stable-diffusion.cpp) built, not yet deployed (Claude)

- ProsperoAI (local clone `prosperoai-audit/src/sd_runtime_ps5.cpp`) already
  runs stable-diffusion.cpp on the PS5: prebuilt `libstable-diffusion.a`
  linked with the same `libggml` as `libllama`, SD-Turbo FP16 (2.4 GiB,
  512×512, 1 step, CFG 1, CUDA RNG, EPS, no mmap), UNet and VAE on their own
  AGC ggml backend `PS5AGC0`, text encoder on CPU. Their published time: 76 s
  per image (34.9 s diffusion, 31.2 s VAE) on firmware 6.02. Their model repo
  `blackbearreloaded/ProsperoAI-SD-Turbo-FP16-PS5` is the upstream
  `stabilityai/sd-turbo` (rev `b261bac`) diffusers FP16 files, renamed, with
  SHA-256s recorded; licence: Stability AI Community License.
- Our build: stable-diffusion.cpp `f89d9b1` (cloned to Aisha
  `~/ps5lm-inference/third_party/sdcpp`, no submodules) as a static library
  against our llama.cpp ggml via `vk/sd/CMakeLists.txt` (imported `ggml`,
  `SD_USE_UPSTREAM_GGML`, `SD_USE_SYSTEM_GGML`, examples/WebP/WebM off),
  build dir `~/sd-vk/build`, ~6 min at -j2.
- **stable-diffusion.cpp requires `GGML_MAX_NAME >= 160`.** The llama.cpp
  cross-build (`~/llama-vk/build`) was reconfigured with
  `-DGGML_MAX_NAME=160` in C/CXX flags and rebuilt (294 steps, 8 min); the
  title compiles with the same define. The previous (64) libraries are kept in
  `~/llama-vk/libs-maxname64-20261009/` (same layout) for rebuilding older titles.
- Link: two weak undefined `_ZTH…sd::parallel_detail::{active,executor}`
  (C++ thread_local init hooks; NULL by ABI design means no dynamic init) are
  now allowed by the weak-null step in `build-gguf-vk-title.sh` (`_ZTH` prefix).
- New `vk/sd_service.{h,cpp}` and `POST /v1/images/generations`:
  `prompt` (required), `model` (folder or file in the app dir, default
  `sd-turbo-fp16`), `size` "WxH" or `width`/`height` (multiples of 64,
  256–1024, ≤1024×1024), `steps` (default 1), `cfg_scale` (default 1.0),
  `seed` (-1 random), `negative_prompt`, `sample_method`, `prediction`,
  `stream`. Returns `{"created","model","seed","width","height","steps",
  "timings":{"load_s","generate_s"},"data":[{"b64_json": PNG}]}`; with
  `stream` the SSE events are `unloading`, `loading` (percent), `generating`
  (`step`/`steps`), then that JSON plus `http_status`. Backend `Vulkan0`,
  4 CPU threads, VAE tiling above 768×768. An image request unloads the LLM;
  `/v1/models/load` unloads the image model; a failed LLM load with no
  previous LLM now fails cleanly (503) instead of rolling back to nothing.
  `/health` adds `image_model` and returns 200 if either kind is loaded.
- Built `out/sd-20261010` (112 MB, sha256 `9b168dd9…`). Model download:
  `~/models/sd/fetch-sd-turbo.sh` (SHA-verified per file); staging job
  `~/models/sd/stage-sd-turbo.py` uploads to `/data/homebrew/PPSA99005/sd-turbo-fp16/`
  with read-back SHA checks (log `~/models/sd/stage.log`). Not deployed yet:
  the PS5 was serving requests at 00:03.

### 2026-10-10 00:12 IST: image generation deployed and working: 0.93 s per 512² SD-Turbo image (Claude)

- Deployed `out/sd-20261010` (rollback `eboot.before-sd-20261010-000520.bin`).
  SD-Turbo FP16 staged at `/data/homebrew/PPSA99005/sd-turbo-fp16/` (three
  files, read-back SHA-256 verified).
- LLM regression with `GGML_MAX_NAME=160`: startup Gemma 3 12B greedy output
  byte-identical to the previous build (text hash `d8c2a75ebfb3`), 38.7 vs
  39.3 tok/s.
- First image (cold, stream): unload LLM, model load 6.0 s, generate 3.49 s
  (first-use weight loading + text encoding 2.03 s, 1-step sampling 0.39 s,
  VAE 1.07 s). Warm: **1 step 0.93 s** (sampling 0.24, VAE 0.65), **4 steps
  1.62 s**. ProsperoAI's AGC path: 76 s (34.9 diffusion, 31.2 VAE), so this is
  ~80× faster. Prediction type auto-detected correctly (no `prediction` needed).
  Images saved in `ps5lm-tests/images/`.
- Reverse swap tested: streaming `/v1/models/load` of the abliterated Gemma 4
  unloaded the image model and restored the exact prior options.
- Built, not yet deployed, `out/sd2-20261010`: SSE image phases are now
  `preparing` (first-use weight loading/text encoding), `generating`
  (sampling `step`/`steps`), `decoding` (VAE), taken from stable-diffusion.cpp's
  log lines; and no empty `unloading` event when no LLM was loaded.

### 2026-10-10 00:58 IST: Z-Image-Turbo (6B) generating on the PS5 (Claude)

- Model choice (supported by stable-diffusion.cpp f89d9b1, fits 12 GiB, ungated):
  Tongyi-MAI Z-Image-Turbo (Apache-2.0, Nov 2025, 623K downloads/30 d).
  Files (pinned revisions, SHA-256 verified), in `/data/homebrew/PPSA99005/z-image-turbo/`:
  `z_image_turbo-Q8_0.gguf` (leejet/Z-Image-Turbo-GGUF@c61c0e4, 6.58 GB),
  `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (unsloth@a06e946, 2.50 GB),
  `ae.safetensors` (Comfy-Org/z_image_turbo@6fc90a3, Flux VAE, 0.34 GB), and
  `sd-model.conf` (repo copy `vk/sd/models/z-image-turbo.sd-model.conf`:
  components, `diffusion_flash_attn=1`, defaults 8 steps / CFG 1.0 / 1024²).
  Rejected: Qwen-Image-2.1 (too large), Krea-2-Turbo / FLUX.2-klein-9B /
  Ideogram 4 (gated), Iris-3B (days old). Runner-up: FLUX.2-klein-4B.
- Downloads: Aisha Wi-Fi ~2.4 MB/s; Mac→Aisha and Mac→PS5 also go through
  Aisha's Wi-Fi (0.3–1.2 MB/s). With Aisha's Ethernet on the router
  (user ran `sudo nmcli con up "Wired connection 1" ifname enp0s31f6`; the
  static multipath default route still prefers Wi-Fi, so aria2 must use
  `--interface=enp0s31f6`) it ran at 25 MB/s. Switch back for the PS5 with
  `sudo nmcli con up direct-ps5`.
- Server: image model folders may hold `sd-model.conf` (component files,
  options, per-model defaults); omitted request fields use them. Responses
  report the width/height/steps/cfg actually used.
- **First attempt failed: "model manager cannot make enough memory available
  ... reported free 616.94 MB / total 12288 MB".** ggml-vulkan derives free
  memory from VK_EXT_memory_budget, and the PS5 RADV numbers are wrong (the
  same ever-growing counter as `/v1/stats`). New opt-in switch
  `GGML_VK_IGNORE_MEMORY_BUDGET` (patch `vk/llama-patches/ggml-vulkan-ignore-memory-budget.patch`)
  reports the whole heap so stable-diffusion.cpp uses its own accounting.
  Now set in the PS5's `vk-env.conf`. Deployed build `sd4-20261010`
  (rollback `eboot.before-sd4-20261010-004956.bin`).
- Results (Z-Image-Turbo Q8_0, 8 steps, warm): 512² 18.2 s (sampling 16.6,
  VAE 1.5), 768² 42.1 s (37.9 / 4.0), 1024² 79.7 s (71.8 / 7.9). First image
  after load at 1024²: 109.5 s (text encoding 7.8 s, sampling 93.2 s with
  first-use weight loading, VAE 8.4 s). Total weights 8.8 GB on Vulkan.
  Quality: prompt text rendered exactly ("PS5 RAMEN" sign); images in
  `ps5lm-tests/images/`.
- PS5 restored to `gemma4_12b_abliterated_Q4_K_M.gguf` (identical `/health`
  parameters), 36.5 tok/s greedy afterwards; lock free.
- Next for speed: per-op GPU profile of a Z-Image step (perf logger works),
  Q6_K/Q4_K DiT variants (less bandwidth), fewer steps, and the
  segment-streaming behaviour of the model manager.

### 2026-10-10 02:56 IST: image model comparison on the PS5 (Claude)

Same three prompts, seed 42; 768×768 (SD-Turbo 512²), model defaults; second
and third prompts are warm (first includes first-use weight loading).
Images: `ps5lm-tests/images/compare-20261010/`.

| Model | Steps | Warm gen (s) | Files / notes |
|---|---:|---:|---|
| SD-Turbo FP16 | 1 | 0.9 | sign text garbled |
| Z-Image-Turbo Q8_0 | 8 | 42.0 | sign correct |
| Z-Image-Turbo Q4_K | 8 | 43.7 | no faster than Q8_0: compute-bound at 768² |
| Z-Image-Turbo + abliterated Qwen3-4B encoder | 8 | 42.0 | `BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1` Q4_K_M |
| FLUX.2-klein-4B Q8_0 | 4 | **17.6** | sign correct; astronaut prompt drew two astronauts, no robot |
| ERNIE-Image-Turbo Q6_K | 8 | 66.1 | sign correct (neon), most stylised |

Model folders on the PS5 (`/data/homebrew/PPSA99005/`): `flux2-klein-4b`,
`ernie-image-turbo` (VAE from flux2-klein-4b), `z-image-turbo-q4k` and
`z-image-turbo-abliterated` (diffusion model / VAE from z-image-turbo),
each with `sd-model.conf` (copies in `vk/sd/models/`). Skipped: Iris-3B (12 GB
unquantized), HiDream-O1 (FP8 → 16 GB in our ggml), ERNIE abliterated (gated).
- "Abliterated image models" are abliterated **text encoders**; diffusion
  models have no refusal behaviour to remove.
- Network lessons: Ethernet-bound downloads die when the port is switched
  back to `direct-ps5`; after the cable returns the PS5 may need
  Settings → Network → Test Internet Connection before it answers ARP.
- klogd fix: it now reconnects after 2 minutes of silence (a dead connection
  from an offline PS5 previously left it waiting forever).

### Image generation on the PS5: current setup and how to use it (summary, 2026-10-10)

Consolidates the four dated image entries above; read those for the history.

**Stack.** stable-diffusion.cpp (`f89d9b1`) is linked into the same PS5 title as
llama.cpp, against the same ggml and ggml-vulkan (RADV) build. Requirements
that follow from that:
- ggml, llama.cpp and the title are built with `GGML_MAX_NAME=160`
  (stable-diffusion.cpp's minimum). The 64-name libraries are kept in
  `~/llama-vk/libs-maxname64-20261009/`.
- stable-diffusion.cpp is built by `vk/sd/CMakeLists.txt` (upstream-ggml mode,
  imported ggml) into Aisha `~/sd-vk/build`; `build-gguf-vk-title.sh` needs
  `SD_SOURCE` and `SD_BUILD` in addition to the llama variables, and allows
  `_ZTH*` weak symbols (thread_local init hooks).
- `GGML_VK_IGNORE_MEMORY_BUDGET=1` must be set in the PS5's `vk-env.conf`:
  the RADV memory-budget numbers are wrong on the PS5 and otherwise
  stable-diffusion.cpp believes the GPU is full.
- Deployed build: `sd4-20261010`.

**API.** `POST /v1/images/generations` (Bearer key), OpenAI-style:
`{"model": "<folder>", "prompt": "...", "negative_prompt", "size": "WxH"
or "width"/"height" (multiples of 64, 256–1024), "steps", "cfg_scale",
"seed" (-1 random), "sample_method", "prediction", "stream": true|false}`.
Omitted fields use the model folder's defaults. Response: `{"model","seed",
"width","height","steps","cfg_scale","timings":{"load_s","generate_s"},
"data":[{"b64_json": PNG}]}`. With `stream`, SSE events `unloading`,
`loading` (progress %), `preparing`, `generating` (step/steps), `decoding`,
then that JSON plus `http_status`. (`preparing`/`decoding` labels come from
build `sd2`+; all later builds include them.)

**Memory rules.** An image request unloads the LLM; `/v1/models/load` unloads
the image model. The image model stays loaded between image requests.
`/health` reports `image_model` and is 200 when either kind is loaded.

**Model folders** (in `/data/homebrew/PPSA99005/`, confs mirrored in
`vk/sd/models/`). A folder is either a diffusers layout (SD-Turbo) or holds
`sd-model.conf` with `diffusion_model`, `llm`, `vae`, `clip_l`, `clip_g`,
`t5xxl`, `model` (paths relative to the folder; `../other/file` allowed),
`diffusion_flash_attn`, `prediction`, and defaults `width`, `height`,
`steps`, `cfg_scale`, `sample_method`.

| Folder | Model | Default | Warm time |
|---|---|---|---|
| `sd-turbo-fp16` | SD-Turbo FP16 (2.6 GB) | 512², 1 step | 0.9 s |
| `z-image-turbo` | Z-Image-Turbo Q8_0 + Qwen3-4B-Instruct Q4_K_M + Flux VAE (9.4 GB) | 1024², 8 steps | 18 s @512², 42 s @768², 80 s @1024² |
| `z-image-turbo-q4k` | same, Q4_K DiT | 1024², 8 | 44 s @768² (no gain) |
| `z-image-turbo-abliterated` | Z-Image Q8_0 + abliterated Qwen3-4B encoder | 1024², 8 | 42 s @768² |
| `flux2-klein-4b` | FLUX.2-klein-4B Q8_0 + Qwen3-4B Q4_K_M + FLUX.2 VAE | 1024², 4 | 17.6 s @768² |
| `ernie-image-turbo` | ERNIE-Image-Turbo Q6_K + Ministral-3-3B Q4_K_M | 1024², 8 | 66 s @768² |

Recommended: `flux2-klein-4b` for everyday use, `z-image-turbo` for best
photorealism, `sd-turbo-fp16` for instant drafts.

**Adding a model.** Check `~/ps5lm-inference/third_party/sdcpp/docs/` for the
architecture's components; download pinned, SHA-verified files on Aisha
(Ethernet on the router with `--interface=enp0s31f6` is ~10× faster than
Wi-Fi); write `sd-model.conf`; stage with `~/ps5lab/stage_folders.py FOLDER`
(read-back SHA check); no rebuild needed. Unquantized FP8/BF16 files are
expanded to F16 at load, so check the size fits 12 GiB together with the text
encoder. Tools: `~/ps5lab/compare_images.py OUTDIR SIZE MODEL...` for
side-by-side runs.

**Open items.** Per-op GPU profile of a Z-Image step (sampling is
compute-bound: Q4_K is no faster than Q8_0); memory not fully reclaimed across
many LLM swaps (restart before the biggest models); CORS for browser frontends.

### 2026-10-10 03:12 IST: standard vs abliterated Z-Image text encoder (Claude)

Same seed 42, 768², 8 steps; three dark prompts (gory zombie, bloody medieval
battlefield, blood-dripping ritual dagger). Images:
`ps5lm-tests/images/encoder-ab/`. Both encoders rendered the gore and blood as
asked; compositions stay closely related (same diffusion model and seed). The
standard encoder did not soften these prompts: in the battle and zombie pairs
it produced slightly *more* visible blood than the abliterated one; the ritual
pair is nearly identical. Conclusion: for Z-Image-Turbo the base Qwen3-4B
encoder does not refuse or dilute violent/horror prompts, so the abliterated
encoder adds little; what the model can depict is set by the diffusion model's
training data. Speed identical (42 s warm). The user was given the A/B command
to test other content themselves.

### 2026-10-10 03:30 IST: speech-to-text (whisper.cpp) working; TTS and video built (Claude)

Build `av-20261010` (deployed; rollback `eboot.before-av-20261010-032557.bin`)
adds three engines to the same title and ggml/Vulkan stack:
- **STT**: whisper.cpp `d1be6fd` (Aisha `~/ps5lm-inference/third_party/whispercpp`),
  `src/whisper.cpp` + `src/aneforge/whisper-aneforge.cpp` compiled by the
  title build (`WHISPER_SOURCE`); compiles unchanged against our ggml.
- **TTS**: llama.cpp mtmd audio generation (the `tools/tts` flow), with
  `LLAMA_BUILD_MTMD=ON` in `~/llama-vk/build` → `tools/mtmd/libmtmd.a`; needs
  `vendor/hash/hash.cpp` + `vendor/hash/sha256/sha256.c` linked.
- **Video**: stable-diffusion.cpp `generate_video` (Wan), via `sd_service`.
- New `vk/audio_service.{h,cpp}`; endpoints:
  - `POST /v1/audio/transcriptions` `{"audio_b64": WAV (any rate/channels,
    PCM 8–32-bit or float), "model" (default
    `whisper/ggml-large-v3-turbo-q5_0.bin`), "language", "translate"}` →
    `{"text","language","segments":[{start,end,text}],"timings"}`.
  - `POST /v1/audio/speech` `{"input", "model" (folder with backbone +
    `mmproj-*.gguf`, default `qwen3-tts-1.7b`), "language", "speaker_b64"
    (reference WAV), "seed", "max_frames", "response_format": json|wav}`.
  - `POST /v1/videos/generations` like images, plus `frames` (4k+1), size in
    multiples of 16, returns `data:[{"b64_png"}]` + `fps`; SSE with `stream`.
  - One big engine at a time: `free_gpu_for()` unloads the LLM, image/video,
    STT and TTS models as needed; `/v1/models/load` unloads them all; `/health`
    reports `image_model`, `stt_model`, `tts_model`.
  - Request body limit raised from 32 KB to 64 MB (audio arrives base64).
- STT results (8.0 s `say`-generated WAV): Whisper tiny.en load 0.96 s,
  transcribe 0.21 s (minor errors); **large-v3-turbo q5_0 load 6.4 s,
  transcribe 0.79 s (10× real time), word-perfect**. Both on `Vulkan0`.
- Models: `/data/homebrew/PPSA99005/whisper/` (pinned, SHA-verified, from
  `ggerganov/whisper.cpp`). Downloads stage file by file to the PS5 and drop
  the Aisha copy (Aisha disk is ~98% full; the USB archive drive dropped off
  with I/O errors and needs replugging/checking).

### 2026-10-10 03:44 IST: text-to-speech (Qwen3-TTS 1.7B) working (Claude)

Model folder `/data/homebrew/PPSA99005/qwen3-tts-1.7b/`:
`Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` (1.04 GB) +
`mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` (0.45 GB) from
`ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF` (pinned, SHA-verified). Load 15.5 s
(mtmd projector compute buffer 875 MiB on Vulkan0). Same sentence, seed 7:
default voice 7.28 s of 24 kHz audio in 3.92 s; cloned from a 4 s reference WAV
(`speaker_b64`) 6.96 s in 3.54 s (≈2× real time). Round trip through
Whisper large-v3-turbo on the PS5 returned the exact sentence for both. Clips
in `ps5lm-tests/audio/`. Test script: `~/ps5lab/tts_test.py`.

### 2026-10-10 05:19 IST: text-to-video (Wan2.1 T2V 1.3B) working (Claude)

Folder `/data/homebrew/PPSA99005/wan2.1-t2v-1.3b/` (pinned, SHA-verified):
`wan2.1_t2v_1.3B_fp16.safetensors` (2.84 GB, Comfy-Org repack),
`umt5-xxl-encoder-Q4_K_M.gguf` (3.66 GB, city96), `wan_2.1_vae.safetensors`
(0.25 GB), `sd-model.conf` (mode=video, euler, 20 steps, CFG 6, flow shift 3,
flash attention, default 416×240, 9 frames, 16 fps; repo copy in
`vk/sd/models/`). Total weights 6.43 GB on Vulkan.
- First attempt failed only because `sd_service_load` required image
  support; now accepts image *or* video models (build `av2-20261010`, deployed,
  rollback `eboot.before-av2-20261010-051553.bin`).
- Smoke test (fox in snow, seed 42): 9 frames 416×240 in 59.6 s: first-use
  weight loading + umt5 text encoding ~18 s, sampling 36.3 s (1.8 s/step),
  VAE decode 4.3 s (tiled). Recognisable motion; MP4 assembled on the Mac with
  ffmpeg: `ps5lm-tests/video/`. Test script `~/ps5lab/video_test.py '{json overrides}'`.
- Known nit: SSE phase labels during video report weight loading as
  `generating` (the log lines differ from images); cosmetic.
- Next sizes to try: 832×480 / 33 frames (the reference) will be ~(4×
  pixels × 4× frames) heavier per step; watch VAE memory and the watchdog.

### 2026-10-10 11:30 IST: archive drive back, media backup running, image-to-video built (Claude)

- **Archive drive.** `PS5_STORAGE` (UUID `6FDF-D05B`) is an NVMe/SATA disk
  behind a JMicron JMS583 USB bridge. It dropped off Aisha overnight (both old
  mount points `/mnt/ps5-storage` and `~/ps5-storage-250gb` are stale and
  return I/O errors; clearing them needs `sudo umount -l`). Through the passive
  USB hub it does not enumerate; plugged directly into Aisha it auto-mounts at
  **`/media/user/PS5_STORAGE`** (use this path now). Reads at ~70 MB/s.
- **Backup job** `~/models/backup-media.sh` (log `~/models/backup-media.log`):
  re-verifies `PS5-Language-Models/` against its manifest, then copies all
  media models to `PS5_STORAGE/PS5-Media-Models/` with SHA-256 checks: image
  models and Whisper from Aisha (`~/models/sd`, `~/models/av`), Qwen3-TTS and
  Wan2.1 from the PS5 over FTP (their Aisha copies were removed after staging).
  The user approved clearing Aisha's `~/models/sd` duplicates once verified
  (Aisha's disk is ~98% full).
- **PS5 memory.** 16 GB GDDR6 total, shared by CPU and GPU (no separate
  system RAM). Our title gets a 12 GiB Vulkan heap plus ~2.3 GB CPU-side, so
  ~14.3 GB. Models larger than that can only run by streaming weights from
  the SSD (stable-diffusion.cpp's model manager does this per segment); our
  file reads run at ~80–100 MB/s, far below the SSD's rating, which is the
  thing to fix for big models (and for LLM load times).
- **Video candidates assessed.** MiniMax-H3 (video + stereo audio, ~25.6 GB
  minimum: 13.1 GB Qwen3-VL-32B encoder, 6.7 GB pruned Q2 DiT, 5.2 GB VAE):
  only possible by SSD streaming, slow. Wan2.2 T2V A14B (two 9.65 GB Q4
  experts, used one after the other): fits one at a time. LTX-2.x: same
  size problem as MiniMax. Chosen next: **HunyuanVideo 1.5 I2V 480p
  CFG-distilled Q4_K_M** (`jayn7/HunyuanVideo-1.5_I2V_480p-GGUF`, 5.09 GB) +
  Qwen2.5-VL-7B Q4_K_M + byT5 GlyphXL + SigCLIP vision + VAE (13.6 GB total),
  and **Wan2.2 TI2V 5B Q8_0** + its VAE (6.8 GB; reuses the umt5 encoder).
  Pinned list `~/models/video/video.list` (20.4 GB), downloader
  `~/models/video/fetch-video.sh` (waits for Ethernet on the router). Configs
  in `vk/sd/models/` (`hunyuanvideo-1.5-i2v-480p`, `wan2.2-ti2v-5b`).
- **Build `i2v-20261010`** (built, not deployed): `/v1/videos/generations`
  accepts `init_image_b64` (PNG/JPEG first frame; size follows the image when
  none is given), and `sd-model.conf` accepts `clip_vision`.

### 2026-10-10 12:05 IST: media backup verified; Aisha space freed (Claude)

`PS5_STORAGE/PS5-Language-Models/` re-verified: all 12 GGUFs match the
manifest. `PS5_STORAGE/PS5-Media-Models/` now holds every media model (37
files, 41 GB; per-file SHA-256 checks, log `~/models/backup-media.log`,
download lists copied alongside): `sd-turbo-fp16`, `z-image-turbo`,
`z-image-turbo-q4k`, `z-image-turbo-abliterated`, `flux2-klein-4b`,
`ernie-image-turbo`, `whisper`, `qwen3-tts-1.7b`, `wan2.1-t2v-1.3b`. With the
user's approval the 13 duplicated weight files in Aisha `~/models/sd` were
removed after confirming each on the drive (size + checksum); configs kept.
Aisha: 3.8 GB → 36 GB free. Every media model is now on the PS5 and on the
drive. The video batch downloader is armed and waits for Ethernet.

### 2026-10-10 13:06 IST: image-to-video working (HunyuanVideo 1.5, Wan2.2 5B); disk-backed weights (Claude)

- Downloaded over Ethernet (20.4 GB, 16 min, pinned + SHA-verified) and staged
  with read-back checks: `/data/homebrew/PPSA99005/hunyuanvideo-1.5-i2v-480p/`
  (CFG-distilled I2V Q4_K_M 5.09 GB, Qwen2.5-VL-7B Q4_K_M, byT5 GlyphXL,
  SigCLIP vision, VAE fp16) and `.../wan2.2-ti2v-5b/` (Q8_0 5.4 GB + its VAE;
  umt5 shared from `wan2.1-t2v-1.3b`). Aisha copies in `~/models/video/`.
- **Key setting: `params_backend`.** With all weights on the GPU
  (`Vulkan0`), HunyuanVideo (11.7 GB of weights) sampled fine but failed at
  VAE decode: 9.3 GB of encoder + DiT weights stayed resident and
  stable-diffusion.cpp does not evict GPU-resident parameters. Its
  `--params-backend` accepts per-module placement including **`disk`**
  (weights loaded for their stage, then released). `sd-model.conf` now takes
  `params_backend` (default `Vulkan0`) and `max_vram`; the video configs use
  `params_backend=all=Vulkan0,te=disk,clip_vision=disk,vae=disk` (Wan:
  `te=disk,vae=disk`). This is also the route for models larger than the heap
  (MiniMax-H3, LTX): put more modules on disk.
  Note: `sd_service_load` skips reloading when the same model name is already
  loaded, so a changed `sd-model.conf` needs another model loaded in between
  (or a restart).
- Results, fox start image from FLUX.2-klein (832×448, 31 s), prompt "the fox
  looks around, then walks forward…", 416×224, 17 frames, 20 steps, seed 42:

  | Model | Total | Encode | Sampling | VAE | Quality at this size |
  |---|---:|---:|---:|---:|---|
  | HunyuanVideo 1.5 I2V (8.3B, Q4_K_M) | 107.0 s | 15.3 s | 63.4 s (3.2 s/step) | 24.6 s | clean, keeps the start image, natural motion |
  | Wan2.2 TI2V 5B (Q8_0) | 91.3 s | 19.0 s | 54.6 s (2.7 s/step) | 16.6 s | heavy artifacts; this model targets 704×1280 and its VAE compresses 16×, so 416×224 is far below its range |

  Clips in `ps5lm-tests/video/`. Test script `~/ps5lab/i2v_test.py OUTDIR 'JSON' [IMAGE]`.
- Server (build `i2v2-20261010`, deployed; rollback
  `eboot.before-i2v2-20261010-125758.bin`): `/v1/videos/generations` accepts
  `init_image_b64`; `sd-model.conf` accepts `clip_vision`, `params_backend`,
  `max_vram`. Bug fixed: after a network change `accept()` failed in a tight
  loop (flooding klog, port 8081 dead while the title still ran); the listener
  is now closed and reopened with a 2 s back-off.
- Other video models assessed for the 12 GiB heap: LTX-2.3 22B distilled
  (audio + video; Q2_K 7.9 GB + connectors 2.3 + VAEs 1.8; text encoder is the
  Gemma 3 12B already on the PS5) plausible; Wan2.1 T2V 14B Q4_K_M 10.1 GB
  fits; Wan2.2 A14B fits one expert at a time; LingBot dense 1.3B easy;
  LTX-2.5 (≥12.1 GB DiT), LingBot MoE 30B, MiniMax-H3 need disk streaming.

### 2026-10-10 13:20 IST: voice agent (STT + small LLM + TTS loaded together) working (Claude)

Build `voice-20261010` (deployed; rollback `eboot.before-voice-20261010-131504.bin`).
- **Coexistence.** Speech models and an LLM of up to 5 GB (`kSmallLlmBytes`)
  now stay loaded together: `free_gpu_for("stt"|"tts"|"voice")` keeps a small
  LLM and both speech models and only unloads image/video models or a big
  LLM; `/v1/models/load` keeps the speech models when the new LLM file is
  ≤ 5 GB. Voice stack: Whisper large-v3-turbo + Qwen3-4B Q4_K_M + Qwen3-TTS
  1.7B ≈ 6.5 GB of the 12 GiB heap.
- **`POST /v1/voice/chat`** (always SSE): `{"audio_b64": WAV, "messages":
  optional history, "system", "language", "speaker_b64" (reference voice),
  "max_tokens" (200), "llm" (default `Qwen3-4B-Q4_K_M.gguf`), "stt_model",
  "tts_model"}`. Loads whatever part of the stack is missing, then emits
  `{"type":"transcript","text"}`, `{"type":"text","delta"}`,
  `{"type":"audio","index","text","audio_b64","audio_s","tts_s","at_s"}` once
  per sentence (the reply is spoken sentence by sentence, split at . ! ?
  followed by whitespace), and `{"type":"done","transcript","reply",
  "timings"}`. Thinking is disabled in the chat template; markdown symbols
  are stripped before TTS; without `speaker_b64` the first sentence's audio
  is reused as the reference so the voice stays the same within a reply.
- **Measured at the client** (`~/ps5lab/voice_test.py q1.wav q2.wav q3.wav`,
  questions made with macOS `say`):

  | | Turn 1 (cold) | Turn 2 | Turn 3 |
  |---|---:|---:|---:|
  | transcript | 54.4 s (53.7 s loading 3 models) | 0.68 s | 0.68 s |
  | first LLM text | 54.5 s | 0.90 s | 0.98 s |
  | **first audio** | 56.3 s | **2.45 s** | **2.65 s** |
  | reply complete | 58.4 s (6.8 s speech) | 4.4 s (6.1 s speech) | 2.7 s (3.0 s speech) |

  TTS runs at ~1.8× real time (e.g. 3.0 s of speech in 1.66 s), so after the
  first sentence audio arrives faster than it plays. Turn 2 used the history
  ("How far is it from London?" → "Paris is about 200 miles from London").
  Audio in `ps5lm-tests/audio/voice-agent/`.
- **Where the 2.5 s goes:** STT ~0.5 s, LLM first sentence ~0.3 s, TTS of the
  first sentence ~1.5–1.7 s. To cut it: split the first chunk at a comma/short
  clause, a faster TTS (Kokoro 82M or Pocket TTS, both far smaller than
  Qwen3-TTS 1.7B), or streaming TTS frames instead of whole sentences.

### 2026-10-10 14:05 IST: "memory not reclaimed across swaps" is heap fragmentation; fixed with 256 MB blocks (Claude)

- **Symptom.** From the user's frontend, a Z-Image request after voice-agent
  use failed 0.04 s into sampling: `ggml_vulkan: Device memory allocation of
  size 1045893120 failed (ErrorOutOfDeviceMemory)` → `model manager alloc
  params backend buffer failed, size = 997.44MB`. The frontend kept showing its
  last progress text (a failed stream ends with an event carrying
  `"error"` and `"http_status":500`, then `[DONE]`; clients must check it).
- **Controlled tests** (`~/ps5lab/leak_test.py CASE`, fresh restart before
  each, then Z-Image-Turbo 512²):

  | Used before the image | 1 GB blocks (default) | 256 MB blocks |
  |---|---|---|
  | nothing | OK (46.7 s) | |
  | Whisper | OK, one 1.2 GB allocation failed then retried | |
  | Qwen3-TTS | OK, one 0.87 GB allocation failed then retried | |
  | voice stack (Qwen3-4B + Whisper + TTS, two turns) | **FAILED** (1.0 GB allocation) | **OK (47.0 s)** |

- **Cause.** Fragmentation, not a leak: each Vulkan allocation on the PS5 is
  one contiguous range of direct memory, and ggml-vulkan asks for blocks up to
  `suballocation_block_size` (default 1 GiB). After models of different sizes
  have been loaded and freed, total free memory is sufficient but no 1 GiB
  range is. This also explains the earlier Qwen3-30B 503 after many LLM swaps.
- **Fix.** `GGML_VK_SUBALLOCATION_BLOCK_SIZE=268435456` in the PS5's
  `vk-env.conf` (now alongside `GGML_VK_IGNORE_MEMORY_BUDGET=1`). No speed
  change: abliterated Gemma 4 12B greedy 36.5 tok/s with identical output hash
  (`c10ea785014f`), Z-Image 47.0 s, voice first audio 2.3 s. A single large
  compute buffer (~1.2 GB) can still fail once and be retried by
  stable-diffusion.cpp. Not yet re-tested: Qwen3-30B after many swaps.

## 2026-10-10 (afternoon): voice endpoint fixes and the rebuilt web UI

**PS5 title `voicefix-20261010`** (deployed; rollback `eboot.before-voicefix-20261010-145337.bin`):

- `/v1/voice/chat` accepts `"messages": []` as "no history" (it used to return 422, which is what broke the web UI's first voice turn).
- A 422 from `/v1/voice/chat` now names its reason in the body and in klog (`BAD VOICE REQUEST: <reason>`).
- A peer that connects and closes without sending anything is dropped silently. These were the `HTTP REQUEST REJECTED` lines: health probes from the web proxy that queued behind a long generation and were abandoned. None logged since the deploy.
- The cold-start phases were already on the voice stream as `{"status":"unloading"|"loading","model":...}` events before the first `{"type":"transcript"}`; clients should show them. A cold voice turn (after image or video work) is 42–56 s, a warm one about 2.5 s to first audio.
- Build command, for the record: `PS5_VULKAN_ROOT=~/ps5-vulkan-release/ps5vk/ps5-vulkan-4271e2e309d9 LLAMA_SOURCE=~/ps5lm-inference/third_party/llama.cpp LLAMA_BUILD=~/llama-vk/build SD_SOURCE=~/ps5lm-inference/third_party/sdcpp SD_BUILD=~/sd-vk/build WHISPER_SOURCE=~/ps5lm-inference/third_party/whispercpp OUT=$PWD/out/<tag> bash ./build-gguf-vk-title.sh` in `~/ps5lm-inference/vk` on Aisha (about 45 s).

**Web UI** (on the Mac, `/Users/user/Documents/antigravity/proud-salk/ps5_chat`, served by `server.py` on :8080 and https :8443):

- Rebuilt as `index.html` + `app.css` + `js/{core,chat,voice,media}.js`: five tabs (Chat, Voice, Image, Video, Audio), bottom tab bar on phones, left rail on desktop, no Tailwind. The previous single-file page is kept as `index.legacy.html`; `server.py.before-ui-rebuild` is the old proxy.
- The page runs one PS5 job at a time and shows it in the status pill. Voice handles `error` events and the loading phases, plays sentences through Web Audio, and has a hands-free mode (stops on a pause, listens again after the reply).
- Proxy changes: `/api/health` answers from its last reading with `busy: true` while a job holds the proxy's lock (or when the PS5 accepts connections but does not answer in 5 s), adds `voice_ready`; load timeout 600 s, video 900 s; static files are sent `Cache-Control: no-cache`.
- SSE streams through the proxy must be ended by the client at `data: [DONE]`; the proxy can hold the socket open afterwards.
- Video progress events with `steps` in the hundreds are weights streaming from disk, not sampling steps.
- The microphone needs a secure context: use `https://host.example.ts.net:8443` from a phone, not the http address.

## 2026-10-10 (evening): image editing

Title `imgedit-20261010` (rollback `eboot.before-imgedit-20261010-161240.bin`). `/v1/images/generations` takes two more optional inputs:

- `ref_images_b64`: array of up to 4 base64 PNG/JPEG reference images, for models with reference editing. Verified on **FLUX.2-klein-4B**: "the same fox wearing a blue knitted scarf, in a neon-lit city street at night" keeps the subject and changes the rest, 13.8 s at 512², 4 steps (plain generation at the same size is about 7 s warm).
- `init_image_b64` + `strength` (0..1, default 0.75): classic image-to-image on any model. It runs (`strength` 0.6 = 3 of 4 steps, 6.7 s on klein) but on a 4-step distilled model it barely restyles; reference editing is the useful mode.
- Z-Image-Turbo and ERNIE have no edit mode. Qwen-Image-Edit / FLUX Kontext would be new downloads.
- stable-diffusion.cpp resizes the inputs itself (`src/runtime/image_preprocess.cpp`).
- Web UI: "Edit an image" in the Image tab and an Edit button on every gallery image; klein gets `ref_images_b64`, other models get `init_image_b64` with a three-level strength choice.
- The web proxy's video timeout is now 3600 s: a 33-frame 832×480 HunyuanVideo run took 2136 s on the PS5, longer than the old 900 s limit, so the page gave up while the PS5 kept working.

## 2026-10-10 (evening): image input for chat, and the PS5 driving a simulated robot arm

**Title `vision-20261010`** (deployed; rollback `eboot.before-vision-20261010-163025.bin`). The build script now passes `-I $llama_src/tools/mtmd` to `http_server.cpp`.

- `/v1/chat/completions` accepts OpenAI content parts: `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` (PNG or JPEG, up to 4 per request) alongside `{"type":"text"}`.
- The loaded LLM's vision projector is looked for as `/app0/<model name without .gguf>.mmproj.gguf` and loaded on the first request with an image; it is released whenever the LLM is. A model without that file answers 422 with a message saying so.
- Only projector on the PS5 so far: `gemma-3-12b-it-Q4_K_M.mmproj.gguf` (854 MB, from `ggml-org/gemma-3-12b-it-GGUF`, `mmproj-model-f16.gguf`; copy on Aisha in `~/models/vision/`).
- Cost per image on Gemma 3 12B: SigLIP encode at 896² about 1.6 s on Vulkan, 256 image tokens, about 2.7 s until generation starts. "Which colour is the block on the right?" answers in 2.9 s.
- `/v1/models` lists the loaded model under the id `ps5-local` (its `filename` field has the real name) and lists `.mmproj.gguf` files as if they were models; clients have to account for both.
- Web UI: an attach-image button in Chat; models with a projector are tagged "sees images".

**Simulator experiment** (`ps5lab/sim/`, on Aisha in `~/ps5lab/sim` with its own venv: `mujoco==3.2.7`, `gymnasium-robotics 1.4.2`; MuJoCo 3.15 breaks the Fetch environments; run with `MUJOCO_GL=egl`, Intel HD 530 renders 896² fine):

- `blocks_env.py`: Fetch arm, three coloured cubes on pads (left, middle, right), a plate, one camera. Commands `pick X`, `place X`, `done`; the arm motion for each is scripted (6/6 on its self-test, including stacking). So this is a vision-language model choosing high-level steps, not a VLA producing motor actions.
- `vla_agent.py`: each turn makes two PS5 calls, one that only describes the image and one that picks a command from that description, the task and the history. About 7.6 s per turn.
- Result, 12 seeded tasks (place a cube, place the cube that is "neither X nor Y", place and stack), strict success check, Gemma 3 12B Q4_K_M:

  | | first pick is the right cube | task completed |
  |---|---|---|
  | with the camera image | 11/12 | 5/12 |
  | blind control (no image) | 2/12 | 2/12 |

- So the PS5 does see and act on the scene. What fails is the multi-step part: it said `done` in only 2 of 12 episodes, often carried on after finishing (undoing the result), and its descriptions of later frames have slips (a phantom cube on the yellow plate, a held blue cube called red).
- What mattered while getting there: (1) one combined "describe and decide" prompt made the model describe what the task wanted rather than the image, so splitting the calls took the row description from about 60% to 100% right; (2) rigid `key=value` output hurt perception, free sentences were better; (3) cubes 20 px wide were misread, so the camera moved closer and renders at 896², Gemma 3's native input size; (4) grey pads were needed for it to report "empty".
- Next steps if this continues: a model with better grounding (Qwen2.5-VL class) for the describe step, a stronger or reasoning model for the decide step, then a real VLA port (SmolVLA first).

## 2026-10-11 (night): Qwen2.5-VL, a crash that needed a reboot, and recovery notes

**Qwen2.5-VL 7B Instruct Q4_K_M** (`ggml-org/Qwen2.5-VL-7B-Instruct-GGUF`, 4.68 GB + `mmproj ... f16` 1.35 GB, SHA-256 checked against Hugging Face; on the PS5 as `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` and `.mmproj.gguf`, copies in Aisha `~/models/vision/`):

- **A q4_0 KV cache breaks it.** With `cache_type_k/v=q4_0` it emits garbage or stops after 0–3 tokens, text-only as well ("Say hello" -> `exemple\n Rencontre`). q8_0 and f16 are correct, with flash attention on. Other models (Qwen3-4B, Gemma) are fine on q4_0. Not investigated further; the model uses M-RoPE positions, which is the obvious suspect. The web UI now loads it with q8_0.
- Speed: 896×896 image = 1024 image tokens, encode about 1.3 s, 1160-token prompt evaluated in 3.9 s, generation 62 tok/s.
- **It did not improve the robot-arm result.** Same 12 tasks, same prompts:

  | model | first pick right | said `done` | completed | per turn |
  |---|---|---|---|---|
  | Gemma 3 12B | 11/12 | 2/12 | 5/12 | 7.6 s |
  | Qwen2.5-VL 7B | 10/12 | 0/12 | 5/12 | 6.3 s |
  | blind control | 2/12 | 0/12 | 2/12 | 1.7 s |

  Both models see the first frame correctly and both fail the same way afterwards (repeating `pick` while already holding, never stopping). The bottleneck is the decide step and the loop design, not the eyes.

**Title `vision2-20261010`** (deployed): before loading a vision projector it unloads Whisper and the TTS model, and sets `image_max_tokens = 1024`. Without that, Qwen-VL's projector warms up for a 1288×1288 image, the 652 MB work buffer fails to allocate, and the title dies with a null-pointer SIGSEGV in the encoder (2026-10-10 23:08).

**That crash left the app unlaunchable until a reboot.** Every `launch PPSA99005` afterwards failed in SceLncService with `lnc_app_info.cpp(184) query: 0x80940033 (m_is_pkg value is false)`; on screen "Can't start the game or app". Ruled out: the new eboot (the previous one failed identically), a pending error dialog, and the missing nullfs mount (`mount -t nullfs /data/homebrew/PPSA99005 /system_ex/app/PPSA99005` from shsrv worked but did not help). A reboot and re-jailbreak fixed it. Earlier exits by `kill -s KILL` never caused this; a real crash (coredump path) did. So: avoid crashing the title, a crash can cost a reboot.

**After a reboot and re-jailbreak, what is and is not back:**

- The Payload Manager autoloads kstuff, ShadowMount+, websrv, etaHEN, nanoDNS, ps5upload, ftpsrv, the web file manager, Pegasus DL, elfldr, shsrv. On 2026-10-11 `klogsrv` (last in `/data/pldmgr/autoload.txt`) did not start, and Tailscale was never in the list.
- Without Tailscale the tailnet address is dead, but the PS5 answers on the direct cable at `10.42.0.211`: FTP 2121, shsrv 2323 and **elfldr 9021** are open there. The inference server on 8081 accepts the TCP connection but only serves tailnet clients.
- To bring Tailscale back from Aisha: send `~/ps5lm-inference/tailscale/tailscale-0.7.1.elf` (checksum in `SHA256SUMS.txt` beside it) as raw bytes to `10.42.0.211:9021`. The loader answers "launcher: starting the Go program" and the node is online in about 10 s. `klogsrv` can be sent the same way (`/data/pldmgr/payloads/klogsrv/klogsrv_v0.9.elf`), then `systemctl --user restart ps5-klogd.service`.
- The Tailscale and Payload Manager tiles are web shortcuts (`http://127.0.0.1:8090/` and `:8084/`). "Can't connect to the internet" on those tiles means the program behind the page is not running, not that the network is down.
- Tailscale is now also on the PS5 at `/data/pldmgr/payloads/tailscale/tailscale-0.7.1.elf` and in `autoload.txt` before `klogsrv` (previous list saved as `autoload.before-tailscale.txt`). Not yet verified across a reboot.
- `ps5ctl.py` and `klogd.py` take `PS5_HOST` from the environment if the tailnet address is unavailable.

## 2026-10-11 (01:20): OpenVLA port, state before the first PS5 run

Target: the original OpenVLA (autoregressive, 7 action tokens), LIBERO-Spatial fine-tune, so the result can be compared with the paper's ~85% on that suite.

- **Weights, no conversion needed:** `arashakb/openvla-spatial-original-gguf` splits the checkpoint into `llm_q4_k_m.gguf` (Llama-2 7B, 4.08 GB, loads as an ordinary GGUF), `dinov2.gguf` (1.21 GB), `siglip.gguf` (1.65 GB), `proj.gguf` (0.29 GB); vision parts are F32 with clip-style tensor names and `clip.vision.feature_layer` set (22 of 24 for DINOv2, 25 of 27 for SigLIP). All four SHA-256-verified. On Aisha in `~/models/openvla/` with `dataset_statistics.json` from `openvla/openvla-7b-finetuned-libero-spatial`.
- **Related, not used:** `vla.cpp` (VinRobotics) is a llama.cpp-based engine for SmolVLA, π0, GR00T, OpenVLA-OFT etc. with single-file GGUFs on `huggingface.co/vrfai`. Its OpenVLA-OFT file is 14 GiB BF16, too big for the 12 GiB heap as shipped. Worth a look for SmolVLA/π0 later.
- **Code** (`vk/`): `vla_vision.{h,cpp}` builds both ViTs and the projector on ggml (same operations as `clip.cpp`, plus class/register tokens and layer scale for DINOv2, exact GELU); `vla_policy.{h,cpp}` does one step: prompt `In: What action should the robot take to {instruction}?\nOut:` + token 29871, BOS then 256 patch embeddings then the text, greedy 7 tokens, token -> bin -> [-1, 1]. Endpoint `POST /v1/vla/act` (`instruction`, `rgb_b64` = 224×224×3 raw bytes, optional `model`); model layout on the PS5 is `/app0/<model>.gguf` + `/app0/<model>/{dinov2,siglip,proj}.gguf`. Title `vla-20261011` is built, **not yet deployed or run on the PS5**.
- **Verified on Aisha's CPU:** the ggml image encoder against timm's DINOv2/SigLIP with the same weights (`vla/reference_vision.py`): cosine ≥ 0.999991 on all 256 embeddings, mean abs difference 0.0003. Tokenisation of the prompt matches `tokenizer.json` (291 prompt positions with the image). The full model could not be run on Aisha (7.8 GB RAM, it swaps).
- **Simulator:** LIBERO runs on Aisha under Python 3.13 in the `~/ps5lab/sim` venv (`robosuite 1.4.1`, `bddl`, `mujoco 3.2.7`, CPU torch; `PYTHONPATH=~/vla/LIBERO`, `~/.libero/config.yaml` written by hand, `torch.load(weights_only=False)` for the init states). `vla/libero_eval.py` follows OpenVLA's LIBERO evaluation (10 idle steps, 220 max, JPEG round trip + Lanczos 224 + 0.9 centre crop, gripper binarised and sign-flipped).
- **Next:** stage the four files, deploy `vla-20261011`, one sanity step (tokens should fall in 31744..31999), then tasks 0–9 × 3 trials.

**Archive drive:** models are being moved from Aisha `~/models` to `PS5_STORAGE` (`~/models/offload.py`: copy, SHA-256 read-back, then delete local). The drive dropped off USB once mid-copy (re-enumerated sdd -> sde); mount it by label. Muse-Glimmer-30B and qwen2.5-coder-3b stay on Aisha, the drive is too full for them. Still owed: re-check the three files moved before the disconnect against the PS5's copies.

**Pending for the next deploy:** rename the title to "PSllama" with the icon in `vk/branding/` (see its README).

## 2026-10-11 (01:51): OpenVLA runs on the PS5 and completes LIBERO tasks (evaluation in progress)

**It works.** Title `vla-20261011` is deployed (rollback `eboot.before-vla-20261011-013049.bin`). Model on the PS5: `/app0/openvla-libero-spatial.gguf` (Llama-2 7B Q4_K_M, 4.08 GB) and `/app0/openvla-libero-spatial/{dinov2,siglip,proj}.gguf`, all SHA-256-verified after upload.

- **First step:** all seven tokens in the action range (e.g. `31872 31854 31871 31870 31862 31857 31744`), identical across repeated calls.
- **Speed:** about 1.0 s per control step warm: image encoder 0.16 s (both ViTs and the projector on Vulkan; scheduler reports 998 nodes, 2 splits, 20 MiB compute buffer), 291-position prompt 0.73 s, seven action tokens 0.11 s. Cold start 79 s (language model 4 GB + 3.15 GB of F32 vision weights). KV cache f16, context 1024, flash attention on.
- **First LIBERO episode succeeded** (task 0, "pick up the black bowl between the plate and the ramekin and place it on the plate", 88 steps, 104 s). Replay: `vla/openvla-first-success.gif`.
- **Evaluation running since 01:35:** LIBERO-Spatial tasks 0–9 × 3 trials (`~/vla/libero-runs.log` and `libero-runs/results.json` on Aisha), 1.1 s per step including the simulator. **At 01:51: 6 of 30 episodes done, 6 succeeded, 0 failed** (task 0: 3/3 in 84–116 steps; task 1: 3/3 in 107–135 steps). Reference: the OpenVLA paper reports about 85% on this suite for this checkpoint. Final numbers to be added here when it ends (expected 02:40–03:15).
- To run it: `cd ~/vla && PYTHONPATH=$HOME/vla/LIBERO MUJOCO_GL=egl ~/ps5lab/sim/venv/bin/python libero_eval.py --tasks 0-9 --trials 3`; a single step: `python3 one_step.py libero-frame.rgb "<instruction>"`.

**Archive move finished (01:42).** About 67 GB moved from Aisha `~/models` to `PS5_STORAGE` with SHA-256 read-back before each local delete; Aisha went from 7.7 GB to 60 GB free. The drive is full (2.2 GB left), so three files remain on Aisha only (and on the PS5): `Muse-Glimmer-30B-UD-IQ2_XXS.gguf`, `vision/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`, `qwen2.5-coder-3b-instruct-q4_k_m.gguf`. A size-then-hash scan of the drive found no duplicate files; the `Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf` inside the HunyuanVideo folder is a different build from the standalone one. Re-check of the three files moved before the USB drop, against the PS5's copies: `Qwen3-4B-Q4_K_M.gguf` matches; the other two are being hashed (`~/models/recheck.log`).

**Still pending:** apply the "PSllama" name and icon (`vk/branding/`): `titleName` in four `param.json` copies and four `icon0.png` copies on the PS5. Held until the evaluation ends, since ShadowMount rescans the app folder every 15 s and the effect of changing those files under a running app is unknown.

## 2026-10-11 (02:03): first state-space model on the PS5 (Mamba smoke test)

- Our llama.cpp (`552f18f91`) already has Mamba, Mamba-2, RWKV-6/7, Falcon-H1, Granite hybrid and Jamba, and ggml-vulkan has shaders for `SSM_CONV`, `SSM_SCAN`, `RWKV_WKV6/7` and `L2_NORM`. No code was needed.
- **`mamba-130m-hf.Q4_K_M.gguf`** (90 MB, `QuantFactory/mamba-130m-hf-GGUF`, SHA-256 checked, on the PS5 and in Aisha `~/models/ssm/`): loads in 1 s, generates without a crash or error. **65.5 to 67.9 tok/s** (24, 64 and 256 new tokens), 37-token prompt in 118 ms. Loaded with f16 "KV" settings and flash attention on; neither should apply to it.
- **That is slow for a 90 MB model.** By the bandwidth rule (3.3 ms fixed + bytes ÷ 386 GB/s) it should be near 300 tok/s; it takes about 15 ms per token. Not investigated. Suspects: the per-layer state-space ops being slow or run on the CPU with copies, or per-op overhead across 24 layers.
- **Output quality could not be judged.** It is a base model with no chat template, and the server only has `/v1/chat/completions`, which wraps the prompt in ChatML; the model just echoed `<|im_start|>user`. Whether the arithmetic is right needs a raw `/v1/completions` endpoint (not written yet) or an instruct-tuned state-space model.
- Candidates that fit, not downloaded: RWKV-7 G1 2.9B (2.05 GB), Jamba Reasoning 3B (1.93 GB), Falcon-Mamba 7B Instruct (4.20 GB), Mamba-Codestral 7B (4.15 GB), Granite 4.0 H Tiny (4.23 GB), RWKV-7 G1 7.2B (4.58 GB), Falcon-H1 7B (4.60 GB); large hybrids at IQ2_XXS: Falcon-H1 34B (9.49 GB), Granite 4.0 H Small (10.93 GB).

**LIBERO evaluation** was frozen with SIGSTOP right after its 10th episode (02:00:21, `~/vla/pause_after_episode.sh`), the PS5 swapped to Mamba, and the run resumed at 02:02:33; the next action request reloads OpenVLA by itself (about 80 s). At the pause: 10 of 30 episodes done.

**Archive re-check complete:** all three files moved before the USB drop match the PS5's copies byte for byte (Qwen3-4B, abliterated Qwen3.8-27B, abliterated Gemma 4 12B).

### Mamba-130m: GPU against CPU at four prompt lengths (02:08)

`ps5lab/ssm_bench.py` on Aisha (pauses LIBERO at an episode boundary, runs, resumes). 64 new tokens per run, greedy, rates from the title's klog lines.

| prompt tokens | GPU (`gpu_layers=999`) read | GPU write | CPU (`gpu_layers=0`, 4 threads) read | CPU write |
|---|---|---|---|---|
| 37 | 311 tok/s | 68.5 tok/s | 262 tok/s | 137.0 tok/s |
| 267 | 378 tok/s | 67.3 tok/s | 360 tok/s | 132.3 tok/s |
| 1027 | 377 tok/s | 68.5 tok/s | 376 tok/s | 137.0 tok/s |
| 3327 | 392 tok/s | 68.5 tok/s | 377 tok/s | 123.7 tok/s |

- **For this tiny model the CPU is twice as fast at generating** (about 135 against 68 tok/s) and the same at reading prompts.
- Prompt reading is flat at about 377 tok/s on both and does not improve with length or with the GPU, which points at the same bottleneck in both modes. Not confirmed; the natural suspects are the sequential state-space scan and/or those ops running on the CPU even when layers are "on the GPU", with the GPU run paying extra for transfers on every token.
- Neither rate depends on prompt length, as expected for a model with a fixed-size state: writing after a 3327-token prompt is as fast as after 37.
- This says nothing yet about 3–7B state-space models, where the weights are large enough for GPU bandwidth to matter. Output correctness is still unverified (base model, chat-only endpoint).

### Mamba-1.4B (base): GPU and CPU are the same speed, and why (02:34)

`mamba-1.4b-hf.Q4_K_M.gguf` (0.82 GB, `mradermacher/mamba-1.4b-hf-GGUF`), same benchmark as the 130M model.

| prompt tokens | GPU read | GPU write | CPU read | CPU write |
|---|---|---|---|---|
| 37 | 83 tok/s | 26.3 tok/s | 79 tok/s | 23.8 tok/s |
| 267 | 96 tok/s | 26.1 tok/s | 90 tok/s | 24.0 tok/s |
| 1027 | 100 tok/s | 26.3 tok/s | 97 tok/s | 24.6 tok/s |
| 3327 | 100 tok/s | 26.3 tok/s | 98 tok/s | 24.0 tok/s |

- GPU and CPU are within 10% of each other. A transformer of this file size does roughly 180 tok/s on this GPU, so Mamba is running at about a seventh of what the hardware allows.
- **Cause, from the load log:** all 49 layers and 768 MiB of weights are on Vulkan0, but the scheduler reports `graph: nodes = 2360, splits = 98`, about two splits per layer (a transformer has 1 or 2 in total). Some operation in every Mamba layer is not accepted by the Vulkan backend and runs on the CPU, so each token crosses GPU -> CPU -> GPU 48 times. Which op (most likely `SSM_SCAN` or `SSM_CONV` failing its `supports_op` conditions on this build) has not been identified yet. That is the thing to fix before state-space models are worth running here.
- Rates are flat across prompt lengths again (fixed-size state: `llama_memory_recurrent` 14.25 MiB for 48 layers, against hundreds of MiB of KV cache for a transformer at 4096 context).
- Output still cannot be judged: base model, chat-only endpoint, it echoes the ChatML markers. The instruct fine-tune (`mamba-1.4b-instruct-hf.Q4_K_M.gguf`) is being benchmarked next.

### Mamba-1.4B Instruct: same speed, and the arithmetic looks right (02:40)

`mamba-1.4b-instruct-hf.Q4_K_M.gguf` (0.82 GB, `RichardErkhov/scottsus_-_mamba-1.4b-instruct-hf-gguf`).

- Speed is the same as the base model: GPU 26.1–26.3 tok/s writing and 84–100 tok/s reading; CPU 24.0–24.4 and 79–97. Same per-layer GPU/CPU bouncing.
- **Output is fluent, grammatical English**, e.g. "The largest port in the world is the Port of Sasebo in Japan." and "This example uses the `|` operator to create a blank space, then uses the `im_start` and `im_end` variables to define the f…". A broken computation gives noise, so the state-space path on this build is computing sensibly on both GPU and CPU.
- It does not answer the questions, because the server wraps every prompt in ChatML and this fine-tune was trained on a different prompt format; it treats the markers as text. Judging answer quality needs a raw `/v1/completions` endpoint or a per-model template.

## 2026-10-11 (03:10): rename attempt left the app unlaunchable again; Mamba-1 GPU scan built but untested

**Do not edit the registered copies of `param.json` / `icon0.png`.** `ps5lab/apply_branding.py` rewrote `titleName` and the icon in `/data/homebrew/PPSA99005/sce_sys`, `/user/appmeta/PPSA99005`, `/system_data/priv/appmeta/PPSA99005` and `/user/app/PPSA99005[/sce_sys]` while the title was stopped for a deploy. The next `launch PPSA99005` failed exactly as after the 2026-10-10 crash: `lnc_app_info.cpp(184) query: 0x80940033 (m_is_pkg value is false)`, "Can't start the game or app". No crash was involved this time, so the two incidents have different triggers and the same end state: the system's app record goes invalid and stays so until a reboot (ShadowMount re-registers at boot). Restoring the files did not bring it back without a reboot.

- **The backups were not made.** The script tested existence with FTP `SIZE`, which this server answers for missing files too, so it skipped the "save original" step. Restored afterwards: `icon0.png` byte-exact from Aisha `~/prosperoai-audit/sce_sys/icon0.png` (481382 bytes); `param.json` with the original contents (as printed earlier in the session) but different whitespace (734 bytes, originally 667; no byte-exact copy exists).
- A rename probably has to go through ShadowMount's own registration (new `param.json`/icon in the source folder, then uninstall and re-register from its web page), not by editing the system's copies. Not tried.
- FTP on this server: `SIZE`, `NLST` and `MDTM` are unreliable or missing; use `LIST` to test for a file.

**Mamba-1 on the GPU.** Cause of the CPU fallback: ggml-vulkan's `SSM_SCAN` only handled Mamba-2 (`supports_op` returned false otherwise; d_state 128/256, head_dim % 16). Patch: `vk/llama-patches/ssm_scan_mamba1.comp` + `apply_mamba1_scan.py` (new pipeline `ssm_scan_mamba1_f32`, one invocation per channel, d_state 16, K = 1; originals saved as `*.before-mamba1-20261011` in the llama.cpp tree). `~/llama-vk/build` rebuilt and title `mamba1-20261011` built and **deployed** (rollback `eboot.before-mamba1-20261011-*.bin`), but it has not started yet because of the launch failure above, so the shader is untested.

**Waiting on a PS5 reboot.** LIBERO evaluation is paused (SIGSTOP) after 26 of 30 episodes: 21 succeeded, 5 failed. Three small Mamba-2 models are downloaded to Aisha `~/models/ssm/` and not yet on the PS5: `mamba2-130m-hf-q8_0.gguf`, `granite-4.0-h-350m-Q4_K_M.gguf`, `Falcon-H1-0.5B-Instruct-Q4_K_M.gguf`.

## 2026-10-11 (03:31): Mamba-1 on the GPU works, about 6 to 7 times faster

Title `mamba1b-20261011` (deployed; includes the OpenVLA endpoint). The first attempt (`mamba1-20261011`) crashed with SIGFPE on the first Mamba-1 request: `ggml_vk_ssm_scan` still ran the Mamba-2 workgroup sizing (`d_state / subgroup_size` = 16 / 32 = 0, then `CEIL_DIV` by it). Fixed by sizing per path; `vk/llama-patches/apply_mamba1_scan.py` has the corrected version. That crash did **not** leave the app unlaunchable (it was relaunched from the home screen), unlike the 2026-10-10 one.

Same benchmark as before (`ps5lab/ssm_bench.py`, 64 new tokens, greedy):

| model | prompt tokens | GPU read | GPU write | CPU read | CPU write |
|---|---|---|---|---|---|
| Mamba-1.4B Instruct Q4_K_M | 37 | 755 tok/s | 170.7 tok/s | 78 | 24.2 |
| | 267 | 1428 | 159.8 | 90 | 24.0 |
| | 1027 | 1641 | 159.8 | 96 | 24.4 |
| | 3327 | 1657 | 159.8 | 97 | 23.8 |
| Mamba-130M Q4_K_M | 37 | 4625 | 479.5 | 282 | 137.0 |
| | 267 | 7026 | 479.5 | 361 | 132.3 |
| | 1027 | 8350 | 479.5 | 375 | 137.0 |
| | 3327 | 8920 | 383.6 | 376 | 132.3 |

- Before the patch the "GPU" run of the 1.4B model wrote 26 tok/s and read 100 tok/s; now 160 and about 1650. Graph splits went from 98 to 2.
- 160 tok/s for a 0.82 GB file is in line with transformers of that size on this GPU, so the scan is no longer the bottleneck.
- Writing speed is still flat across prompt lengths (fixed-size state).
- **Correctness:** with the new shader the GPU answers are character-for-character what the unpatched build produced in GPU mode, when the scan itself still ran on the CPU (e.g. "…The largest port in the world is the Port of Sasebo in Japan."). So the shader reproduces the CPU scan. Pure-CPU runs diverge from GPU runs after a dozen or so tokens on some prompts, as they did before the patch; that comes from the other operations' rounding, not from the scan.
- Autoload: `tailscale-0.7.1.elf` and `klogsrv_v0.9.elf` at the end of `/data/pldmgr/autoload.txt` did not load after the 03:15 reboot (everything before them did). Both were loaded by hand through elfldr again.
- Rename: after "Uninstall" in ShadowMount's page, a scan could not re-register the title in the same session (`[REG] internal AppInstallTitleDir bridge unavailable`); after a reboot ShadowMount installed it from the source folder as "PSllama" with the new icon. So: change `sce_sys/param.json` and `icon0.png` in the source folder only, uninstall in ShadowMount (never "Delete source"), reboot.

## 2026-10-11 (03:38): Mamba-2 and the Mamba-2 hybrids run on the GPU out of the box

No patch involved for these (the stock `ssm_scan.comp` handles Mamba-2); all load with 2 graph splits on the GPU. Same benchmark, 64 new tokens unless the model stopped earlier.

| model | file | GPU write | CPU write | GPU read (37 / 267 / 1027 / 3327 prompt tokens) | CPU read |
|---|---|---|---|---|---|
| mamba2-130m (pure Mamba-2, base) | Q8_0, 0.18 GB | 479.5 tok/s | 137.0 | 3083 / 8613 / 7900 / 11394 tok/s | 881 / 1534 / 1802 / 1873 |
| Granite 4.0 H 350M (hybrid, instruct) | Q4_K_M, 0.22 GB | 180–214 | 39–44 | 710 / 2362 / 2579 / 5106 | 587 / 835 / 1041 / 905 |
| Falcon-H1 0.5B Instruct (hybrid) | Q4_K_M, 0.31 GB | 183–192 | 34–63 | 1136 / 2356 / 2827 / 2566 | 129 / 493 / 581 / 368 |

- GPU is 3.5 to 5 times faster than the PS5's CPU at writing on all three.
- **The two instruct hybrids answer correctly** through the ordinary chat endpoint ("The capital of France is Paris."; a sensible two-sentence answer about the ocean; three fruits), and GPU and CPU give the same answers apart from a word or two late in longer replies. These are the first state-space-family models here whose output could be judged, and it is right.
- Falcon-H1's CPU writing slows as the prompt grows (62.9 -> 33.7 tok/s from 25 to 3644 tokens), because half of it is attention with a growing cache; on the GPU it stays at about 183 tok/s. Pure Mamba is flat on both.
- The 480 tok/s ceiling on the two 130M models looks like the server's own per-token overhead (about 2 ms), not the models.
- Files are on the PS5 and in Aisha `~/models/ssm/`. Not tried yet: anything at 3–7B (RWKV-7, Falcon-Mamba 7B, Mamba-Codestral 7B, Granite 4.0 H Tiny, Falcon-H1 7B) or RWKV at all.

LIBERO resumed at 03:38:09 from 26 of 30 episodes (21 succeeded, 5 failed).

## 2026-10-11 (03:57): OpenVLA on the PS5, LIBERO-Spatial result: 23 of 30 (77%)

`openvla-libero-spatial` (Llama-2 7B Q4_K_M + F32 vision towers) on the PS5, LIBERO on Aisha, tasks 0–9 × 3 trials, OpenVLA's own evaluation procedure (`vla/libero_eval.py`; results in Aisha `~/vla/libero-runs/results.json`, replays of each task's first trial as GIFs beside it).

| task | instruction (… and place it on the plate) | succeeded | steps when successful |
|---|---|---|---|
| 0 | black bowl between the plate and the ramekin | 3/3 | 84–116 |
| 1 | black bowl next to the ramekin | 3/3 | 107–135 |
| 2 | black bowl from table center | 2/3 | 95–112 |
| 3 | black bowl on the cookie box | 3/3 | 87–115 |
| 4 | black bowl in the top drawer of the wooden cabinet | 1/3 | 125 |
| 5 | black bowl on the ramekin | 2/3 | 95–100 |
| 6 | black bowl next to the cookie box | 2/3 | 106 |
| 7 | black bowl on the stove | 3/3 | 114–134 |
| 8 | black bowl next to the plate | 2/3 | 107–158 |
| 9 | black bowl on the wooden cabinet | 2/3 | 119–125 |

- **23/30 = 77%**; the OpenVLA paper reports 84.7% for this checkpoint on this suite (500 episodes, full precision). With 30 episodes the uncertainty is roughly ±15 points, so this is consistent with the paper but does not show that 4-bit weights cost nothing. Failures all ran to the 220-step limit.
- 4110 control steps at 1.25 s each including the simulator; the PS5 part is 0.94 s (vision 0.16, prompt 0.68, action tokens 0.10). Mean 112 steps per successful episode, about 2 minutes.
- The run was paused with SIGSTOP five times at episode boundaries while other models were loaded on the PS5, and survived a console reboot between episodes 26 and 27; each resume reloads OpenVLA in about 80 s.
- Not done: other LIBERO suites, more trials, a wrist camera, action chunking (OpenVLA-OFT), or any speed work on the 291-token prompt (the 256 image tokens are re-evaluated every step).

## Media index (for the repository page), as of 2026-10-11 04:05

Everything generated on the PS5 that was saved as a file is now collected on the Mac under **`ps5lm-tests/media/`** (72 MB), copied from Aisha; older picks are also in `ps5lm-tests/images/` and `ps5lm-tests/video/`. Originals remain on Aisha at the paths in brackets.

| What | Path under `ps5lm-tests/` | Source on Aisha |
|---|---|---|
| **OpenVLA driving LIBERO**: one replay GIF per task (first trial), `results.json`, the first camera frame | `media/libero-openvla/` | `~/vla/libero-runs/` |
| OpenVLA's very first successful episode (GIF) | `media/libero-openvla-first/`, also `vla/openvla-first-success.gif` | `~/vla/libero-first/` |
| Blocks-and-plate arm, Gemma 3 12B deciding: 12 episode GIFs, `vision.json`, `blind.json` | `media/sim-blocks-gemma3/`, also `ps5lab/sim/demo-stack-episode.gif`, `ps5lab/sim/scene.png` | `~/ps5lab/sim/runs/` |
| Same, Qwen2.5-VL 7B: 12 episode GIFs, `vision.json` | `media/sim-blocks-qwen25vl/` | `~/ps5lab/sim/runs-qwen/` |
| Image models compared on the same prompts (SD-Turbo, Z-Image-Turbo Q8/Q4, FLUX.2-klein, ERNIE; fox, ramen, astronaut) + `timings.json` | `media/compare-20261010/` (also `images/compare-20261010/`) | `~/ps5lab/compare-20261010/` |
| Abliterated Z-Image encoder, same prompts | `media/compare-20261010-abl/` | `~/ps5lab/compare-20261010-abl/` |
| Standard vs abliterated encoder on darker prompts | `media/encoder-ab/` (also `images/encoder-ab/`) | `~/ps5lab/encoder-ab/` |
| Image editing with FLUX.2-klein: base, reference edit (scarf, neon street), image-to-image | `media/edit-test/` | `~/ps5lab/edit-test/` |
| First images and resolution series (SD-Turbo, Z-Image 512/768/1024) | `images/ps5-*.png`, `media/misc/zfox-*.png`, `media/misc/fox-*.png` | `~/ps5lab/*.png` |
| Video frames: Wan 2.1 1.3B (9 frames), Wan 2.2 5B (17), HunyuanVideo 1.5 I2V (17) | `media/video-test/`, `media/wan22-test1/`, `media/hv-test1/`, `video/frames-*/` | `~/ps5lab/video-test/`, `wan22-test1/`, `hv-test1/` |
| Voice agent turns (WAV), TTS plain and cloned, STT test clip, the questions used | `media/voice-test/`, `media/misc/*.wav` | `~/ps5lab/voice-test/`, `~/ps5lab/*.wav` |
| PSllama icon (512 px) | `media/psllama-icon-512.png`, `vk/branding/icon0.png` | `~/ps5lab/icon0.png` |

Not saved as files anywhere: images and clips made by hand in the web UI (they live only in that browser's gallery, e.g. the 33-frame 832×480 HunyuanVideo run and the wizard-hat fox edit), and the videos exist only as PNG frames, not as encoded clips. Before publishing: review `compare-20261010-abl` and `encoder-ab` for suitability, and note the PSllama icon contains the PlayStation logo.
