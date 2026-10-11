#!/usr/bin/env bash
# Builds the GGUF-on-Vulkan test title: our harness + llama.cpp/ggml-vulkan
# (cross-built with PS5_Vulkan's payload SDK fork) + RADV, linked the way
# PS5_Vulkan's tools/build-radv-title.sh links its RADV smoke title.
# Output: $out/eboot.bin and $out/sce_module/libc.prx (sce_sys is left to the
# slot it is deployed into). Never deploys or launches anything.
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${PS5_VULKAN_ROOT:?set PS5_VULKAN_ROOT to the PS5_Vulkan tree}
llama_src=${LLAMA_SOURCE:?set LLAMA_SOURCE to the llama.cpp source}
llama_build=${LLAMA_BUILD:?set LLAMA_BUILD to the ggml-vulkan cross-build directory}
sd_src=${SD_SOURCE:?set SD_SOURCE to the stable-diffusion.cpp source}
sd_build=${SD_BUILD:?set SD_BUILD to the stable-diffusion.cpp cross-build (sd/CMakeLists.txt)}
whisper_src=${WHISPER_SOURCE:?set WHISPER_SOURCE to the whisper.cpp source}
api_key=${PSLLAMA_API_KEY:?set PSLLAMA_API_KEY to the Bearer key the server should require}
archive=${RADV_ARCHIVE:-$root/.deps/native/radv-release/lib/libvulkan_radeon.ps5.a}
sdk_root="$root/.deps/native/ps5-payload-sdk"
native="$root/tooling/native"
tool="$root/build/host/ps5-native-tool"
out=${OUT:-$here/out}
work="$out/work"

llama_libs=("$llama_build/src/libllama.a" "$llama_build/ggml/src/libggml.a"
    "$llama_build/ggml/src/ggml-vulkan/libggml-vulkan.a" "$llama_build/ggml/src/libggml-cpu.a"
    "$llama_build/ggml/src/libggml-base.a" "$llama_build/common/libllama-common.a"
    "$llama_build/common/libllama-common-base.a" "$sd_build/sdcpp/libstable-diffusion.a"
    "$llama_build/tools/mtmd/libmtmd.a")
for file in "$archive" "$tool" "$sdk_root/bin/prospero-lld" "$root/runtime/libc.prx" "${llama_libs[@]}"; do
    [[ -e $file ]] || { echo "missing $file" >&2; exit 2; }
done

mkdir -p "$work/obj" "$work/stubs" "$out/sce_module"
cp "$here/inference.conf" "$out/inference.conf"
cc() { PS5_PAYLOAD_SDK="$sdk_root" sh "$root/tooling/prospero-clang18" "$@"; }
# stable-diffusion.cpp needs ggml tensor names of up to 160 bytes; ggml,
# llama.cpp and everything including ggml.h must agree on the struct layout.
maxname=-DGGML_MAX_NAME=160

cc -std=c++17 -O2 -Wall -Wextra -ffunction-sections -fdata-sections $maxname \
    -I "$here" -I "$llama_src/include" -I "$llama_src/ggml/include" \
    -c "$here/gguf_vulkan.cpp" -o "$work/obj/gguf_vulkan.o"
cc -std=c++17 -O2 -Wall -Wextra -fexceptions -frtti -ffunction-sections -fdata-sections $maxname \
    -I "$here" -I "$llama_src/include" -I "$llama_src/ggml/include" -I "$llama_src/common" -I "$llama_src/tools/mtmd" \
    -DPSLLAMA_API_KEY="\"$api_key\"" \
    -c "$here/http_server.cpp" -o "$work/obj/http_server.o"
cc -std=c++17 -O2 -Wall -Wextra -fexceptions -ffunction-sections -fdata-sections $maxname \
    -I "$here" -I "$sd_src/include" -I "$sd_src/thirdparty" -I "$llama_src/ggml/include" \
    -c "$here/sd_service.cpp" -o "$work/obj/sd_service.o"
cc -std=c++17 -O2 -Wall -Wextra -fexceptions -ffunction-sections -fdata-sections $maxname \
    -I "$here" -I "$whisper_src/include" -I "$llama_src/include" -I "$llama_src/ggml/include" \
    -I "$llama_src/tools/mtmd" -c "$here/audio_service.cpp" -o "$work/obj/audio_service.o"
for unit in vla_vision vla_policy; do
    cc -std=c++17 -O2 -Wall -Wextra -fexceptions -ffunction-sections -fdata-sections $maxname \
        -I "$here" -I "$llama_src/include" -I "$llama_src/ggml/include" \
        -c "$here/$unit.cpp" -o "$work/obj/$unit.o"
done
cc -std=c++17 -O2 -fexceptions -ffunction-sections -fdata-sections $maxname -DWHISPER_VERSION='"whispercpp"' \
    -I "$whisper_src/include" -I "$whisper_src/src" -I "$llama_src/ggml/include" \
    -c "$whisper_src/src/whisper.cpp" -o "$work/obj/whisper.o"
# whisper.cpp always references its ANEForge encoder (used only when
# ANEFORGE_ENCODER is set); mtmd needs llama.cpp's vendored SHA-256.
cc -std=c++17 -O2 -fexceptions $maxname -I "$whisper_src/include" -I "$whisper_src/src" -I "$llama_src/ggml/include" \
    -c "$whisper_src/src/aneforge/whisper-aneforge.cpp" -o "$work/obj/whisper-aneforge.o"
cc -std=c++17 -O2 -I "$llama_src/vendor/hash" -c "$llama_src/vendor/hash/hash.cpp" -o "$work/obj/hash.o"
cc -std=c11 -O2 -I "$llama_src/vendor/hash" -c "$llama_src/vendor/hash/sha256/sha256.c" -o "$work/obj/sha256.o"
cc -std=c11 -O2 -Wall -Wextra -c "$here/platform_shims.c" -o "$work/obj/platform_shims.o"
cc -std=c++20 -O2 -Wall -Wextra -fno-exceptions -fno-rtti -ffunction-sections -fdata-sections \
    -c "$native/app_crt.cpp" -o "$work/obj/app_crt.o"
cc -std=c++20 -O2 -Wall -Wextra -fno-exceptions -fno-rtti -ffunction-sections -fdata-sections \
    -c "$native/app_cpp_runtime.cpp" -o "$work/obj/app_cpp_runtime.o"

stub() {
    local library=$1 source=$2
    cc -std=c11 -O2 -fPIC -c "$root/$source" -o "$work/obj/${library}_stub.o"
    "$sdk_root/bin/prospero-lld" --shared -soname "${library}.prx" \
        -o "$work/stubs/${library}.so" "$work/obj/${library}_stub.o"
}
stub libSceAgc vendor/ps5/sdk/stubs/agc_canary_link_stub.c
stub libSceAgcDriver vendor/ps5/sdk/stubs/agc_driver_canary_link_stub.c

source "$root/tools/radv-link.sh"
radv_link_recipe "$root" "$sdk_root" "$archive" || exit 2
# ggml-vulkan dispatches dynamically from the global vkGetInstanceProcAddr; the
# archive's entry point is the ICD one.
radv_link_flags+=("--defsym=vkGetInstanceProcAddr=vk_icdGetInstanceProcAddr")
# The few Vulkan functions ggml-vulkan calls directly rather than through its
# dispatcher, bound to Mesa's common implementations in the archive.
for name in CmdCopyBuffer GetDeviceProcAddr GetPhysicalDeviceFeatures2; do
    radv_link_flags+=("--defsym=vk$name=vk_common_$name")
done

link() {
    "$sdk_root/bin/prospero-lld" "${radv_linker_script[@]}" --eh-frame-hdr "${radv_link_flags[@]}" "$@" \
        --version-script "$native/app-symbols.map" --exclude-libs=ALL \
        -e _start -o "$work/llvm-pie.elf" \
        "$work/obj/app_crt.o" "$work/obj/app_cpp_runtime.o" "$work/obj/gguf_vulkan.o" "$work/obj/http_server.o" "$work/obj/sd_service.o" "$work/obj/audio_service.o" "$work/obj/vla_vision.o" "$work/obj/vla_policy.o" "$work/obj/whisper.o" "$work/obj/whisper-aneforge.o" "$work/obj/hash.o" "$work/obj/sha256.o" "$work/obj/platform_shims.o" \
        --start-group "${llama_libs[@]}" --end-group \
        "$work/stubs/libSceAgc.so" "$work/stubs/libSceAgcDriver.so" \
        "${radv_link_inputs[@]}" \
        --as-needed "$sdk_root"/target/lib/*.so
}
# Mesa's dispatch tables name every entry point of every layer (sqtt_, rmv_,
# wsi_, vk_cmd_enqueue_, ...) through weak references; the ones no layer in
# this build defines must resolve to NULL. LLD 18 (Debian) exports undefined
# weak symbols of a PIE as dynamic imports, which the title converter refuses,
# so bind exactly those to address 0 and link again. Only Mesa's prefixes are
# accepted; anything else unresolved is an error.
link
# _ZTH*: C++ thread_local init hooks, weak by ABI design; NULL means the
# variable needs no dynamic initialization (stable-diffusion.cpp has two).
mesa_prefixes='^((radv|vk_cmd|vk_common|wsi|ctx_roll|rmv|rra|utrace|sqtt|annotate|threaded)_|_ZTH)'
"${LLVM_READELF:-llvm-readelf-18}" --dyn-syms "$work/llvm-pie.elf" |
    awk '$7 == "UND" && $5 == "WEAK" {print $8}' | sort -u > "$work/weak-unresolved.txt"
if grep -vE "$mesa_prefixes" "$work/weak-unresolved.txt" | grep -q .; then
    echo "unexpected unresolved weak symbols:" >&2
    grep -vE "$mesa_prefixes" "$work/weak-unresolved.txt" | head >&2
    exit 1
fi
sed 's/.*/--defsym=&=0/' "$work/weak-unresolved.txt" > "$work/weak-null.rsp"
link "@$work/weak-null.rsp"
echo "bound $(wc -l < "$work/weak-unresolved.txt") unresolved Mesa weak references to NULL"
"$tool" link --in "$work/llvm-pie.elf" --out "$work/eboot.elf" \
    --stub-dir "$sdk_root/target/lib" --stub "$work/stubs/libSceAgc.so" \
    --stub "$work/stubs/libSceAgcDriver.so" --module-sdk 0x02000009 \
    --companion-sdk 0x08050001 --file-name eboot.elf
"$tool" self --sign --in "$work/eboot.elf" --out "$out/eboot.bin" --magic 0x1D3D154F
cp "$root/runtime/libc.prx" "$out/sce_module/libc.prx"
"$tool" self --inspect --file "$out/eboot.bin" > /dev/null
printf 'GGUF Vulkan title: %s (%s bytes)\n' "$out" "$(stat -c %s "$out/eboot.bin")"
