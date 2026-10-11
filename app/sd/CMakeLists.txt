# Builds stable-diffusion.cpp as a static library for the PS5 title against the
# ggml that the llama.cpp cross-build already produced (one ggml in the title).
# Configure with the payload SDK's prospero.cmake toolchain; pass
#   -DSDCPP_SOURCE=<stable-diffusion.cpp> -DGGML_SOURCE=<llama.cpp>/ggml
#   -DGGML_BUILD=<llama cross-build>/ggml/src
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.16)
project(sd_ps5 C CXX)

add_library(ggml STATIC IMPORTED GLOBAL)
set_target_properties(ggml PROPERTIES
    IMPORTED_LOCATION "${GGML_BUILD}/libggml.a"
    INTERFACE_INCLUDE_DIRECTORIES "${GGML_SOURCE}/include"
    INTERFACE_COMPILE_DEFINITIONS "GGML_MAX_NAME=160")

set(SD_GGML_SOURCE_DIR "${GGML_SOURCE}" CACHE PATH "" FORCE)
set(SD_USE_UPSTREAM_GGML ON CACHE BOOL "" FORCE)
set(SD_USE_SYSTEM_GGML ON CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
add_subdirectory("${SDCPP_SOURCE}" sdcpp)
