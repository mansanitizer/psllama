// On-screen status panel for PS5 titles built on PS5_Vulkan's platform: one
// 1920x1080 VideoOut buffer the CPU draws text into and presents. Adapted from
// PS5_Vulkan src/demo_renderer.cpp (tiled_byte_offset, buffer setup).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {
int sceKernelDebugOutText(int, const char *);
std::size_t sceKernelGetDirectMemorySize();
int sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int,
                                  std::int64_t *);
int sceKernelMapDirectMemory(void **, std::size_t, int, int, std::int64_t, std::size_t);
int sceSystemServiceHideSplashScreen();
int sceVideoOutOpen(std::int32_t, std::int32_t, std::int32_t, const void *);
int sceVideoOutSetFlipRate(std::int32_t, std::int32_t);
int sceVideoOutSubmitFlip(std::int32_t, std::int32_t, std::uint32_t, std::int64_t);
int sceVideoOutWaitVblank(std::int32_t);
void sceVideoOutSetBufferAttribute2(void *, std::uint64_t, std::uint32_t, std::uint32_t,
                                    std::uint32_t, std::uint64_t, std::uint32_t, std::uint64_t);
int sceVideoOutRegisterBuffers2(std::int32_t, std::int32_t, std::int32_t, void *, std::int32_t,
                                void *, std::int32_t, void *);
}

namespace status_screen {
constexpr unsigned width = 1920, height = 1080;
constexpr std::size_t frame_bytes = 0x1000000;
constexpr std::size_t memory_bytes = frame_bytes * 2;
constexpr std::uint32_t background = 0xff20170f, foreground = 0xfff0ebdc;
inline std::uint8_t *pixels;
inline unsigned cursor_y = 100;
inline char headline_text[128] = {};
inline std::int32_t video_handle = -1;
inline int last_flip_result = 0;

// 5x7 glyphs, one byte per column, bit 0 at the top: digits, A-Z, then . : / - % = ( ) , ' ? !
inline const unsigned char glyphs[][5] = {
    {62, 81, 73, 69, 62}, {0, 66, 127, 64, 0},  {66, 97, 81, 73, 70},  {33, 65, 69, 75, 49},
    {24, 20, 18, 127, 16}, {39, 69, 69, 69, 57}, {60, 74, 73, 73, 48},  {1, 113, 9, 5, 3},
    {54, 73, 73, 73, 54}, {6, 73, 73, 41, 30},  {126, 17, 17, 17, 126}, {127, 73, 73, 73, 54},
    {62, 65, 65, 65, 34}, {127, 65, 65, 34, 28}, {127, 73, 73, 73, 65}, {127, 9, 9, 9, 1},
    {62, 65, 73, 73, 122}, {127, 8, 8, 8, 127}, {0, 65, 127, 65, 0},   {32, 64, 65, 63, 1},
    {127, 8, 20, 34, 65}, {127, 64, 64, 64, 64}, {127, 2, 12, 2, 127}, {127, 4, 8, 16, 127},
    {62, 65, 65, 65, 62}, {127, 9, 9, 9, 6},    {62, 65, 81, 33, 94},  {127, 9, 25, 41, 70},
    {70, 73, 73, 73, 49}, {1, 1, 127, 1, 1},    {63, 64, 64, 64, 63},  {31, 32, 64, 32, 31},
    {63, 64, 56, 64, 63}, {99, 20, 8, 20, 99},  {7, 8, 112, 8, 7},     {97, 81, 73, 69, 67},
    {0, 96, 96, 0, 0},    {0, 54, 54, 0, 0},    {32, 16, 8, 4, 2},     {8, 8, 8, 8, 8},
    {35, 19, 8, 100, 98}, {20, 20, 20, 20, 20}, {0, 28, 34, 65, 0},    {0, 65, 34, 28, 0},
    {0, 80, 48, 0, 0},    {0, 0, 7, 0, 0},      {2, 1, 81, 9, 6},      {0, 0, 95, 0, 0}};
inline const char punctuation[] = ".:/-%=(),'?!";

inline std::size_t tiled_byte_offset(unsigned x, unsigned y) {
    const std::uint32_t offset = ((y << 4) & 0x70U) ^ ((y << 5) & 0xf00U) ^ ((y << 9) & 0x1000U) ^
                                 ((y << 8) & 0x4000U) ^ ((x << 2) & 0xcU) ^ ((x << 5) & 0x380U) ^
                                 ((x << 4) & 0x400U) ^ ((x << 6) & 0x800U) ^ ((x << 9) & 0xa000U);
    const std::uint32_t block = (y >> 7) * ((width + 127U) >> 7) + (x >> 7);
    return (static_cast<std::size_t>(block) << 16) + offset;
}

inline void put(unsigned x, unsigned y, std::uint32_t color) {
    if (x < width && y < height)
        *reinterpret_cast<std::uint32_t *>(pixels + tiled_byte_offset(x, y)) = color;
}

inline void flush() {
    for (std::uint8_t *at = pixels, *end = pixels + frame_bytes; at < end; at += 64)
        __asm__ volatile("clflush (%0)" : : "r"(at) : "memory");
    __asm__ volatile("mfence" ::: "memory");
}

inline int present() {
    flush();
    if (video_handle >= 0) {
        last_flip_result = sceVideoOutSubmitFlip(video_handle, 0, 1, 1);
        if (last_flip_result >= 0) (void)sceVideoOutWaitVblank(video_handle);
    }
    return last_flip_result;
}

inline void clear_rows(unsigned top, unsigned bottom) {
    for (unsigned y = top; y < bottom && y < height; ++y)
        for (unsigned x = 0; x < width; ++x) put(x, y, background);
}

inline void line(const char *text);
inline void headline(const char *text);

inline bool init_failure(const char *stage, int result) {
    char message[160];
    std::snprintf(message, sizeof(message), "[status-screen] %s failed rc=0x%08x\n",
                  stage, static_cast<unsigned>(result));
    sceKernelDebugOutText(0, message);
    return false;
}

// Opens the display; returns false (and the title carries on without a screen)
// if any step fails.
inline bool init(const char *title) {
    (void)sceSystemServiceHideSplashScreen();
    const int video = sceVideoOutOpen(0xff, 0, 0, nullptr);
    if (video < 0) return init_failure("VideoOutOpen", video);
    std::int64_t physical = 0;
    void *mapped = nullptr;
    // Type 3 (write-combined): the CPU only writes here, never reads back.
    const int alloc_result = sceKernelAllocateDirectMemory(
        0, static_cast<std::int64_t>(sceKernelGetDirectMemorySize()), memory_bytes, 0x200000, 3, &physical);
    if (alloc_result < 0) return init_failure("AllocateDirectMemory", alloc_result);
    const int map_result = sceKernelMapDirectMemory(&mapped, memory_bytes, 0x33, 0, physical, 0x200000);
    if (map_result < 0) return init_failure("MapDirectMemory", map_result);
    pixels = static_cast<std::uint8_t *>(mapped);
    clear_rows(0, height);
    struct VideoBuffer { void *data, *metadata, *reserved0, *reserved1; };
    auto *second_frame = static_cast<std::uint8_t *>(mapped) + frame_bytes;
    VideoBuffer buffers[2]{{mapped, nullptr, nullptr, nullptr},
                           {second_frame, nullptr, nullptr, nullptr}};
    std::uint8_t attribute[80] = {};
    (void)sceVideoOutSetFlipRate(video, 0);
    sceVideoOutSetBufferAttribute2(attribute, UINT64_C(0x8000000022000000), 0, width, height, 0, 0, 0);
    const int register_result = sceVideoOutRegisterBuffers2(video, 0, 0, buffers, 2, attribute, 0, nullptr);
    if (register_result < 0) {
        pixels = nullptr;
        return init_failure("VideoOutRegisterBuffers2", register_result);
    }
    video_handle = video;
    headline(title);
    if (last_flip_result < 0) return init_failure("VideoOutSubmitFlip", last_flip_result);
    return true;
}

// Draws one line of text (upper-cased, unknown characters as blanks), wrapping
// back to the top when the screen is full.
inline void line(const char *text) {
    if (!pixels) return;
    constexpr unsigned scale = 3, step = 6 * scale, line_height = 30;
    if (cursor_y + line_height > height - 40) {
        cursor_y = 100;
        clear_rows(0, height);
        headline(headline_text);
    }
    clear_rows(cursor_y, cursor_y + line_height);
    unsigned x = 60;
    for (; *text && *text != '\n' && x + step < width - 40; ++text, x += step) {
        char ch = *text;
        if (ch >= 'a' && ch <= 'z') ch = char(ch - 32);
        const unsigned char *glyph = nullptr;
        if (ch >= '0' && ch <= '9') glyph = glyphs[ch - '0'];
        else if (ch >= 'A' && ch <= 'Z') glyph = glyphs[10 + ch - 'A'];
        else if (const char *p = std::strchr(punctuation, ch); p && ch) glyph = glyphs[36 + (p - punctuation)];
        if (!glyph) continue;
        for (unsigned col = 0; col < 5; ++col)
            for (unsigned row = 0; row < 7; ++row)
                if (glyph[col] & (1u << row))
                    for (unsigned dy = 0; dy < scale; ++dy)
                        for (unsigned dx = 0; dx < scale; ++dx)
                            put(x + col * scale + dx, cursor_y + row * scale + dy, foreground);
    }
    cursor_y += line_height;
    present();
}

// Keep the most important state visible even as progress messages scroll.
inline void headline(const char *text) {
    if (!pixels) return;
    std::strncpy(headline_text, text, sizeof(headline_text) - 1);
    headline_text[sizeof(headline_text) - 1] = '\0';
    constexpr unsigned scale = 4, step = 6 * scale, y = 36;
    clear_rows(20, 82);
    unsigned x = 60;
    for (const char *at = headline_text; *at && x + step < width - 40; ++at, x += step) {
        char ch = *at;
        if (ch >= 'a' && ch <= 'z') ch = char(ch - 'a' + 'A');
        const unsigned char *glyph = nullptr;
        if (ch >= '0' && ch <= '9') glyph = glyphs[ch - '0'];
        else if (ch >= 'A' && ch <= 'Z') glyph = glyphs[10 + ch - 'A'];
        else if (const char *p = std::strchr(punctuation, ch); p && ch) glyph = glyphs[36 + (p - punctuation)];
        if (!glyph) continue;
        for (unsigned col = 0; col < 5; ++col)
            for (unsigned row = 0; row < 7; ++row)
                if (glyph[col] & (1u << row))
                    for (unsigned dy = 0; dy < scale; ++dy)
                        for (unsigned dx = 0; dx < scale; ++dx)
                            put(x + col * scale + dx, y + row * scale + dy, foreground);
    }
    present();
}
} // namespace status_screen
