// libc functions llama.cpp/ggml reference that the PS5 SDK does not provide.
// SPDX-License-Identifier: GPL-3.0-or-later
#include <errno.h>
#include <stddef.h>
#include <sys/types.h>

// ggml_print_backtrace launches gdb through execlp; there is no gdb here.
int execlp(const char *file, const char *arg, ...) {
    (void)file;
    (void)arg;
    errno = ENOSYS;
    return -1;
}

// An advisory hint for mmap'd model files; ignoring it is always correct.
int posix_madvise(void *address, size_t length, int advice) {
    (void)address;
    (void)length;
    (void)advice;
    return 0;
}

// The chat-template common library references this through its cache-path
// helper. The PS5 server never uses llama.cpp's filesystem cache; returning
// NULL preserves the helper's existing "no home directory" failure path.
struct passwd;
struct passwd *getpwuid(uid_t uid) {
    (void)uid;
    return NULL;
}
