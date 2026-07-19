/*
 * libcpp_wrapper.c — shim to provide missing libc++ symbols on macOS 13.
 *
 * MediaPipe's libmediapipe.dylib was compiled against macOS 14's libc++,
 * which introduced __libcpp_verbose_abort.  macOS 13's libc++ doesn't have
 * this symbol.  This wrapper:
 *   1. Provides an empty implementation of the missing symbol
 *   2. Re-exports the system libc++ so all OTHER symbols still resolve
 *
 * Build:
 *   cc -shared -o libcpp_wrapper.dylib libcpp_wrapper.c \
 *      -Wl,-reexport_library,/usr/lib/libc++.1.dylib \
 *      -arch arm64
 */

#include <stdarg.h>
#include <stdlib.h>

/* Called by libc++ when a verbose abort is triggered.
 * We provide a stub — the abort itself is handled by the system libc++. */
void _ZNSt3__122__libcpp_verbose_abortEPKcz(const char *fmt, ...) {
    /* If you want to see what messages are being aborted, uncomment: */
    /* va_list args;
     * va_start(args, fmt);
     * vfprintf(stderr, fmt, args);
     * va_end(args);
     * fprintf(stderr, "\n"); */
    abort();
}
