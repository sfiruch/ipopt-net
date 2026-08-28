/* msvc_compat.c — let MinGW's ld consume MSVC-compiled MKL static archives.
 *
 * Intel oneAPI ships mkl_*.lib as MSVC COFF archives. MinGW's ld accepts them
 * for plain C entry points (pardiso/pardisoinit use the same Windows x64 C ABI
 * as MinGW), but the objects inside reference MSVC-only runtime helpers that
 * the MinGW sysroot does not provide. This file supplies them.
 *
 * Compile with the *cross* compiler, not the host one:
 *   x86_64-w64-mingw32-gcc -O2 -c msvc_compat.c -o msvc_compat.o
 */

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>

/* Stack canary (MSVC /GS security cookie) */
uintptr_t __security_cookie = 0x2B992DDFA232ULL;
void __cdecl __security_check_cookie(uintptr_t cookie) { (void)cookie; }

/* __chkstk (2 underscores, MSVC ABI) -> ___chkstk_ms (3 underscores, MinGW) */
__asm__(".globl __chkstk\n__chkstk:\n\tjmp\t___chkstk_ms\n");

/* __GSHandlerCheck: called by the MSVC SEH unwinder for /GS-protected
   functions. No-op stub: MKL code never overflows stacks in normal use. */
void __GSHandlerCheck(void) {}

/* __guard_dispatch_icall_fptr: Control Flow Guard indirect-call dispatch.
   MSVC-compiled MKL objects call this with the target address in rax.
   For non-CFG MinGW builds: just jump to rax (no CFG check needed). */
__attribute__((naked)) static void _guard_dispatch_icall(void) {
    __asm__("jmp *%rax\n");
}
void (*__guard_dispatch_icall_fptr)(void) = _guard_dispatch_icall;

/* UCRT stdio shims — MKL static libs call these for internal error messages.
   Options and locale parameters are ignored; MinGW equivalents handle the rest. */
int __cdecl __stdio_common_vsprintf(uint64_t opts, char *buf, size_t count,
                                    const char *fmt, void *locale, va_list args) {
    (void)opts; (void)locale;
    return vsnprintf(buf, count, fmt, args);
}
int __cdecl __stdio_common_vsnprintf_s(uint64_t opts, char *buf, size_t count,
                                       size_t maxcount, const char *fmt,
                                       void *locale, va_list args) {
    (void)opts; (void)locale;
    return vsnprintf(buf, count < maxcount ? count : maxcount, fmt, args);
}
int __cdecl __stdio_common_vfprintf(uint64_t opts, FILE *stream, const char *fmt,
                                    void *locale, va_list args) {
    (void)opts; (void)locale;
    return vfprintf(stream, fmt, args);
}
int __cdecl __stdio_common_vsscanf(uint64_t opts, const char *buf, size_t count,
                                   const char *fmt, void *locale, va_list args) {
    (void)opts; (void)locale; (void)count;
    return vsscanf(buf, fmt, args);
}
int __cdecl __stdio_common_vfscanf(uint64_t opts, FILE *stream, const char *fmt,
                                   void *locale, va_list args) {
    (void)opts; (void)locale;
    return vfscanf(stream, fmt, args);
}
