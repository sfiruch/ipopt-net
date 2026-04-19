/* glibc_compat.c — pin versioned glibc/gfortran symbols to old versions.
 *
 * Compiled on Ubuntu 24.04 (glibc 2.39), the following symbols default to
 * versions requiring modern glibc.  We provide local wrappers that call the
 * equivalent pre-2.34/pre-2.29 versioned symbols instead.
 *
 * Requires -Wl,-Bsymbolic-functions in the final library link so that PLT
 * calls within the .so resolve to our local wrappers.
 * The version script (compat.map) hides these wrappers from export.
 *
 * Compile: gcc -O2 -fPIC -c glibc_compat.c -o glibc_compat.o
 */

#include <stddef.h>

/* ---- Macros -------------------------------------------------------------- */

/* Declare an extern under a private name, bind to a versioned glibc symbol. */
#define BIND_V(ret, priv, pub, ver, ...) \
    extern ret priv(__VA_ARGS__); \
    __asm__(".symver " #priv "," #pub "@" ver)

/* Define a wrapper: call priv (= pub@ver) and export as pub. */
#define WRAP1(ret, pub, ver, T1) \
    BIND_V(ret, __compat_##pub, pub, ver, T1 _a); \
    ret pub(T1 _a) { return __compat_##pub(_a); }

#define WRAP2(ret, pub, ver, T1, T2) \
    BIND_V(ret, __compat_##pub, pub, ver, T1 _a, T2 _b); \
    ret pub(T1 _a, T2 _b) { return __compat_##pub(_a, _b); }

/* ---- Math functions (glibc 2.29/2.38 → 2.2.5) --------------------------- */

WRAP1(double, exp,  "GLIBC_2.2.5", double)
WRAP1(double, log,  "GLIBC_2.2.5", double)
WRAP2(double, pow,  "GLIBC_2.2.5", double, double)
WRAP2(double, fmod, "GLIBC_2.2.5", double, double)
WRAP1(float,  expf, "GLIBC_2.2.5", float)
WRAP1(float,  logf, "GLIBC_2.2.5", float)
WRAP2(float,  powf, "GLIBC_2.2.5", float,  float)

/* ---- pthread functions (glibc 2.34 → 2.2.5) ----------------------------- */
/* In glibc 2.34, pthread_* moved from libpthread.so into libc.so at GLIBC_2.34.
 * The GLIBC_2.2.5 aliases still exist in libc.so.6 on all versions.         */

#include <pthread.h>

BIND_V(int, __compat_pthread_create, pthread_create, "GLIBC_2.2.5",
       pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int pthread_create(pthread_t *t, const pthread_attr_t *a,
                   void *(*fn)(void *), void *arg)
{ return __compat_pthread_create(t, a, fn, arg); }

WRAP2(int, pthread_join,          "GLIBC_2.2.5", pthread_t, void **)
WRAP2(int, pthread_setspecific,   "GLIBC_2.2.5", pthread_key_t, const void *)
WRAP1(void *, pthread_getspecific,"GLIBC_2.2.5", pthread_key_t)

BIND_V(int, __compat_pthread_key_create, pthread_key_create, "GLIBC_2.2.5",
       pthread_key_t *, void (*)(void *));
int pthread_key_create(pthread_key_t *k, void (*dtor)(void *))
{ return __compat_pthread_key_create(k, dtor); }

WRAP1(int, pthread_key_delete, "GLIBC_2.2.5", pthread_key_t)

/* ---- dl functions (glibc 2.34 → 2.2.5) ---------------------------------- */
/* dlopen/dlsym/dlclose/dlerror/dladdr moved to libc.so at GLIBC_2.34.       */

BIND_V(void *, __compat_dlopen, dlopen, "GLIBC_2.2.5", const char *, int);
void *dlopen(const char *f, int m) { return __compat_dlopen(f, m); }

BIND_V(void *, __compat_dlsym, dlsym, "GLIBC_2.2.5", void *, const char *);
void *dlsym(void *h, const char *s) { return __compat_dlsym(h, s); }

BIND_V(int, __compat_dlclose, dlclose, "GLIBC_2.2.5", void *);
int dlclose(void *h) { return __compat_dlclose(h); }

BIND_V(char *, __compat_dlerror, dlerror, "GLIBC_2.2.5");
char *dlerror(void) { return __compat_dlerror(); }

typedef struct { const char *dli_fname; void *dli_fbase; const char *dli_sname; void *dli_saddr; } compat_Dl_info;
BIND_V(int, __compat_dladdr, dladdr, "GLIBC_2.2.5", const void *, compat_Dl_info *);
int dladdr(const void *a, compat_Dl_info *i) { return __compat_dladdr(a, i); }

/* ---- std::ios_base_library_init() (GLIBCXX_3.4.32 → stub) -------------- */
/* GCC 13 emits a reference to this symbol (mangled: _ZSt21ios_base_library_initv)
 * to initialize I/O streams (C++23 std::print support).  A no-op is safe for
 * Ipopt since stream initialization goes through other static-init paths.    */
void _ZSt21ios_base_library_initv(void) {}

/* ---- __isoc23_strtol (glibc 2.38 → strtol@2.2.5) ------------------------ */
/* GCC 13 with glibc 2.38 headers emits __isoc23_strtol for strtol() calls.
 * Fall back to the classic strtol which has been in glibc since 2.2.5.       */

#include <stdlib.h>
BIND_V(long, __compat_strtol, strtol, "GLIBC_2.2.5", const char *, char **, int);
long __isoc23_strtol(const char *s, char **e, int b)
{ return __compat_strtol(s, e, b); }

/* ---- memcpy (glibc 2.14 → 2.2.5) ---------------------------------------- */
/* In glibc 2.14, memcpy gained overlap-safe semantics and a new version tag.
 * Pinning to 2.2.5 is safe for all callers that don't use overlapping buffers
 * (which includes all math/solver code).                                      */

#include <stddef.h>
BIND_V(void *, __compat_memcpy, memcpy, "GLIBC_2.2.5", void *, const void *, size_t);
void *memcpy(void *dst, const void *src, size_t n)
{ return __compat_memcpy(dst, src, n); }

/* _gfortran_os_error_at is now provided by the statically-embedded libgfortran.a */

/* ---- Fortran flush_if_preconnected (PIC-compatible stub) ----------------- */
/* unix.o in libgfortran.a references stdin/stdout/stderr via R_X86_64_PC32,
 * which cannot be used when making a shared object.
 * The function takes a gfortran-internal stream* (NOT a FILE*), so we
 * cannot call fflush() on it. A no-op is safe: this only flushes stderr/
 * stdout when switching to another I/O unit, which is optional.             */
void _gfortrani_flush_if_preconnected(void *s) { (void)s; }

/* ---- Fortran async I/O stubs --------------------------------------------- */
/* async.o in libgfortran.a defines 'thread_unit' with an initial-exec TLS
 * relocation (R_X86_64_TPOFF32) which cannot be used in a shared object.
 * MUMPS never performs async Fortran I/O, so these symbols are only needed
 * to satisfy linker references from atomic.o and error.o. Provide PIC-
 * compatible no-op stubs.                                                    */

/* thread_unit: thread-local Fortran I/O unit pointer (gfc_unit *).
 * Declared with global-dynamic TLS model (PIC-compatible).                  */
__thread void *thread_unit = NULL;

/* All async functions below are no-ops; MUMPS never calls them. */
void _gfortrani_async_close()              {}
void _gfortrani_async_wait()               {}
void _gfortrani_async_wait_id()            {}
void _gfortrani_collect_async_errors()     {}
void _gfortrani_enqueue_close()            {}
void _gfortrani_enqueue_data_transfer_init() {}
void _gfortrani_enqueue_done()             {}
void _gfortrani_enqueue_done_id()          {}
void _gfortrani_enqueue_transfer()         {}
void _gfortrani_init_async_unit()          {}
