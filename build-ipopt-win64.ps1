# Build IPOPT 3.14.19 for Windows x64     small, self-contained DLL
#
# Compilers:
#   C/C++   - MinGW-W64 gcc/g++ (via MSYS2)     produces compatible PE DLL
#   Fortran - MinGW-W64 gfortran (via MSYS2)     for MUMPS and LAPACK
#
# All Fortran/GCC runtimes are statically linked into ipopt-3.dll so no
# extra runtime DLLs are needed.
#
# Intel MKL Pardiso support:
#   Uses static MKL libraries from Intel oneAPI MKL (winget install Intel.oneMKL).
#   mkl_intel_lp64.lib, mkl_sequential.lib, mkl_core.lib are MSVC COFF archives;
#   MinGW's ld accepts them for C-interface functions (pardiso, pardisoinit).
#   The resulting ipopt-3.dll has MKL baked in     no separate MKL DLLs needed.
#       Same approach as the Linux build (statically links libmkl_*.a).
#
# Result:
#   ipopt-3.dll (~25 MB)     MUMPS + MKL Pardiso + all runtimes statically linked

$ErrorActionPreference = "Stop"

#        Paths                                                                                                                                                                                                                

$ScriptDir  = $PSScriptRoot
$Msys2Root  = "C:\msys64"
$Bash       = "$Msys2Root\usr\bin\bash.exe"
$OutputDir  = "$ScriptDir\IpoptNet\runtimes\win-x64\native"

# These paths are written into the bash script.
# Convert C:\path     /c/path (MSYS2 convention) and use forward slashes.
function ConvertTo-Msys2Path([string] $winPath) {
    if ($winPath -match '^([A-Za-z]):\\(.*)$') {
        '/' + $Matches[1].ToLower() + '/' + ($Matches[2] -replace '\\', '/')
    } else {
        $winPath -replace '\\', '/'
    }
}
# Use a space-free path     autotools configure rejects paths with spaces
$SourceDir  = "/c/ipopt-src"
$InstallDir = "/c/ipopt-install"

$IpoptRelease = "releases/3.14.19"

#        Intel oneAPI MKL (static linking)                                                                                                                            
#
# Intel oneAPI MKL provides static .lib archives for build-time linking.
# mkl_intel_lp64.lib + mkl_sequential.lib + mkl_core.lib are linked into
# ipopt-3.dll     no separate MKL DLLs required at runtime.
#
# MSVC .lib files are COFF archives; MinGW's ld handles them for plain C
# functions. The pardiso/pardisoinit symbols use the Windows x64 C ABI
# (same as MinGW), so no calling-convention issues arise.
#
# Threading: mkl_sequential.lib (single-threaded, no OpenMP dependency).
# Integer size: mkl_intel_lp64.lib (32-bit indices, matching IPOPT's default).

function Find-MklStaticDir {
    foreach ($base in @("C:\Program Files (x86)\Intel\oneAPI\mkl", "C:\Program Files\Intel\oneAPI\mkl")) {
        if (-not (Test-Path $base)) { continue }
        $ver = Get-ChildItem $base | Sort-Object Name -Descending | Select-Object -First 1
        if (-not $ver) { continue }
        # oneAPI 2025+: libs directly in lib\; older layout: lib\intel64\
        foreach ($sub in @("lib", "lib\intel64")) {
            if (Test-Path "$($ver.FullName)\$sub\mkl_intel_lp64.lib") {
                return "$($ver.FullName)\$sub"
            }
        }
    }
    return $null
}

$MklLibDir = Find-MklStaticDir

if (-not $MklLibDir) {
    Write-Host "`nIntel oneAPI MKL not found - installing Intel.oneMKL..." -ForegroundColor Cyan
    Write-Host "  (Provides ~500 MB of static libraries for pardisomkl support.)" -ForegroundColor Cyan
    winget install Intel.oneMKL --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Intel.oneMKL installation failed     building without pardisomkl." -ForegroundColor Yellow
    }
    $MklLibDir = Find-MklStaticDir
}

$MklStaticLflags = ""

if ($MklLibDir) {
    Write-Host "Intel oneAPI MKL: $MklLibDir" -ForegroundColor Green
    Write-Host "  pardisomkl will be statically linked into ipopt-3.dll." -ForegroundColor Green

    # Autotools configure rejects paths with spaces (e.g. "Program Files (x86)").
    # Windows maintains 8.3 short names for all paths on the system drive (NTFS default).
    # %~si in cmd expands to the short path     e.g. C:\PROGRA~2\Intel\...     no spaces.
    $mklShort   = (cmd /c "for %i in (`"$MklLibDir`") do @echo %~si").Trim()
    $mklLibMsys = ConvertTo-Msys2Path $mklShort

    # mkl_sequential: threading-free layer; avoids OpenMP/TBB dependencies.
    # mkl_intel_lp64: LP64 interface (32-bit integers, matching IPOPT's default).
    # mkl_core: main compute library.  All three have circular deps     --start-group.
    # IPOPT's IpPardisoMKLSolverInterface.cpp declares pardiso/pardisoinit inline
    # via extern "C", so no MKL header is needed during compilation.
    # Libtool strips bare .lib paths from --start-group blocks (unknown extension).
    # Wrapping each path in -Xlinker forces libtool to pass them straight to ld.
    # libmsvcrt.a and libm.a are included in the same group so they are rescanned
    # together with the MKL archives (libtool moves -l flags before MKL otherwise).
    # --allow-multiple-definition: MKL bundles some CRT symbols (vfprintf, _vsnprintf)
    # that also appear in libmsvcrt.a; keep the first (msvcrt.dll) definition.
    $MklStaticLflags = "-Wl,--allow-multiple-definition -Wl,--start-group -Xlinker $mklLibMsys/mkl_intel_lp64.lib -Xlinker $mklLibMsys/mkl_sequential.lib -Xlinker $mklLibMsys/mkl_core.lib -Xlinker /mingw64/lib/libmsvcrt.a -Xlinker /mingw64/lib/libm.a -Xlinker /mingw64/lib/libmingwex.a -Xlinker /mingw64/lib/libkernel32.a -Wl,--end-group"
} else {
    Write-Host "Intel oneAPI MKL not available - building without pardisomkl." -ForegroundColor Yellow
}

#        Install MSYS2 if absent                                                                                                                                                          

if (-not (Test-Path $Bash)) {
    Write-Host "`nInstalling MSYS2..." -ForegroundColor Cyan
    winget install --id MSYS2.MSYS2 --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "MSYS2 installation failed." }
    # First-run initialisation
    & $Bash -lc "exit 0"
    Start-Sleep -Seconds 5
}

#        Install required packages (idempotent)                                                                                                             

Write-Host "`nInstalling MSYS2 packages..." -ForegroundColor Cyan
& $Bash -lc "pacman -S --noconfirm --needed base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-lapack git patch wget curl pkg-config"

#        Write and execute the build script                                                                                                                      
#
# We run inside MSYS2's MinGW64 shell (mingw64.exe or bash --login with PATH set)
# so that gcc/gfortran/make are the MinGW-W64 variants.
# The key static-link flags ensure no MinGW runtime DLLs leak into the output.

$BuildSh = @"
#!/usr/bin/env bash
set -euo pipefail

# Activate MinGW-W64 environment
export PATH="/mingw64/bin:`$PATH"

echo "Compiler versions:"
gcc    --version | head -1
gfortran --version | head -1

SRC="$SourceDir"
INST="$InstallDir"
MKL_STATIC_LFLAGS="$MklStaticLflags"

mkdir -p "`$SRC"
cd "`$SRC"

#        coinbrew                                                                                                                                                                                                       
if [ ! -f coinbrew/coinbrew ]; then
  git clone --depth=1 https://github.com/coin-or/coinbrew.git coinbrew
fi
COINBREW="`$SRC/coinbrew/coinbrew"
chmod +x "`$COINBREW"

#        Fetch IPOPT and ThirdParty sources                                                                                                                         
# coinbrew fetch downloads: Ipopt, ThirdParty-Mumps (+ MUMPS source),
# ThirdParty-Lapack (+ Netlib LAPACK), ThirdParty-Metis (+ METIS source).
# --skip-update avoids re-fetching on subsequent runs.
if [ ! -d Ipopt ]; then
  "`$COINBREW" fetch Ipopt \
    --release=$IpoptRelease \
    --no-prompt
fi

#        Common flags                                                                                                                                                                                           
GCC_VER=`$(gcc -dumpversion)
GCC_LIB="/mingw64/lib/gcc/x86_64-w64-mingw32/`$GCC_VER"
STATIC_RUNTIMES="-Wl,--start-group `$GCC_LIB/libgcc_eh.a `$GCC_LIB/libgcc.a /mingw64/lib/libstdc++.a /mingw64/lib/libgfortran.a /mingw64/lib/libquadmath.a /mingw64/lib/libwinpthread.a -Wl,--end-group -Wl,-s"
LAPACK_LFLAGS="-Wl,--start-group /mingw64/lib/liblapack.a /mingw64/lib/libblas.a /mingw64/lib/libgfortran.a /mingw64/lib/libquadmath.a -Wl,--end-group"
COIN_PKGCFG="`$INST/lib/pkgconfig:/mingw64/lib/pkgconfig:/mingw64/share/pkgconfig"
NPROC=`$(nproc 2>/dev/null || echo 4)

#        Phase 1: Build MUMPS as static-only                                                                                                                      
# With --disable-shared, libtool installs only libcoinmumps.a (no .dll.a).
# When IPOPT links as a shared lib against a static-only dep, libtool inlines
# the archive     so all MUMPS code ends up inside ipopt-3.dll.
if [ ! -f "`$INST/lib/libcoinmumps.a" ]; then
  mkdir -p "`$SRC/build-mumps"
  cd "`$SRC/build-mumps"
  PKG_CONFIG_PATH="`$COIN_PKGCFG" \
  "`$SRC/ThirdParty/Mumps/configure" \
    --prefix="`$INST" CC=gcc FC=gfortran \
    CFLAGS="-O2 -march=x86-64 -DNDEBUG" FFLAGS="-O2 -march=x86-64" \
    LDFLAGS="`$STATIC_RUNTIMES" \
    --disable-shared --enable-static \
    --with-lapack-lflags="`$LAPACK_LFLAGS"
  make -j"`$NPROC" && make install
fi

#        Phase 2: Configure Ipopt
# Use configure directly (not coinbrew build) so we can patch libtool before make.
# If MKL static libs are available, append them to LAPACK_LFLAGS so configure
# detects pardiso() and enables pardisomkl.  Otherwise builds MUMPS-only.
if [ -n "`$MKL_STATIC_LFLAGS" ]; then
  # MSVC-compiled MKL .lib files reference MSVC intrinsics not provided by MinGW:
  #   __security_cookie / __security_check_cookie  (stack canary)
  #   __chkstk  (stack probe; MinGW uses ___chkstk_ms with an extra underscore)
  # Compile a tiny stub with MinGW gcc and prepend it to the link flags so the
  # configure link test (and the final DLL link) can resolve these symbols.
  cat > "`$SRC/msvc_compat.c" << 'COMPAT_EOF'
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>

/* Stack canary (MSVC /GS security cookie) */
uintptr_t __security_cookie = 0x2B992DDFA232ULL;
void __cdecl __security_check_cookie(uintptr_t cookie) { (void)cookie; }

/* __chkstk (2 underscores, MSVC ABI) -> ___chkstk_ms (3 underscores, MinGW) */
__asm__(".globl __chkstk\n__chkstk:\n\tjmp\t___chkstk_ms\n");

/* __GSHandlerCheck: called by MSVC SEH unwinder for /GS-protected functions.
   No-op stub: MKL code never overflows stacks in normal use. */
void __GSHandlerCheck(void) {}

/* __guard_dispatch_icall_fptr: Control Flow Guard indirect-call dispatch.
   MSVC-compiled MKL objects call this with the target address in rax.
   For non-CFG MinGW builds: just jump to rax (no CFG check needed). */
__attribute__((naked)) static void _guard_dispatch_icall(void) {
    __asm__("jmp *%rax\n");
}
void (*__guard_dispatch_icall_fptr)(void) = _guard_dispatch_icall;

/* UCRT stdio shims - MKL static libs call these for internal error messages.
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
COMPAT_EOF
  gcc -O2 -march=x86-64 -c "`$SRC/msvc_compat.c" -o "`$SRC/msvc_compat.o"
  # MKL static libs include pardiso(); configure will find it and enable pardisomkl.
  LAPACK_LFLAGS_IPOPT="`$SRC/msvc_compat.o `$LAPACK_LFLAGS `$MKL_STATIC_LFLAGS"
else
  LAPACK_LFLAGS_IPOPT="`$LAPACK_LFLAGS"
fi

rm -rf "`$SRC/build-ipopt"
mkdir -p "`$SRC/build-ipopt"
cd "`$SRC/build-ipopt"
PKG_CONFIG_PATH="`$COIN_PKGCFG" \
"`$SRC/Ipopt/configure" \
  --prefix="`$INST" CC=gcc CXX=g++ FC=gfortran \
  CFLAGS="-O2 -march=x86-64 -DNDEBUG" CXXFLAGS="-O2 -march=x86-64 -DNDEBUG" FFLAGS="-O2 -march=x86-64" \
  LDFLAGS="`$STATIC_RUNTIMES" \
  --enable-shared --disable-static --without-asl \
  --with-lapack-lflags="`$LAPACK_LFLAGS_IPOPT"

#        Phase 3: Patch libtool                                                                                                                                                             
# configure probes the C++ compiler and bakes -lgcc_s into postdeps, which
# forces dynamic loading of libgcc_s_seh-1.dll regardless of LDFLAGS.
# -lgcc_eh is the static-only SEH archive (no .dll.a counterpart), so replacing
# -lgcc_s with -lgcc_eh makes the final link pull in the .a instead.
find "`$SRC/build-ipopt" -name "libtool" \
  -exec sed -i 's/-lgcc_s\b/-lgcc_eh/g' {} \;

#        Phase 4: Build and install Ipopt                                                                                                                            
make -C "`$SRC/build-ipopt" -j"`$NPROC"
make -C "`$SRC/build-ipopt" install

echo ""
echo "Build complete."
echo "DLL location: `$INST/bin/"
ls -lh "`$INST/bin/"*.dll 2>/dev/null || true
"@

$TmpScript = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.sh'
[System.IO.File]::WriteAllText($TmpScript, $BuildSh, (New-Object System.Text.UTF8Encoding $false))
# Convert path to MSYS2 form: C:\...     /c/...
$TmpScriptMsys = ConvertTo-Msys2Path $TmpScript

Write-Host "`nRunning build (first run takes 15-40 min)..." -ForegroundColor Cyan
& $Bash -lc "bash '$TmpScriptMsys'"
if ($LASTEXITCODE -ne 0) {
    Remove-Item $TmpScript -Force -ErrorAction SilentlyContinue
    throw "Build failed - see output above."
}
Remove-Item $TmpScript -Force -ErrorAction SilentlyContinue

#        Copy output to NuGet runtime directory                                                                                                             

Write-Host "`nCopying DLL to $OutputDir..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# MinGW builds produce lib-prefixed names; rename libipopt-3.dll     ipopt-3.dll
# (the P/Invoke target name expected by .NET)
$BuiltIpopt = "C:\ipopt-install\bin\libipopt-3.dll"

if (-not (Test-Path $BuiltIpopt)) {
    throw "Could not find libipopt-3.dll after build.`nExpected: $BuiltIpopt`nCheck build output above for errors."
}

Copy-Item $BuiltIpopt "$OutputDir\ipopt-3.dll" -Force
Write-Host ("  ipopt-3.dll   {0} MB" -f [math]::Round((Get-Item "$OutputDir\ipopt-3.dll").Length / 1MB, 1)) -ForegroundColor Green

# Remove stale DLLs from previous builds (MKL component DLLs no longer needed)
foreach ($dll in @("libcoinmumps-3.dll","libgcc_s_seh-1.dll","libwinpthread-1.dll",
                   "coinmumps-3.dll","libifcoremd.dll","libiomp5md.dll","libmmd.dll","svml_dispmd.dll",
                   "mkl_rt.2.dll","mkl_core.2.dll","mkl_intel_thread.2.dll","mkl_def.2.dll")) {
    $p = Join-Path $OutputDir $dll
    if (Test-Path $p) {
        Remove-Item $p -Force
        Write-Host "  Removed $dll" -ForegroundColor DarkGray
    }
}

$TotalMb = [math]::Round((Get-ChildItem $OutputDir -Filter "*.dll" | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Host "`nDone!  Total bundle: $TotalMb MB" -ForegroundColor Green
Write-Host "  $OutputDir\" -ForegroundColor Green

if ($MklLibDir) {
    Write-Host ""
    Write-Host "MKL Pardiso is statically linked - no external MKL DLLs required." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Built without pardisomkl (Intel oneAPI MKL not available)." -ForegroundColor Yellow
    Write-Host "To include pardisomkl: winget install Intel.oneMKL, then re-run this script." -ForegroundColor Yellow
}
