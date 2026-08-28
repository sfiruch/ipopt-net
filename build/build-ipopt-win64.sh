#!/usr/bin/env bash
# Build IPOPT 3.14.20 for Windows x64 — one self-contained DLL, cross-compiled
# from Linux/WSL. No MSYS2, no Visual Studio, no Intel Fortran.
#
# Compilers:
#   C/C++   - x86_64-w64-mingw32-gcc / g++   (Ubuntu's mingw-w64 cross toolchain)
#   Fortran - x86_64-w64-mingw32-gfortran    (for MUMPS)
#
# All GCC/Fortran runtimes are linked statically, so the result needs no
# libgcc_s_seh-1.dll / libgfortran-5.dll / libwinpthread-1.dll at runtime — and
# no Microsoft VC++ redistributable either, since nothing is compiled by MSVC.
#
# Intel MKL Pardiso:
#   Uses the *Windows* static archives from the host's Intel oneAPI install,
#   read through /mnt/c. mkl_intel_lp64.lib + mkl_sequential.lib + mkl_core.lib
#   are MSVC COFF archives; MinGW's ld accepts them for C-interface functions
#   (pardiso, pardisoinit), with msvc_compat.c supplying the MSVC-only runtime
#   helpers those objects reference. MKL also provides BLAS/LAPACK, so no
#   separate netlib build is needed. Same approach as the Linux build.
#
# Building from the same coinbrew sources as build-ipopt-linux64.sh keeps the
# two platforms on identical IPOPT and MUMPS versions.
#
# Usage (from a native Linux shell, or from Windows via WSL):
#   ./build-ipopt-win64.sh [--output /path/to/output/dir]
#   wsl [-d <distro>] bash build/build-ipopt-win64.sh
#   Default output: <repo>/IpoptNet/runtimes/win-x64/native/
#
# Prerequisites:
#   The mingw-w64 cross toolchain, which needs sudo to install:
#     sudo apt-get install -y gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 \
#                             gfortran-mingw-w64-x86-64
#   On a distro where sudo prompts for a password, install it beforehand.
#
#   Intel oneAPI MKL for *Windows* must be present on the host; this script
#   reads its static .lib archives through /mnt/c (winget install Intel.oneMKL).
#   Note this is the Windows MKL, not the Linux one build-ipopt-linux64.sh uses,
#   so the two builds can want different distros — pick with `wsl -d`.
#
# Result:
#   ipopt-3.dll (~80 MB) — MUMPS + MKL Pardiso + all runtimes statically linked

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$HOME/ipopt-src-win"
INSTALL_DIR="$HOME/ipopt-install-win"
IPOPT_RELEASE="releases/3.14.20"
HOST=x86_64-w64-mingw32
NPROC=$(nproc 2>/dev/null || echo 4)

OUTPUT_DIR="${IPOPT_WIN64_OUTPUT:-$SCRIPT_DIR/../IpoptNet/runtimes/win-x64/native}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Toolchain ─────────────────────────────────────────────────────────────────
for t in $HOST-gcc $HOST-g++ $HOST-gfortran; do
    command -v "$t" >/dev/null || {
        echo "ERROR: $t not found. Install the cross toolchain with:" >&2
        echo "  sudo apt-get install -y gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 gfortran-mingw-w64-x86-64" >&2
        exit 1
    }
done
echo "Compiler versions:"
$HOST-gcc --version | head -1
$HOST-gfortran --version | head -1

# ── Intel oneAPI MKL (Windows static archives, via /mnt/c) ────────────────────
# Autotools chokes on paths containing spaces ("Program Files (x86)"), so the
# three archives are copied into a space-free directory inside the WSL
# filesystem. That also avoids repeatedly reading ~490 MB across the 9p mount.
find_mkl_win_dir() {
    shopt -s nullglob
    for base in "/mnt/c/Program Files (x86)/Intel/oneAPI/mkl" \
                "/mnt/c/Program Files/Intel/oneAPI/mkl"; do
        [[ -d "$base" ]] || continue
        for ver in "$base"/latest "$base"/*; do
            for sub in lib lib/intel64; do
                [[ -f "$ver/$sub/mkl_intel_lp64.lib" ]] && { echo "$ver/$sub"; return 0; }
            done
        done
    done
    return 1
}

if ! MKL_WIN_DIR=$(find_mkl_win_dir); then
    echo "ERROR: Intel oneAPI MKL (Windows) not found under /mnt/c." >&2
    echo "Install it on the Windows side: winget install Intel.oneMKL" >&2
    exit 1
fi
echo "Intel oneAPI MKL (Windows): $MKL_WIN_DIR"

MKL_LOCAL="$SOURCE_DIR/mkl-win"
mkdir -p "$MKL_LOCAL"
for lib in mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib; do
    if [[ ! -f "$MKL_LOCAL/$lib" ]] || \
       [[ "$MKL_WIN_DIR/$lib" -nt "$MKL_LOCAL/$lib" ]]; then
        echo "  copying $lib ..."
        cp "$MKL_WIN_DIR/$lib" "$MKL_LOCAL/$lib"
    fi
done

# ── Link flags ────────────────────────────────────────────────────────────────
GCC_LIB="$(dirname "$($HOST-gcc -print-file-name=libgcc.a)")"
MINGW_LIB="$(dirname "$($HOST-gcc -print-file-name=libmsvcrt.a)")"

# Two libtool behaviours have to be worked around here:
#
#   * Plain -static makes libtool conclude a shared library is impossible, so it
#     silently builds libipopt.a and no DLL at all - with make and make install
#     both still reporting success. Never pass it.
#   * A bare .a path on the link line is dropped ("the file extensions .a of
#     this argument makes me believe that it is just a static archive that I
#     should not use here"), which would quietly leave the runtimes dynamic.
#     Wrapping each in -Xlinker hands it straight to ld, past that heuristic -
#     the same trick the MKL archives need below.
#
# libgcc_eh is the static-only SEH archive; pulling it (rather than -lgcc_s) is
# what keeps libgcc_s_seh-1.dll out of the result. -Wl,-s strips symbols.
STATIC_RUNTIMES="-Wl,--start-group \
  -Xlinker $GCC_LIB/libgcc_eh.a \
  -Xlinker $GCC_LIB/libgcc.a \
  -Xlinker $GCC_LIB/libstdc++.a \
  -Xlinker $GCC_LIB/libgfortran.a \
  -Xlinker $GCC_LIB/libquadmath.a \
  -Xlinker $MINGW_LIB/libwinpthread.a \
  -Wl,--end-group -Wl,-s"

# msvc_compat is wrapped in an archive rather than passed as a bare .o: libtool
# refuses non-libtool objects when linking libcoinmumps.la ("cannot build
# libtool library ... from non-libtool objects on this host"). It sits inside
# the --start-group so its symbols are rescanned along with the MKL archives
# that reference them.
#
# mkl_sequential: threading-free layer, no OpenMP/TBB dependency.
# mkl_intel_lp64: LP64 interface (32-bit integers, matching IPOPT's default).
# mkl_core: main compute library. Circular deps -> --start-group.
# Libtool strips bare .lib paths from a --start-group block (it does not
# recognise the extension), so each one is wrapped in -Xlinker to pass it
# straight through to ld. libmsvcrt/libm/libmingwex/libkernel32 are rescanned
# inside the same group because libtool otherwise reorders -l flags ahead of
# the MKL archives.
# --allow-multiple-definition: MKL bundles CRT symbols (vfprintf, _vsnprintf)
# that also live in libmsvcrt.a; keep the first (msvcrt.dll) definition.
MSVC_COMPAT_A="$SOURCE_DIR/libmsvccompat.a"
MKL_LFLAGS="-Wl,--allow-multiple-definition -Wl,--start-group \
  -Xlinker $MSVC_COMPAT_A \
  -Xlinker $MKL_LOCAL/mkl_intel_lp64.lib \
  -Xlinker $MKL_LOCAL/mkl_sequential.lib \
  -Xlinker $MKL_LOCAL/mkl_core.lib \
  -Xlinker $MINGW_LIB/libmsvcrt.a \
  -Xlinker $MINGW_LIB/libm.a \
  -Xlinker $MINGW_LIB/libmingwex.a \
  -Xlinker $MINGW_LIB/libkernel32.a \
  -Wl,--end-group"

# IPOPT additionally needs MUMPS. libtool will not inline a static-only .la into
# a DLL on this host ("This system cannot link to static lib archive
# libcoinmumps.la ... I can only do this if you have a shared version"), so the
# archive is handed to ld directly, ahead of MKL because MUMPS calls BLAS/LAPACK.
# This is kept separate from MKL_LFLAGS because MUMPS's own configure runs
# before the archive exists and would fail its link test on the missing file.
#
# The GCC/Fortran runtimes are repeated here, inside the group. LDFLAGS is
# placed to the left of these flags on the link line, and ld resolves archives
# left to right — so libgfortran.a there is scanned *before* MUMPS is even
# read, and MUMPS's _gfortran_* references would go unresolved. Inside the
# group they are rescanned until every reference is satisfied.
IPOPT_LFLAGS="-Wl,--allow-multiple-definition -Wl,--start-group \
  -Xlinker $INSTALL_DIR/lib/libcoinmumps.a \
  -Xlinker $MSVC_COMPAT_A \
  -Xlinker $MKL_LOCAL/mkl_intel_lp64.lib \
  -Xlinker $MKL_LOCAL/mkl_sequential.lib \
  -Xlinker $MKL_LOCAL/mkl_core.lib \
  -Xlinker $GCC_LIB/libgfortran.a \
  -Xlinker $GCC_LIB/libquadmath.a \
  -Xlinker $GCC_LIB/libstdc++.a \
  -Xlinker $GCC_LIB/libgcc_eh.a \
  -Xlinker $GCC_LIB/libgcc.a \
  -Xlinker $MINGW_LIB/libwinpthread.a \
  -Xlinker $MINGW_LIB/libmingw32.a \
  -Xlinker $MINGW_LIB/libmsvcrt.a \
  -Xlinker $MINGW_LIB/libm.a \
  -Xlinker $MINGW_LIB/libmingwex.a \
  -Xlinker $MINGW_LIB/libkernel32.a \
  -Wl,--end-group"

# ── coinbrew + fetch IPOPT ────────────────────────────────────────────────────
mkdir -p "$SOURCE_DIR"
cd "$SOURCE_DIR"

if [[ ! -f coinbrew/coinbrew ]]; then
    git clone --depth=1 https://github.com/coin-or/coinbrew.git coinbrew
fi
COINBREW="$SOURCE_DIR/coinbrew/coinbrew"
chmod +x "$COINBREW"

if [[ ! -d Ipopt ]]; then
    "$COINBREW" fetch Ipopt --release="$IPOPT_RELEASE" --no-prompt
fi

COIN_PKGCFG="$INSTALL_DIR/lib/pkgconfig"

# ── MSVC compatibility shims (compiled with the cross compiler) ──────────────
$HOST-gcc -O2 -march=x86-64 -c "$SCRIPT_DIR/msvc_compat.c" -o "$SOURCE_DIR/msvc_compat.o"
rm -f "$MSVC_COMPAT_A"
$HOST-ar rcs "$MSVC_COMPAT_A" "$SOURCE_DIR/msvc_compat.o"

# ── Phase 1: MUMPS, static-only ──────────────────────────────────────────────
# With --disable-shared libtool installs only libcoinmumps.a. When IPOPT links
# as a shared lib against a static-only dep, libtool inlines the archive, so
# all MUMPS code ends up inside ipopt-3.dll.
if [[ ! -f "$INSTALL_DIR/lib/libcoinmumps.a" ]]; then
    rm -rf "$SOURCE_DIR/build-mumps"
    mkdir -p "$SOURCE_DIR/build-mumps"
    cd "$SOURCE_DIR/build-mumps"
    PKG_CONFIG_PATH="$COIN_PKGCFG" \
    "$SOURCE_DIR/ThirdParty/Mumps/configure" \
        --host="$HOST" --prefix="$INSTALL_DIR" \
        CC=$HOST-gcc FC=$HOST-gfortran \
        CFLAGS="-O2 -march=x86-64 -DNDEBUG" FFLAGS="-O2 -march=x86-64" \
        --disable-shared --enable-static \
        --with-lapack-lflags="$MKL_LFLAGS"
    make -j"$NPROC"
    make install
fi

# ── Phase 2: configure IPOPT ─────────────────────────────────────────────────
# MKL supplies both LAPACK and pardiso(); configure detects the latter and
# enables pardisomkl.
#
# sIPOPT and the Java interface are disabled: IpoptNet binds to neither, and
# sIPOPT's link pulls in libstdc++.dll.a alongside our static libstdc++.a,
# which collides ("multiple definition of std::__throw_logic_error"). Skipping
# it avoids the conflict and trims the build.
rm -rf "$SOURCE_DIR/build-ipopt"
mkdir -p "$SOURCE_DIR/build-ipopt"
cd "$SOURCE_DIR/build-ipopt"

PKG_CONFIG_PATH="$COIN_PKGCFG" \
"$SOURCE_DIR/Ipopt/configure" \
    --host="$HOST" --prefix="$INSTALL_DIR" \
    CC=$HOST-gcc CXX=$HOST-g++ FC=$HOST-gfortran \
    CFLAGS="-O2 -march=x86-64 -DNDEBUG" \
    CXXFLAGS="-O2 -march=x86-64 -DNDEBUG" \
    FFLAGS="-O2 -march=x86-64" \
    LDFLAGS="$STATIC_RUNTIMES" \
    --enable-shared --disable-static --without-asl \
    --disable-sipopt --disable-java \
    --with-lapack-lflags="$IPOPT_LFLAGS"

# ── Phase 3: patch libtool ───────────────────────────────────────────────────
# configure probes the C++ compiler and bakes its runtime libs into postdeps,
# which forces dynamic imports regardless of LDFLAGS:
#   -lgcc_s   -> libgcc_s_seh-1.dll.  -lgcc_eh is the static-only SEH archive,
#               so substituting it pulls in the .a instead.
#   -lstdc++  -> libstdc++-6.dll, because -lstdc++ finds libstdc++.dll.a (the
#               import library) ahead of our static libstdc++.a. Naming the
#               archive by absolute path leaves no import stub to resolve.
#
# Note the asymmetry in the two patterns: -lgcc_s takes a \b because 's' is a
# word character, but -lstdc++ must not, since '+' is not — there is no word
# boundary between '+' and the following space, so -lstdc++\b matches nothing.
find "$SOURCE_DIR/build-ipopt" -name libtool -exec sed -i \
    -e 's/-lgcc_s\b/-lgcc_eh/g' \
    -e "s|-lstdc++|$GCC_LIB/libstdc++.a|g" {} \;

# ── Phase 4: build and install ───────────────────────────────────────────────
make -j"$NPROC"
make install

# ── Phase 5: copy to output ──────────────────────────────────────────────────
# MinGW names the import-style output libipopt-3.dll; .NET's P/Invoke expects
# ipopt-3.dll.
BUILT=""
for cand in "$INSTALL_DIR/bin/libipopt-3.dll" "$INSTALL_DIR/lib/libipopt-3.dll"; do
    [[ -f "$cand" ]] && BUILT="$cand" && break
done
[[ -n "$BUILT" ]] || { echo "ERROR: libipopt-3.dll not found after build." >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
# The csproj globs *.dll from this directory, so stale files from an earlier
# build (e.g. the official release's companion DLLs) must not linger.
rm -f "$OUTPUT_DIR"/*.dll
cp "$BUILT" "$OUTPUT_DIR/ipopt-3.dll"

SIZE_MB=$(stat -c%s "$OUTPUT_DIR/ipopt-3.dll" | awk '{printf "%.1f", $1/1024/1024}')
echo ""
echo "Build complete."
echo "  ipopt-3.dll   $SIZE_MB MB"
echo "  $OUTPUT_DIR/"

# ── Verification ─────────────────────────────────────────────────────────────
# Dump once: piping objdump/nm straight into `grep -q` lets the reader exit
# first, and under `pipefail` the resulting SIGPIPE would fail the pipeline
# even on a successful match.
IMPORTS=$($HOST-objdump -p "$OUTPUT_DIR/ipopt-3.dll" 2>/dev/null | grep -i 'DLL Name:' | sort -u || true)

echo ""
echo "DLL imports (must be system DLLs only — no libgcc/libgfortran/MSVCP140):"
sed 's/^/  /' <<< "$IMPORTS"

LEAKED=$(grep -iE 'libgcc|libgfortran|libwinpthread|libquadmath|libstdc|MSVCP140|VCRUNTIME|libifcore|libiomp|libmmd|svml_disp|coinmumps' <<< "$IMPORTS" || true)
if [[ -n "$LEAKED" ]]; then
    echo ""
    echo "ERROR: the DLL is not self-contained — it imports:"
    sed 's/^/  /' <<< "$LEAKED"
    exit 1
fi

# The DLL is stripped (-Wl,-s), so nm shows nothing; check the export table and
# the embedded strings instead. "pardisomkl" is the option value IPOPT registers
# only when IpPardisoMKLSolverInterface is compiled in, and the MKL_PARDISO_*
# diagnostics come from MKL's own Pardiso objects — together they show both
# halves are actually linked, not merely detected by configure.
PARDISO_EXPORTS=$($HOST-objdump -p "$OUTPUT_DIR/ipopt-3.dll" 2>/dev/null | grep -ci pardiso || true)
PARDISO_OPTION=$(strings -a "$OUTPUT_DIR/ipopt-3.dll" | grep -cx 'pardisomkl' || true)
MKL_PARDISO=$(strings -a "$OUTPUT_DIR/ipopt-3.dll" | grep -ci 'MKL_PARDISO' || true)
echo ""
echo "Pardiso: $PARDISO_EXPORTS export(s), pardisomkl option $PARDISO_OPTION, MKL Pardiso strings $MKL_PARDISO"
if [[ "$PARDISO_OPTION" -eq 0 || "$MKL_PARDISO" -eq 0 ]]; then
    echo "ERROR: MKL Pardiso is not linked in — check configure's pardisomkl detection." >&2
    exit 1
fi

if (( $(echo "$SIZE_MB > 200" | bc -l) )); then
    echo "WARNING: DLL is larger than 200 MB — dead-stripping may not be working."
elif (( $(echo "$SIZE_MB < 40" | bc -l) )); then
    echo "WARNING: DLL is smaller than 40 MB — MKL Pardiso may not be fully linked."
fi

echo ""
echo "MKL Pardiso is statically linked — no external DLLs required."
