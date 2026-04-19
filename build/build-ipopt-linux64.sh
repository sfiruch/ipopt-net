#!/usr/bin/env bash
# Build IPOPT 3.14.19 for Linux x64 — small, self-contained shared library
#
# Compilers:
#   C/C++   - GCC/G++        system or Ubuntu toolchain
#   Fortran - GFortran        for MUMPS and LAPACK
#
# All Fortran/GCC runtimes are statically linked into libipopt-3.so so no
# extra runtime .so files are needed.
#
# Intel MKL Pardiso support:
#   Uses static MKL archives from Intel oneAPI MKL (intel-oneapi-mkl-devel).
#   libmkl_intel_lp64.a + libmkl_sequential.a + libmkl_core.a are ELF archives;
#   GNU ld accepts them natively in --start-group/--end-group.
#   The resulting libipopt-3.so has MKL baked in — no separate MKL .so needed.
#       Same approach as the Windows build (statically links mkl_*.lib).
#
# Usage (from a native Linux shell or WSL):
#   ./build-ipopt-linux64.sh [--output /path/to/output/dir]
#   Default output: <repo>/IpoptNet/runtimes/linux/native/
#
# Result:
#   libipopt-3.so (~80-120 MB) — MUMPS + MKL Pardiso + Fortran runtime statically linked
#   Runtime deps: libstdc++.so.6, libgcc_s.so.1 (standard on all Linux distros)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$HOME/ipopt-src"
INSTALL_DIR="$HOME/ipopt-install"
IPOPT_RELEASE="releases/3.14.19"
NPROC=$(nproc 2>/dev/null || echo 4)

# Allow override via env var (set by build-ipopt-linux64.ps1 via WSLENV) or argument
OUTPUT_DIR="${IPOPT_LINUX64_OUTPUT:-$SCRIPT_DIR/IpoptNet/runtimes/linux/native}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────

echo "Compiler versions:"
gcc    --version | head -1
gfortran --version | head -1

# ── Intel oneAPI MKL (static linking) ─────────────────────────────────────────
#
# Intel oneAPI MKL provides static .a archives for build-time linking.
# libmkl_intel_lp64.a + libmkl_sequential.a + libmkl_core.a are linked into
# libipopt-3.so — no separate MKL .so required at runtime.
#
# Threading: libmkl_sequential.a (single-threaded, no OpenMP dependency).
# Integer size: libmkl_intel_lp64.a (32-bit indices, matching IPOPT's default).
# IPOPT's IpPardisoMKLSolverInterface.cpp declares pardiso/pardisoinit via
# extern "C", so no MKL header is needed during compilation.

find_mkl_lib_dir() {
    shopt -s nullglob
    for candidate in \
        /opt/intel/oneapi/mkl/latest/lib/intel64 \
        /opt/intel/oneapi/mkl/latest/lib \
        /opt/intel/oneapi/mkl/*/lib/intel64 \
        /opt/intel/oneapi/mkl/*/lib \
        /usr/lib/x86_64-linux-gnu; do
        [[ -f "$candidate/libmkl_intel_lp64.a" ]] && echo "$candidate" && return 0
    done
    return 1
}

if ! MKL_LIB_DIR=$(find_mkl_lib_dir); then
    echo ""
    echo "Intel oneAPI MKL not found - installing intel-oneapi-mkl-devel..."
    echo "  (Provides ~500 MB of static libraries for pardisomkl support.)"
    # Use SUDO_ASKPASS / sudo -A if available, otherwise fall back to interactive sudo.
    # Download key to file first to avoid hanging wget|sudo pipe.
    curl -fsSL --max-time 60 -o /tmp/intel-key.pub \
        https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    gpg --dearmor /tmp/intel-key.pub
    sudo install -m 644 /tmp/intel-key.pub.gpg /usr/share/keyrings/intel-oneapi-keyring.gpg
    sudo bash -c "echo 'deb [signed-by=/usr/share/keyrings/intel-oneapi-keyring.gpg] https://apt.repos.intel.com/oneapi all main' > /etc/apt/sources.list.d/oneAPI.list"
    sudo apt-get update -q
    sudo apt-get install -y --no-install-recommends intel-oneapi-mkl-devel
    MKL_LIB_DIR=$(find_mkl_lib_dir)
fi

echo "Intel oneAPI MKL: $MKL_LIB_DIR"
echo "  pardisomkl will be statically linked into libipopt-3.so."

# ── Build dependencies ────────────────────────────────────────────────────────
if ! command -v gfortran &>/dev/null || ! command -v git &>/dev/null || ! command -v pkg-config &>/dev/null; then
    sudo apt-get install -y --no-install-recommends \
        build-essential gfortran git wget curl pkg-config
fi

# ── Runtime link flags ────────────────────────────────────────────────────────
# libgfortran.a objects use R_X86_64_TPOFF32 (async.o) and R_X86_64_PC32
# against libc symbols (unix.o), both of which the linker rejects in shared
# libs. We work around this by:
#   1. Building libgfortran_pic.a: remove async.o and unix.o from the archive,
#      add unix.o back as unix_patched.o (same content, renamed so the linker
#      treats it as a new member).
#   2. Compiling glibc_compat.c (repo root) which provides:
#      - _gfortrani_flush_if_preconnected as a no-op, satisfying the symbol
#        before unix_patched.o is pulled in. --gc-sections then dead-strips
#        unix_patched.o's copy of the function (and its PC32 relocs).
#      - thread_unit as __thread void* (PIC-compatible TLS, replaces async.o).
#      - Stubs for all async I/O functions MUMPS never calls.
#      - glibc version pins so the .so runs on systems older than Ubuntu 24.04.
# libstdc++.so.6 and libgcc_s.so.1 are kept dynamic (standard on all distros).
# -Wl,-s: strip symbol table from the final .so.
LIBGFORTRAN_ORIG=$(gfortran -print-file-name=libgfortran.a)
LIBGFORTRAN_PIC="$SOURCE_DIR/libgfortran_pic.a"
if [[ ! -f "$LIBGFORTRAN_PIC" ]]; then
    cp "$LIBGFORTRAN_ORIG" "$LIBGFORTRAN_PIC"
    ar d "$LIBGFORTRAN_PIC" async.o
    TMPDIR_FORT=$(mktemp -d)
    (cd "$TMPDIR_FORT" && ar x "$LIBGFORTRAN_ORIG" unix.o && \
     ar d "$LIBGFORTRAN_PIC" unix.o && \
     cp unix.o unix_patched.o && ar q "$LIBGFORTRAN_PIC" unix_patched.o)
    rm -rf "$TMPDIR_FORT"
fi
GLIBC_COMPAT_O="$SOURCE_DIR/glibc_compat.o"
gcc -O2 -fPIC -c "$SCRIPT_DIR/glibc_compat.c" -o "$GLIBC_COMPAT_O"
STATIC_RUNTIMES="-Wl,-s"

# mkl_sequential: threading-free layer; avoids OpenMP/TBB dependencies.
# mkl_intel_lp64: LP64 interface (32-bit integers, matching IPOPT's default).
# mkl_core: main compute library. All three have circular deps — --start-group.
MKL_LFLAGS="-Wl,--start-group \
  $MKL_LIB_DIR/libmkl_intel_lp64.a \
  $MKL_LIB_DIR/libmkl_sequential.a \
  $MKL_LIB_DIR/libmkl_core.a \
  -Wl,--end-group -lpthread -lm -ldl"

# ── coinbrew + fetch IPOPT ────────────────────────────────────────────────────
mkdir -p "$SOURCE_DIR"
cd "$SOURCE_DIR"

if [[ ! -f coinbrew/coinbrew ]]; then
    git clone --depth=1 https://github.com/coin-or/coinbrew.git coinbrew
fi
COINBREW="$SOURCE_DIR/coinbrew/coinbrew"
chmod +x "$COINBREW"

# coinbrew fetch downloads: Ipopt, ThirdParty-Mumps (+ MUMPS source),
# ThirdParty-Lapack (+ Netlib LAPACK). --skip-update avoids re-fetching on re-runs.
if [[ ! -d Ipopt ]]; then
    "$COINBREW" fetch Ipopt --release="$IPOPT_RELEASE" --no-prompt
fi

COIN_PKGCFG="$INSTALL_DIR/lib/pkgconfig"

# ── Phase 1: Build MUMPS as static-only ──────────────────────────────────────
# With --disable-shared, libtool installs only libcoinmumps.a (no .so.a).
# When IPOPT links as a shared lib against a static-only dep, libtool inlines
# the archive — so all MUMPS code ends up inside libipopt-3.so.
if [[ ! -f "$INSTALL_DIR/lib/libcoinmumps.a" ]]; then
    mkdir -p "$SOURCE_DIR/build-mumps"
    cd "$SOURCE_DIR/build-mumps"
    PKG_CONFIG_PATH="$COIN_PKGCFG" \
    "$SOURCE_DIR/ThirdParty/Mumps/configure" \
        --prefix="$INSTALL_DIR" CC=gcc FC=gfortran \
        CFLAGS="-O2 -march=x86-64 -DNDEBUG -ffunction-sections -fdata-sections" \
        FFLAGS="-O2 -march=x86-64            -ffunction-sections -fdata-sections" \
        LDFLAGS="$STATIC_RUNTIMES" \
        --with-lapack-lflags="$MKL_LFLAGS" \
        --disable-shared --enable-static
    make -j"$NPROC" && make install
fi

# ── Phase 2: Configure IPOPT ──────────────────────────────────────────────────
# --with-lapack-lflags provides MKL static archives; configure auto-detects
# pardiso() in libmkl_core.a and enables pardisomkl.
# -ffunction-sections + -fdata-sections + --gc-sections: dead-strip MKL routines
# that the Pardiso call path never reaches (~200+ MB in archives → ~80 MB .so).
# --exclude-libs,ALL: archive symbols are not exported from the .so symbol table.
rm -rf "$SOURCE_DIR/build-ipopt"
mkdir -p "$SOURCE_DIR/build-ipopt"
cd "$SOURCE_DIR/build-ipopt"

PKG_CONFIG_PATH="$COIN_PKGCFG" \
"$SOURCE_DIR/Ipopt/configure" \
    --prefix="$INSTALL_DIR" CC=gcc CXX=g++ FC=gfortran \
    CFLAGS="-O2   -march=x86-64 -DNDEBUG -ffunction-sections -fdata-sections" \
    CXXFLAGS="-O2 -march=x86-64 -DNDEBUG -ffunction-sections -fdata-sections" \
    FFLAGS="-O2   -march=x86-64            -ffunction-sections -fdata-sections" \
    LDFLAGS="$STATIC_RUNTIMES -Wl,--gc-sections -Wl,--exclude-libs,ALL" \
    --enable-shared --disable-static --without-asl \
    --with-lapack-lflags="$MKL_LFLAGS"

# ── Phase 3: Patch libtool ────────────────────────────────────────────────────
# configure probes the C++ compiler and may bake -lgcc_s into postdeps, which
# forces dynamic loading of libgcc_s.so.1 regardless of -static-libgcc.
# Replace with the static equivalent so the final .so has no dynamic libgcc dep.
find "$SOURCE_DIR/build-ipopt" -name "libtool" \
    -exec sed -i 's/-lgcc_s\b/-lgcc/g' {} \;

# ── Phase 4: Build and install ────────────────────────────────────────────────
make -j"$NPROC"
make install

# ── Phase 4.5: Relink libipopt.so with properly grouped MKL ──────────────────
# libtool strips --start-group/--end-group from LIBADD when building a shared
# library, leaving MKL archives in a plain left-to-right link pass. This causes
# circular-dependency symbols (mkl_pds_*_omp_pardiso defined in libmkl_sequential.a
# but referenced by libmkl_core.a objects pulled in after sequential was scanned)
# to remain UNDEFINED. Fix: bypass libtool and relink the .so directly with g++,
# which preserves the --start-group grouping.
IPOPT_OBJS=$(find "$SOURCE_DIR/build-ipopt/src" -path '*/.libs/*.o' | sort | tr '\n' ' ')
g++ -shared -fPIC \
    $IPOPT_OBJS \
    "$INSTALL_DIR/lib/libcoinmumps.a" \
    -Wl,--start-group \
      "$MKL_LIB_DIR/libmkl_intel_lp64.a" \
      "$MKL_LIB_DIR/libmkl_sequential.a" \
      "$MKL_LIB_DIR/libmkl_core.a" \
    -Wl,--end-group \
    "$GLIBC_COMPAT_O" \
    "$LIBGFORTRAN_PIC" \
    -lpthread -ldl -lm \
    -Wl,-s -Wl,--gc-sections -Wl,--exclude-libs,ALL \
    -Wl,-soname,libipopt.so.3 \
    -march=x86-64 -O2 \
    -o "$INSTALL_DIR/lib/libipopt.so.3.14.19"
# Update the symlinks to point at the fresh binary
ln -sf libipopt.so.3.14.19 "$INSTALL_DIR/lib/libipopt.so.3"
ln -sf libipopt.so.3.14.19 "$INSTALL_DIR/lib/libipopt.so"

# ── Phase 5: Copy to output ───────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
# -L follows the versioned symlink and writes the actual ELF file
cp -L "$INSTALL_DIR/lib/libipopt.so" "$OUTPUT_DIR/libipopt-3.so"

echo ""
echo "Build complete."
SIZE_MB=$(stat -c%s "$OUTPUT_DIR/libipopt-3.so" | awk '{printf "%.1f", $1/1024/1024}')
echo "  libipopt-3.so   $SIZE_MB MB"
echo "  $OUTPUT_DIR/"

echo ""
echo "Unexpected dynamic dependencies (should list only system libs):"
ldd "$OUTPUT_DIR/libipopt-3.so" \
    | grep -v -E 'linux-vdso|libdl|libm\.so|libgcc_s|libstdc\+\+|libc\.so|libpthread|ld-linux' \
    || echo "  (none)"

echo ""
echo "Pardiso symbols (should be non-empty):"
nm -D "$OUTPUT_DIR/libipopt-3.so" 2>/dev/null | grep -i pardiso | head -5 \
    || echo "  WARNING: no Pardiso symbols found — pardisomkl may not be linked"

if ! nm -D "$OUTPUT_DIR/libipopt-3.so" 2>/dev/null | grep -qi pardiso; then
    echo ""
    echo "ERROR: Pardiso symbols not found in the built library."
    echo "Check configure output for 'pardisomkl' detection status."
    exit 1
fi

echo ""
echo "Undefined MKL symbols (should be empty — would indicate incomplete MKL link):"
nm -D "$OUTPUT_DIR/libipopt-3.so" 2>/dev/null | grep ' U mkl_' | head -5 \
    || echo "  (none — good)"

echo ""
if (( $(echo "$SIZE_MB > 200" | bc -l) )); then
    echo "WARNING: .so is larger than 200 MB — dead-stripping may not be working."
elif (( $(echo "$SIZE_MB < 50" | bc -l) )); then
    echo "WARNING: .so is smaller than 50 MB — MKL Pardiso may not be fully linked."
fi

echo "MKL Pardiso is statically linked — no external MKL .so files required."
