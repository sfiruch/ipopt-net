# Build IPOPT 3.14.19 for Linux x64 via WSL — self-contained shared library
#
# Thin PowerShell wrapper around build-ipopt-linux64.sh. Invokes the bash
# script inside WSL (Ubuntu) and copies the result to the NuGet runtime dir.
# Mirrors the structure and UX of build-ipopt-win64.ps1.
#
# Prerequisites:
#   WSL2 with Ubuntu (any recent LTS).
#   The bash script installs Intel oneAPI MKL and build tools automatically
#   via apt if they are not already present.
#
# Result:
#   IpoptNet\runtimes\linux-x64\native\libipopt-3.so (~80-120 MB)
#   MUMPS + MKL Pardiso + all GCC/Fortran runtimes statically linked.
#   No external .so files required at runtime.

$ErrorActionPreference = "Stop"

$ScriptDir  = $PSScriptRoot
$RepoRoot   = Split-Path $ScriptDir -Parent
$OutputDir  = "$RepoRoot\IpoptNet\runtimes\linux-x64\native"

# ── Convert Windows path → WSL /mnt/... path ─────────────────────────────────
function ConvertTo-WslPath([string] $winPath) {
    if ($winPath -match '^([A-Za-z]):\\(.*)$') {
        '/mnt/' + $Matches[1].ToLower() + '/' + ($Matches[2] -replace '\\', '/')
    } else {
        $winPath -replace '\\', '/'
    }
}

$WslScriptPath = ConvertTo-WslPath "$ScriptDir\build-ipopt-linux64.sh"
$WslOutputDir  = ConvertTo-WslPath $OutputDir

# ── Verify WSL is available ───────────────────────────────────────────────────
try {
    $null = wsl --status 2>$null
} catch {
    throw "WSL is not available. Install WSL2 with Ubuntu: wsl --install"
}

# ── Run the build script inside WSL ──────────────────────────────────────────
# WSLENV /p flag: WSL translates the Windows path to its /mnt/... form,
# handling spaces in the path (e.g. "OneDrive - iru.ch") safely.
Write-Host "`nRunning build inside WSL (first run takes 15-40 min)..." -ForegroundColor Cyan

$savedWslEnv = $env:WSLENV
$env:WSLENV = "IPOPT_LINUX64_OUTPUT/p"
$env:IPOPT_LINUX64_OUTPUT = $OutputDir
try {
    wsl -d Ubuntu-22.04 -- bash "$WslScriptPath"
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed (exit code $LASTEXITCODE) - see output above."
    }
} finally {
    $env:WSLENV = $savedWslEnv
    Remove-Item Env:IPOPT_LINUX64_OUTPUT -ErrorAction SilentlyContinue
}

# ── Report ────────────────────────────────────────────────────────────────────
$so = Get-Item "$OutputDir\libipopt-3.so" -ErrorAction SilentlyContinue
if (-not $so) {
    throw "libipopt-3.so was not found in $OutputDir after build."
}

$totalMb = [math]::Round($so.Length / 1MB, 1)
Write-Host "`nDone!  $totalMb MB" -ForegroundColor Green
Write-Host "  $OutputDir\" -ForegroundColor Green
Write-Host ""
Write-Host "MKL Pardiso is statically linked - no external MKL .so files required." -ForegroundColor Green
