# configure.ps1 — mLLM 환경 자동 감지 및 local.cmake 생성
#
# 사용법: .\configure.ps1
# (또는 setup.bat 을 실행하세요)

$ErrorActionPreference = "Continue"
$outFile = "$PSScriptRoot\local.cmake"

Write-Host ""
Write-Host "=== mLLM 환경 설정 ===" -ForegroundColor Cyan
Write-Host ""

# ── 1. LibTorch 감지 ──────────────────────────────────────────────────────────
$torchDir = $null
$libtorchLibDir = $null

$cudaCandidates = @("C:\libtorch-cuda", "D:\libtorch-cuda", "C:\libtorch_cuda")
$cpuCandidates  = @("C:\libtorch",      "D:\libtorch",      "C:\libtorch_cpu")

foreach ($p in $cudaCandidates) {
    if (Test-Path "$p\share\cmake\Torch") {
        $torchDir      = "$p/share/cmake/Torch"
        $libtorchLibDir = "$p/lib"
        Write-Host "[OK] LibTorch CUDA: $p" -ForegroundColor Green
        break
    }
}
if (-not $torchDir) {
    foreach ($p in $cpuCandidates) {
        if (Test-Path "$p\share\cmake\Torch") {
            $torchDir       = "$p/share/cmake/Torch"
            $libtorchLibDir = "$p/lib"
            Write-Host "[OK] LibTorch CPU: $p" -ForegroundColor Yellow
            Write-Host "     (GPU 없이 실행 — 속도가 매우 느릴 수 있습니다)" -ForegroundColor Yellow
            break
        }
    }
}
if (-not $torchDir) {
    Write-Host "[없음] LibTorch 를 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "       https://pytorch.org/get-started/locally/ 에서 LibTorch 를 다운로드하여"
    Write-Host "       C:\libtorch-cuda (CUDA) 또는 C:\libtorch (CPU) 에 압축 해제하세요."
    Write-Host ""
    $torchDir = "C:/libtorch-cuda/share/cmake/Torch"
    Write-Host "       → 일단 기본값으로 설정합니다 (나중에 local.cmake 를 직접 수정하세요)"
}

# ── 2. CUDA Toolkit 감지 ──────────────────────────────────────────────────────
$cudaRoot    = $null
$nvccPath    = $null
$cudaVersion = $null

$nvidiBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if (Test-Path $nvidiBase) {
    $versions = Get-ChildItem $nvidiBase -Directory | Sort-Object Name -Descending
    if ($versions) {
        $latest = $versions[0]
        $nvcc = "$($latest.FullName)\bin\nvcc.exe"
        if (Test-Path $nvcc) {
            $cudaRoot    = $latest.FullName -replace '\\', '/'
            $nvccPath    = $nvcc -replace '\\', '/'
            $cudaVersion = $latest.Name
            Write-Host "[OK] CUDA: $cudaVersion ($cudaRoot)" -ForegroundColor Green
        }
    }
}
if (-not $cudaRoot) {
    Write-Host "[없음] CUDA Toolkit 를 찾을 수 없습니다 (GPU 없이 빌드됩니다)." -ForegroundColor Yellow
    $cudaRoot = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
    $nvccPath = "$cudaRoot/bin/nvcc.exe"
}

# ── 3. GPU SM 아키텍처 감지 ───────────────────────────────────────────────────
$smArch = "8.6"  # 기본값 (RTX 3060)
try {
    $cap = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
    if ($cap -match "(\d+\.\d+)") {
        $smArch = $matches[1]
        Write-Host "[OK] GPU compute capability: $smArch" -ForegroundColor Green
    }
} catch {}
if ($smArch -eq "8.6") {
    Write-Host "[기본] GPU 아키텍처: sm_86 (RTX 3060 기준, 다른 GPU면 local.cmake 수정)" -ForegroundColor Yellow
}

# ── 4. MSVC 컴파일러 감지 ─────────────────────────────────────────────────────
$msvcCl = $null
$vsBase = "C:\Program Files\Microsoft Visual Studio\2022"
$editions = @("Community", "Professional", "Enterprise", "BuildTools")
foreach ($ed in $editions) {
    $msvcRoot = "$vsBase\$ed\VC\Tools\MSVC"
    if (Test-Path $msvcRoot) {
        $msvcVers = Get-ChildItem $msvcRoot -Directory | Sort-Object Name -Descending
        if ($msvcVers) {
            $cl = "$msvcRoot\$($msvcVers[0].Name)\bin\Hostx64\x64\cl.exe"
            if (Test-Path $cl) {
                $msvcCl = $cl -replace '\\', '/'
                Write-Host "[OK] MSVC: $($msvcVers[0].Name) ($ed)" -ForegroundColor Green
                break
            }
        }
    }
}
if (-not $msvcCl) {
    Write-Host "[없음] Visual Studio 2022 를 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "       https://visualstudio.microsoft.com/ 에서 VS 2022 Community 를 설치하고"
    Write-Host "       'C++ 데스크톱 개발' 워크로드를 선택하세요."
    $msvcCl = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
}

# ── 5. local.cmake 생성 ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "local.cmake 생성 중..." -ForegroundColor Cyan

$content = @"
# local.cmake — configure.ps1 자동 생성 ($((Get-Date).ToString("yyyy-MM-dd HH:mm")))
# 수동 수정도 가능합니다.

set(Torch_DIR "$torchDir")
set(CUDA_TOOLKIT_ROOT_DIR "$cudaRoot")
set(CMAKE_CUDA_COMPILER   "$nvccPath")
set(CMAKE_CUDA_HOST_COMPILER "$msvcCl")
set(TORCH_CUDA_ARCH_LIST "$smArch")
"@

Set-Content -Path $outFile -Value $content -Encoding UTF8
Write-Host "[완료] local.cmake 생성됨" -ForegroundColor Green

# ── 6. 다음 단계 안내 ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== 다음 단계 ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. 빌드:      build_mllm.bat"
Write-Host "2. 모델 다운로드: python scripts\download_model.py --help"
Write-Host "3. 채팅 실행:  chat.bat"
Write-Host "4. 서버 실행:  serve.bat"
Write-Host ""
