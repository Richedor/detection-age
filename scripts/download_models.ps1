param(
    [string]$ModelsDir = "D:\vision\env\models"
)

# Crée le dossier cible s'il n'existe pas
if (-not (Test-Path -Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir | Out-Null
}

function Download-File {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Destination
    )
    # Try curl.exe first (handles redirects, can retry)
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        $proc = Start-Process -FilePath $curl.Source -ArgumentList @('-L', '--retry', '5', '--retry-delay', '3', '-o', $Destination, $Url) -NoNewWindow -PassThru -Wait
        if ($proc.ExitCode -eq 0 -and (Test-Path $Destination)) { return $true }
    }

    # Fallback: Invoke-WebRequest with User-Agent
    try {
        $headers = @{ 'User-Agent' = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) PowerShell' }
        Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri $Url -OutFile $Destination -TimeoutSec 600
        if (Test-Path $Destination) { return $true }
    }
    catch {}

    # Fallback: Start-BitsTransfer
    try {
        Start-BitsTransfer -Source $Url -Destination $Destination -Description "download_models.ps1" -ErrorAction Stop
        if (Test-Path $Destination) { return $true }
    }
    catch {}

    return $false
}

function Download-FromList {
    param(
        [Parameter(Mandatory=$true)][string[]]$Urls,
        [Parameter(Mandatory=$true)][string]$Destination,
        [Parameter(Mandatory=$true)][int]$MinBytes
    )
    foreach ($u in $Urls) {
        Write-Host "Telechargement depuis: $u"
        if (Download-File -Url $u -Destination $Destination) {
            if (Test-Path $Destination) {
                $len = (Get-Item $Destination).Length
                if ($len -ge $MinBytes) {
                    return $true
                } else {
                    Write-Warning "Fichier trop petit ($len bytes), tentative d'un autre miroir"
                    try { Remove-Item $Destination -Force -ErrorAction SilentlyContinue } catch {}
                }
            }
        }
        Write-Warning "Echec pour: $u"
    }
    return $false
}

$files = @(
    # Face detector (SSD Caffe)
    @{ Urls = @("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"); Out = "deploy.prototxt" },
    @{ Urls = @("https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"); Out = "res10_300x300_ssd_iter_140000.caffemodel" },

    # AgeNet (try multiple mirrors)
    @{ Urls = @(
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
        "https://cdn.jsdelivr.net/gh/spmallick/learnopencv@master/AgeGender/age_deploy.prototxt"
      ); Out = "age_deploy.prototxt" },
    @{ Urls = @(
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
        "https://cdn.jsdelivr.net/gh/spmallick/learnopencv@master/AgeGender/age_net.caffemodel"
      ); Out = "age_net.caffemodel" }
)

foreach ($f in $files) {
    $target = Join-Path $ModelsDir $f.Out
    if (-not (Test-Path -Path $target)) {
        Write-Host "Telechargement de $($f.Out) ..."
        # Taille minimale d'acceptation
        $minBytes = if ($f.Out -like '*.caffemodel') { 5000000 } else { 500 }
        if (-not (Download-FromList -Urls $f.Urls -Destination $target -MinBytes $minBytes)) {
            Write-Error "Echec du telechargement pour $($f.Out). Essayez de reexecuter plus tard ou telechargez manuellement."
            continue
        }
    } else {
        Write-Host "$($f.Out) existe deja, saut."
    }
}

Write-Host "Fichiers telecharges dans $ModelsDir"
