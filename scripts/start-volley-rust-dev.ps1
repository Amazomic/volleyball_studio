[CmdletBinding()]
param(
    [ValidateSet("Start","Detach","Status","Stop","Restart","Build","Doctor","Check")]
    [string]$Action="Start",
    [string]$Bind="127.0.0.1:8875",
    [string]$DataRoot="",
    [string]$OrtLibrary="",
    [string]$OpenvinoLibrary="",
    [switch]$ReadOnly,
    [switch]$CheckOnly
)
$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest

if($CheckOnly){$Action="Check"}
if($Bind -notmatch '^127\.0\.0\.1:([0-9]{4,5})$'){throw "Only an IPv4 loopback port is allowed"}
$port=[int]$Matches[1]
if($port -lt 1024 -or $port -gt 65535){throw "Invalid unprivileged port"}
if($ReadOnly -and $DataRoot){throw "ReadOnly and DataRoot cannot be combined"}
foreach($path in @($DataRoot,$OrtLibrary,$OpenvinoLibrary)){
    if($path -and ($path -notmatch '^[a-zA-Z]:\\' -or $path -match '(^|\\)\.\.(\\|$)')){
        throw "Native Windows runtime paths must be absolute drive paths without parent traversal"
    }
}
$repo=(Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtime=Join-Path $repo ".studio-runtime"
$ownerPath=Join-Path $runtime "windows-owner.json"
$release=Join-Path $repo "rust\target\release\volley-app.exe"
$data=if($DataRoot){$DataRoot}elseif($env:VOLLEY_STUDIO_DATA_ROOT){$env:VOLLEY_STUDIO_DATA_ROOT}else{Join-Path $env:LOCALAPPDATA "VolleyballStudio\data"}
$bindUrl="http://$Bind"
$ort=if($OrtLibrary){$OrtLibrary}else{$env:ORT_DYLIB_PATH}
$openvino=if($OpenvinoLibrary){$OpenvinoLibrary}else{$env:VOLLEY_OPENVINO_C_LIB}

function Git-Value([string[]]$GitArgs) {
    $output=@(& git -C $repo @GitArgs 2>&1)
    if($LASTEXITCODE -ne 0){throw "Git $($GitArgs -join ' ') failed: $(($output|Out-String).Trim())"}
    return (($output|Out-String).Trim())
}
function Confirm-Source {
    if((Git-Value -GitArgs @("symbolic-ref","--short","HEAD")) -ne "develop"){
        throw "Native Studio launch requires a clean develop clone"
    }
    if((Git-Value -GitArgs @("rev-parse","HEAD")) -ne (Git-Value -GitArgs @("rev-parse","refs/remotes/origin/develop"))){
        throw "Native Studio launch requires HEAD == origin/develop"
    }
    if(Git-Value -GitArgs @("status","--porcelain","--untracked-files=no")){
        throw "Native Studio launch refuses uncommitted tracked changes"
    }
}
function Get-Sha {
    return Git-Value -GitArgs @("rev-parse","HEAD")
}
function Get-Candidate {
    return Join-Path (Join-Path $runtime "bin") ("volley-app-"+(Get-Sha)+".exe")
}
function Read-Status {
    try {return Invoke-RestMethod -Uri "$bindUrl/internal/rewrite-status" -TimeoutSec 3}
    catch [System.Net.WebException] {return $null}
}
function Read-Owner {
    if(-not (Test-Path -LiteralPath $ownerPath)){return $null}
    return (Get-Content -LiteralPath $ownerPath -Raw | ConvertFrom-Json)
}
function Assert-Owner($owner,$live) {
    if($null -eq $owner -or $null -eq $live){throw "No verified native Studio owner for $Bind"}
    if($owner.bind -ne $Bind -or $owner.pid -ne $live.pid -or
       $live.name -ne "Volleyball Studio - development instance"){
        throw "Port $Bind is owned by an unknown or different Studio; refusing process control"
    }
    $bin=[System.IO.Path]::GetFullPath([string]$owner.binary)
    $prefix=[System.IO.Path]::GetFullPath((Join-Path $runtime "bin")) + [System.IO.Path]::DirectorySeparatorChar
    if(-not $bin.StartsWith($prefix,[System.StringComparison]::OrdinalIgnoreCase)){
        throw "Native Studio owner executable is outside the deployment bin directory"
    }
    $proc=Get-CimInstance Win32_Process -Filter "ProcessId = $($owner.pid)" -ErrorAction Stop
    if($null -eq $proc -or -not $proc.ExecutablePath -or
       -not ([string]$proc.ExecutablePath).Equals($bin,[System.StringComparison]::OrdinalIgnoreCase)){
        throw "Native Studio PID/executable ownership does not match the recorded deployment"
    }
}
function Assert-FreePort {
    $live=Read-Status
    if($null -ne $live){throw "Port $Bind is already serving another instance"}
    if(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue){
        throw "Port $port is already occupied by an unidentified listener"
    }
}
function Configure-Environment {
    $settings=@{
        VOLLEY_RUST_BIND=$Bind
        VOLLEY_REPOSITORY_ROOT=$repo
        VOLLEY_DEV_UI_ROOT=(Join-Path $repo "rust\volley-app\static")
        VOLLEY_PERSON_DETECTOR="rfdetr-medium"
        VOLLEY_RUST_DATA_ROOT=if($ReadOnly){$null}else{$data}
        ORT_DYLIB_PATH=if($ort){$ort}else{$null}
        VOLLEY_OPENVINO_C_LIB=if($openvino){$openvino}else{$null}
    }
    foreach($entry in $settings.GetEnumerator()){
        [Environment]::SetEnvironmentVariable($entry.Key,$entry.Value,"Process")
    }
    if($ort){$env:PATH=([System.IO.Path]::GetDirectoryName($ort))+";"+$env:PATH}
    if($openvino){$env:PATH=([System.IO.Path]::GetDirectoryName($openvino))+";"+$env:PATH}
}
function Build-Windows {
    Confirm-Source
    Push-Location (Join-Path $repo "rust")
    try {
        & cargo build -p volley-web --target wasm32-unknown-unknown --release --offline --locked
        if($LASTEXITCODE -ne 0){throw "Windows-hosted WASM build failed"}
        Copy-Item "target\wasm32-unknown-unknown\release\volley_web.wasm" "volley-app\static\volley_web.wasm" -Force
        & cargo build --release --offline --locked -p volley-app
        if($LASTEXITCODE -ne 0){throw "Native Windows Rust release build failed"}
    }finally{Pop-Location}
    if(-not (Test-Path -LiteralPath $release)){throw "Windows executable missing: $release"}
    $target=Get-Candidate
    New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
    if(-not (Test-Path -LiteralPath $target)){
        $temporary="$target.$PID.tmp"
        try {
            Copy-Item -LiteralPath $release -Destination $temporary -Force
            Move-Item -LiteralPath $temporary -Destination $target -ErrorAction Stop
        }finally{
            if(Test-Path -LiteralPath $temporary){Remove-Item -LiteralPath $temporary -Force}
        }
    }
    Write-Output "WINDOWS_BUILD: $(Get-Sha) $target"
}
function Test-NativeDependencies {
    foreach($cmd in @("cargo","ffmpeg","ffprobe","git")){
        if(-not (Get-Command $cmd -ErrorAction SilentlyContinue)){throw "Windows dependency unavailable: $cmd"}
    }
    foreach($name in @(
        "models\VballNetV4c_seq9_grayscale_20260908_213829.onnx",
        "models\rf-detr-medium.onnx",
        "models\RTMPose-M-Body7.onnx",
        "models\openvino\person-reidentification-retail-0287.xml"
    )){
        $file=Join-Path $repo $name
        if(-not (Test-Path -LiteralPath $file) -or (Get-Item -LiteralPath $file).Length -eq 0){
            throw "Windows model missing: $file"
        }
    }
    if(-not $ort -or -not (Test-Path -LiteralPath $ort)){
        throw "Native CUDA unavailable: configure ORT_DYLIB_PATH to an ONNX Runtime CUDA Windows DLL"
    }
    $provider=Join-Path (Split-Path -Parent $ort) "onnxruntime_providers_cuda.dll"
    if(-not (Test-Path -LiteralPath $provider)){
        throw "Native CUDA provider missing: $provider"
    }
    Write-Output "Native Windows dependencies present; actual GPU inference smoke is still required"
}
function Start-Windows([bool]$detached){
    Confirm-Source
    $candidate=Get-Candidate
    if(-not (Test-Path -LiteralPath $candidate)){throw "No built Windows executable for $(Get-Sha); use /build"}
    $live=Read-Status
    if($null -ne $live){
        Assert-Owner (Read-Owner) $live
        Write-Output "Native Studio already running at $bindUrl/studio PID $($live.pid)"
        return
    }
    Assert-FreePort
    Configure-Environment
    if(-not $detached){
        & $candidate
        if($LASTEXITCODE -ne 0){throw "Studio exited with error $LASTEXITCODE"}
        return
    }
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    $bytes=New-Object byte[] 32
    $rng=[Security.Cryptography.RandomNumberGenerator]::Create()
    try {$rng.GetBytes($bytes)}finally{$rng.Dispose()}
    $token=[BitConverter]::ToString($bytes).Replace("-","").ToLowerInvariant()
    $env:VOLLEY_SHUTDOWN_TOKEN=$token
    $out=Join-Path $runtime "studio-windows.stdout.log"
    $err=Join-Path $runtime "studio-windows.stderr.log"
    $process=Start-Process -FilePath $candidate -WorkingDirectory (Join-Path $repo "rust") -PassThru -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
    $owner=@{
        pid=$process.Id
        binary=$candidate
        bind=$Bind
        token=$token
        sha=Get-Sha
    }
    $temporary="$ownerPath.$PID.tmp"
    try{
        $owner | ConvertTo-Json -Compress | Set-Content -LiteralPath $temporary -Encoding UTF8
        Move-Item -LiteralPath $temporary -Destination $ownerPath -Force
    }finally{if(Test-Path -LiteralPath $temporary){Remove-Item -LiteralPath $temporary -Force}}
    foreach($attempt in 1..80){
        Start-Sleep -Milliseconds 250
        $live=Read-Status
        if($null -ne $live){
            Assert-Owner (Read-Owner) $live
            Write-Output "WINDOWS_STARTED: $(Get-Sha) $bindUrl/studio PID $($live.pid)"
            return
        }
        if($process.HasExited){break}
    }
    throw "Native Windows Studio did not become ready. Check $err; the process was not force-killed"
}
function Stop-Windows {
    $live=Read-Status
    if($null -eq $live){
        if(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue){
            throw "Port $Bind has an unknown listener; no process stopped"
        }
        Write-Output "Native Windows Studio not running at $Bind"
        return
    }
    $owner=Read-Owner
    Assert-Owner $owner $live
    if($live.shutdown_safe -ne $true -or @($live.active_jobs).Count -gt 0){
        throw "Studio has active work or HTTP streams; graceful shutdown refused"
    }
    $result=Invoke-WebRequest -Method Post -UseBasicParsing -Uri "$bindUrl/internal/windows-shutdown" -Headers @{"x-volley-shutdown-token"=[string]$owner.token} -TimeoutSec 5
    if($result.StatusCode -ne 202){throw "Studio graceful shutdown rejected: $($result.StatusCode)"}
    foreach($attempt in 1..80){
        Start-Sleep -Milliseconds 250
        if($null -eq (Get-CimInstance Win32_Process -Filter "ProcessId = $($owner.pid)")) {
            Write-Output "WINDOWS_STOPPED: PID $($owner.pid)"
            return
        }
    }
    throw "Windows Studio did not exit gracefully; no forced termination was attempted"
}
switch($Action){
    "Doctor" {Test-NativeDependencies}
    "Check" {
        Push-Location (Join-Path $repo "rust")
        try {
            & cargo check --workspace --offline --locked
            if($LASTEXITCODE -ne 0){throw "Native Windows cargo check failed"}
        }finally{Pop-Location}
    }
    "Build" {Build-Windows}
    "Status" {
        $live=Read-Status
        if($null -eq $live){
            if(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue){
                throw "Port $Bind belongs to an unidentified listener"
            }
            Write-Output "Windows Studio not running at $Bind"
        }else{
            Assert-Owner (Read-Owner) $live
            Write-Output "Windows Studio $bindUrl/studio, PID $($live.pid); idle=$($live.shutdown_safe)"
        }
    }
    "Stop" {Stop-Windows}
    "Start" {Build-Windows;Start-Windows $false}
    "Detach" {Start-Windows $true}
    "Restart" {Build-Windows;Stop-Windows;Start-Windows $true}
}
