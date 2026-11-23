[CmdletBinding()]
param(
    [string]$PythonExecutable="python",
    [string]$NativeDepsRoot=(Join-Path $env:LOCALAPPDATA "VolleyballStudio\native-deps"),
    [string]$OpenvinoVersion="2025.4.0"
)
$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest

$target=Join-Path $NativeDepsRoot "openvino"
$library=Join-Path $target "openvino\libs\openvino_c.dll"
if(Test-Path -LiteralPath $library -PathType Leaf){
    Write-Output "OPENVINO_READY: $library"
    exit 0
}
New-Item -ItemType Directory -Path $target -Force | Out-Null
& $PythonExecutable -m pip install --disable-pip-version-check --no-input --target $target "openvino==$OpenvinoVersion"
if($LASTEXITCODE -ne 0){throw "Failed to install OpenVINO Windows runtime"}
if(-not (Test-Path -LiteralPath $library -PathType Leaf)){
    throw "OpenVINO installation did not provide openvino_c.dll: $library"
}
Write-Output "OPENVINO_READY: $library"
