[CmdletBinding()]
param(
    [string]$ExplicitLibrary="",
    [string]$NativeDepsRoot=(Join-Path $env:LOCALAPPDATA "VolleyballStudio\native-deps"),
    [string]$PythonRoot=(Join-Path $env:LOCALAPPDATA "Programs\Python")
)
$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest

$requested=if($ExplicitLibrary){$ExplicitLibrary}else{$env:VOLLEY_OPENVINO_C_LIB}
if($requested){
    if(-not [System.IO.Path]::IsPathRooted($requested) -or
       -not (Test-Path -LiteralPath $requested -PathType Leaf)){
        throw "OpenVINO C runtime not found at configured path: $requested"
    }
    Write-Output (Resolve-Path -LiteralPath $requested).Path
    return
}

$managed=Join-Path $NativeDepsRoot "openvino\openvino\libs\openvino_c.dll"
if(Test-Path -LiteralPath $managed -PathType Leaf){
    Write-Output (Resolve-Path -LiteralPath $managed).Path
    return
}

if(Test-Path -LiteralPath $PythonRoot -PathType Container){
    $installations=Get-ChildItem -LiteralPath $PythonRoot -Directory |
        Where-Object { $_.Name -match '^Python\d{3}$' } |
        Sort-Object Name -Descending
    foreach($install in $installations){
        $library=Join-Path $install.FullName "Lib\site-packages\openvino\libs\openvino_c.dll"
        if(Test-Path -LiteralPath $library -PathType Leaf){
            Write-Output (Resolve-Path -LiteralPath $library).Path
            return
        }
    }
}
