[CmdletBinding()]
param(
    [ValidateSet("Deploy","Probe","LauncherProbe")]
    [string]$Action="Deploy",
    [string]$DeploymentRoot=$env:VOLLEY_STUDIO_DEPLOY_ROOT,
    [ValidatePattern('^127\.0\.0\.1:[0-9]{4,5}$')]
    [string]$Bind="127.0.0.1:8875",
    [int]$RestartFromPid=0,
    [string]$RuntimeDataRoot="",
    [switch]$RuntimeReadOnly,
    [ValidateRange(1,3600)]
    [int]$LockTimeoutSeconds=1800,
    [ValidateRange(0,3600)]
    [int]$IdleTimeoutSeconds=900,
    [ValidateRange(1,12)]
    [int]$MaxBuildAttempts=6,
    [string]$ProbeTrace="",
    [string]$ProbeState="",
    [string]$ProbeVersion="",
    [ValidateRange(0,5000)]
    [int]$ProbeDelayMilliseconds=200
)
$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest

if([string]::IsNullOrWhiteSpace($DeploymentRoot)){
    $DeploymentRoot=Join-Path $PSScriptRoot ".."
}
$root=[System.IO.Path]::GetFullPath($DeploymentRoot).TrimEnd([char[]]@('\','/'))
$bindUrl="http://$Bind"
$env:VOLLEY_STUDIO_BIND=$Bind
if($RestartFromPid -lt 0){throw "Invalid restart PID"}
if($RestartFromPid -gt 0){
    if($RuntimeReadOnly -and $RuntimeDataRoot){throw "Conflicting runtime data modes"}
    $env:VOLLEY_STUDIO_RESTART_READONLY=if($RuntimeReadOnly){"1"}else{"0"}
    if($RuntimeDataRoot){$env:VOLLEY_STUDIO_DATA_ROOT=$RuntimeDataRoot}
    $env:VOLLEY_STUDIO_RESTART_GENERATION=[guid]::NewGuid().ToString("N")
}
$shaHasher=[System.Security.Cryptography.SHA256]::Create()
try{
    $digest=$shaHasher.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($root.ToLowerInvariant()))
}finally{
    $shaHasher.Dispose()
}
$lockName="Global\VolleyballStudioDeploy_"+([BitConverter]::ToString($digest).Replace("-","").Substring(0,24))
$mutex=New-Object System.Threading.Mutex($false,$lockName)
$ownsLock=$false

function Git-Text([string[]]$Arguments){
    $previous=$ErrorActionPreference
    try{
        $ErrorActionPreference="Continue"
        $output=@(& git -C $root @Arguments 2>&1)
        $exitCode=$LASTEXITCODE
    }finally{
        $ErrorActionPreference=$previous
    }
    if($exitCode -ne 0){throw "git $($Arguments -join ' ') failed (exit $exitCode): $(($output|Out-String).Trim())"}
    $stdout=@($output|Where-Object {$_ -isnot [System.Management.Automation.ErrorRecord] -and $_.ToString() -notmatch '^warning:'})
    return (($stdout|Out-String).Trim())
}

function Confirm-Checkout {
    if(-not (Test-Path -LiteralPath (Join-Path $root "start-volley-ui.cmd"))){
        throw "Missing Studio launcher in deployment clone: $root"
    }
    $branch=Git-Text @("symbolic-ref","--quiet","--short","HEAD")
    if($branch -ne "develop"){throw "Deployment clone must use develop, got: $branch"}
    $modified=Git-Text @("status","--porcelain","--untracked-files=no")
    if($modified){throw "Deployment clone has uncommitted tracked changes"}
}

function Update-Checkout {
    Confirm-Checkout
    [void](Git-Text @("fetch","--quiet","origin","develop"))
    [void](Git-Text @("merge","--ff-only","--quiet","origin/develop"))
    $sha=Git-Text @("rev-parse","HEAD")
    $published=Git-Text @("rev-parse","refs/remotes/origin/develop")
    if($sha -ne $published){throw "Deployment develop does not match origin/develop"}
    return $sha
}

function Published-Sha {
    $response=Git-Text @("ls-remote","origin","refs/heads/develop")
    $match=[regex]::Match($response,"^([0-9a-fA-F]{40})\s+refs/heads/develop$")
    if(-not $match.Success){throw "Cannot verify published develop SHA"}
    return $match.Groups[1].Value.ToLowerInvariant()
}

function Get-LiveSha {
    try{
        $response=Invoke-WebRequest -UseBasicParsing -Uri "$bindUrl/studio" -TimeoutSec 5
        if($response.StatusCode -ne 200){return ""}
        $match=[regex]::Match($response.Content,'class="studio-build-commit"\s+title="[^"]*?([0-9a-fA-F]{40})"')
        if($match.Success){return $match.Groups[1].Value.ToLowerInvariant()}
    }catch [System.Net.WebException]{
        return ""
    }
    return ""
}

function Invoke-Launcher([string]$Command){
    $previous=$ErrorActionPreference
    try{
        $ErrorActionPreference="Continue"
        if($Command -eq "/daemon"){
            $launcher=Join-Path $root "start-volley-ui.cmd"
            $arguments='/d /s /c ""'+$launcher+'" /daemon >NUL 2>&1"'
            $process=Start-Process -FilePath $env:ComSpec -ArgumentList $arguments -WindowStyle Hidden -PassThru
            $process.WaitForExit()
            $exitCode=$process.ExitCode
            $output=@("Check Studio runtime logs for daemon startup details")
        }else{
            $output=@(& (Join-Path $root "start-volley-ui.cmd") $Command 2>&1)
            $exitCode=$LASTEXITCODE
        }
    }finally{
        $ErrorActionPreference=$previous
    }
    if($exitCode -ne 0){throw "Studio launcher $Command failed (exit $exitCode): $(($output|Select-Object -Last 12|Out-String).Trim())"}
    Write-Output "Launcher $($Command): OK"
}

function Wait-For-Idle {
    $deadline=[DateTime]::UtcNow.AddSeconds($IdleTimeoutSeconds)
    while($true){
        try{
            $status=Invoke-RestMethod -Uri "$bindUrl/internal/rewrite-status" -TimeoutSec 5
        }catch [System.Net.WebException]{
            return
        }
        if($status.name -notin @("Volleyball Studio - development instance","Volleyball Studio - standalone Rust rewrite (development only)")){
            throw "Port $Bind belongs to an unrecognized service"
        }
        if($status.shutdown_safe -eq $true -and @($status.active_jobs).Count -eq 0){
            return
        }
        if([DateTime]::UtcNow -ge $deadline){
            throw "Studio is busy; deployment remains pending (no active job was stopped)"
        }
        Start-Sleep -Seconds 5
    }
}

function Stop-When-Idle {
    $deadline=[DateTime]::UtcNow.AddSeconds($IdleTimeoutSeconds)
    while($true){
        Wait-For-Idle
        try{
            Invoke-Launcher "/stop"
            return
        }catch{
            if($_.Exception.Message -notmatch 'graceful shutdown refused|409|Conflict' -or [DateTime]::UtcNow -ge $deadline){throw}
            Start-Sleep -Milliseconds 500
        }
    }
}

function Get-LivePid {
    try{
        $status=Invoke-RestMethod -Uri "$bindUrl/internal/rewrite-status" -TimeoutSec 5
        return [int]$status.pid
    }catch [System.Net.WebException]{
        return 0
    }
}

function Restart-Satisfied {
    if($RestartFromPid -eq 0){return $true}
    $livePid=Get-LivePid
    return ($livePid -gt 0 -and $livePid -ne $RestartFromPid)
}

function Invoke-Deployment {
    for($attempt=1;$attempt -le $MaxBuildAttempts;$attempt++){
        Confirm-Checkout
        Invoke-Launcher "/status"
        $published=Published-Sha
        $local=Git-Text @("rev-parse","HEAD")
        if($local -eq $published -and (Get-LiveSha) -eq $published){
            if((Restart-Satisfied) -and (Published-Sha) -eq $published){
                Write-Output "SKIPPED: Studio already serves $published"
                return
            }
        }
        Wait-For-Idle
        $target=Update-Checkout
        $live=Get-LiveSha
        if($live -eq $target -and (Restart-Satisfied)){
            if((Published-Sha) -eq $target){
                Write-Output "SKIPPED: Studio already serves $target"
                return
            }
            Write-Output "SUPERSEDED: latest develop changed during the queue check"
            continue
        }
        Write-Output "BUILD: $target"
        Invoke-Launcher "/build"
        if((Published-Sha) -ne $target){
            Write-Output "SUPERSEDED: a newer develop commit arrived during build"
            continue
        }

        Wait-For-Idle
        if((Published-Sha) -ne $target){
            Write-Output "SUPERSEDED: a newer develop commit arrived before restart"
            continue
        }

        $previousPid=Get-LivePid
        Stop-When-Idle
        Invoke-Launcher "/daemon"
        $deployed=Get-LiveSha
        if($deployed -ne $target){
            throw "Live Studio SHA $deployed does not match built SHA $target"
        }
        $deployedPid=Get-LivePid
        if($deployedPid -le 0 -or $deployedPid -eq $previousPid -or ($RestartFromPid -gt 0 -and $deployedPid -eq $RestartFromPid)){
            throw "Studio did not start a new verified process"
        }
        if((Published-Sha) -eq $target){
            Write-Output "DEPLOYED: $target $bindUrl/studio PID $deployedPid"
            return
        }
        Write-Output "SUPERSEDED: another published commit requires the next build"
    }
    throw "Develop advanced too often; no outdated build was marked complete"
}

try{
    try{
        $ownsLock=$mutex.WaitOne(0)
        if(-not $ownsLock){
            Write-Output "QUEUED: another Studio deployment is already running"
            $ownsLock=$mutex.WaitOne([TimeSpan]::FromSeconds($LockTimeoutSeconds))
        }
    }catch [System.Threading.AbandonedMutexException]{
        $ownsLock=$true
    }
    if(-not $ownsLock){throw "Deployment queue wait timed out ($LockTimeoutSeconds seconds)"}

    if($Action -eq "LauncherProbe"){
        Invoke-Launcher "/daemon"
    }elseif($Action -eq "Probe"){
        if(-not $ProbeTrace -or -not $ProbeState -or $ProbeVersion -notmatch '^[a-zA-Z0-9._-]{1,40}$'){
            throw "Probe requires trace, state and safe version"
        }
        $existing=if(Test-Path -LiteralPath $ProbeState){(Get-Content -LiteralPath $ProbeState -Raw).Trim()}else{""}
        if($existing -eq $ProbeVersion){
            Add-Content -LiteralPath $ProbeTrace -Value "skip:$ProbeVersion"
        }else{
            Add-Content -LiteralPath $ProbeTrace -Value "build:$ProbeVersion"
            Start-Sleep -Milliseconds $ProbeDelayMilliseconds
            Set-Content -LiteralPath $ProbeState -Value $ProbeVersion -Encoding ascii
        }
    }else{
        Invoke-Deployment
    }
}finally{
    if($ownsLock){$mutex.ReleaseMutex()}
    $mutex.Dispose()
}
