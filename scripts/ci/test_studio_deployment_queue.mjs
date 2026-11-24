import assert from "node:assert/strict";
import {readFileSync,mkdtempSync,rmSync,existsSync,readFileSync as read,writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {join,resolve} from "node:path";
import {spawn} from "node:child_process";

const root=resolve(import.meta.dirname,"../..");
const ps=join(root,"scripts","deploy-volley-queue.ps1");
const text=readFileSync(ps,"utf8");
assert.match(text,/System\.Threading\.Mutex/);
assert.match(text,/WaitOne\(/);
assert.match(text,/origin\/develop/);
assert.match(text,/Get-LiveSha/);
assert.match(text,/\/build/);
assert.match(text,/\/stop/);
assert.match(text,/\/daemon/);
assert.match(text,/Global\\VolleyballStudioDeploy_/);
assert.match(text,/studio-build-commit/);
assert.match(text,/\$ErrorActionPreference="Continue"/);
assert.match(text,/\$exitCode=\$LASTEXITCODE/);
assert.doesNotMatch(text,/Коммит сборки/);
assert.match(text,/finally\s*\{/);
assert.match(text,/function Wait-For-Idle/);
assert.match(text,/shutdown_safe/);
assert.match(text,/active_jobs/);
assert.match(text,/SUPERSEDED:/);
assert.match(text,/Invoke-Launcher "\/status"/);
assert.match(text,/if\(\$local -eq \$published -and \(Get-LiveSha\) -eq \$published\)/);
assert.ok(text.indexOf("if($local -eq $published") < text.indexOf("        Wait-For-Idle",text.indexOf("function Invoke-Deployment")),"Already-live SHA must skip before waiting for active jobs");
if(process.platform==="win32"){
  const temp=mkdtempSync(join(tmpdir(),"volley-deploy-queue-test-"));
  const trace=join(temp,"trace.txt"),state=join(temp,"state.txt");
  const run=version=>new Promise((done,reject)=>{
    const child=spawn("powershell.exe",[
      "-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",ps,
      "-Action","Probe","-DeploymentRoot",join(temp,"fake-deploy"),
      "-ProbeTrace",trace,"-ProbeState",state,"-ProbeVersion",version,
      "-ProbeDelayMilliseconds","170","-LockTimeoutSeconds","15"
    ],{windowsHide:true});
    let stdout="",stderr="";
    child.stdout.on("data",chunk=>stdout+=chunk);
    child.stderr.on("data",chunk=>stderr+=chunk);
    child.on("error",reject);
    child.on("close",code=>code===0?done(stdout):reject(new Error(`exit ${code}: ${stderr} ${stdout}`)));
  });
  try{
    await Promise.all(Array.from({length:4},()=>run("v1")));
    let lines=read(trace,"utf8").trim().split(/\r?\n/);
    assert.equal(lines.filter(x=>x==="build:v1").length,1,"Concurrent identical requests must build only once");
    assert.equal(lines.filter(x=>x==="skip:v1").length,3);
    await Promise.all([run("v2"),run("v2")]);
    lines=read(trace,"utf8").trim().split(/\r?\n/);
    assert.equal(lines.filter(x=>x==="build:v2").length,1);
    assert.equal(lines.filter(x=>x==="skip:v2").length,1);
    assert.equal(read(state,"utf8").trim(),"v2");
  }finally{rmSync(temp,{recursive:true,force:true});}
}
if(process.platform==="win32"){
  const temp=mkdtempSync(join(tmpdir(),"volley-daemon-inherit-"));
  try{
    writeFileSync(join(temp,"start-volley-ui.cmd"),'@echo off\r\nstart "" /b powershell.exe -NoProfile -Command Start-Sleep -Seconds 8\r\necho launched\r\n');
    const start=Date.now();
    const elapsed=await new Promise((done,reject)=>{
      const child=spawn("powershell.exe",["-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",ps,"-Action","LauncherProbe","-DeploymentRoot",temp],{windowsHide:true});
      let output="";
      child.stdout.on("data",chunk=>output+=chunk);
      child.stderr.on("data",chunk=>output+=chunk);
      child.on("error",reject);
      child.on("exit",code=>code===0?done(Date.now()-start):reject(new Error(`exit ${code}: ${output}`)));
    });
    assert.ok(elapsed<5000,`Launcher waited for detached child: ${elapsed}ms`);
    writeFileSync(join(temp,"start-volley-ui.cmd"),'@echo off\r\nexit /b 17\r\n');
    const failed=await new Promise((done,reject)=>{
      const child=spawn("powershell.exe",["-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",ps,"-Action","LauncherProbe","-DeploymentRoot",temp],{windowsHide:true});
      let output="";
      child.stdout.on("data",chunk=>output+=chunk);
      child.stderr.on("data",chunk=>output+=chunk);
      child.on("error",reject);
      child.on("close",code=>done({code,output}));
    });
    assert.notEqual(failed.code,0);
    assert.match(failed.output,/exit 17/);
  }finally{rmSync(temp,{recursive:true,force:true});}
}
console.log("PASS: deployment queue serializes builds and coalesces duplicate requests");