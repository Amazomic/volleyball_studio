import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

if (process.platform === "win32") {
  const dir = mkdtempSync(join(tmpdir(), "volley-openvino-discovery-"));
  const finder = resolve("scripts/find-openvino-runtime.ps1");
  const managed = join(dir, "native-deps");
  const pythonRoot = join(dir, "Python");
  const managedDll = join(managed, "openvino", "openvino", "libs", "openvino_c.dll");
  const pythonDll = join(pythonRoot, "Python311", "Lib", "site-packages", "openvino", "libs", "openvino_c.dll");
  const externalDll = join(dir, "external", "openvino_c.dll");
  for (const dll of [managedDll, pythonDll, externalDll]) {
    mkdirSync(join(dll, ".."), { recursive: true });
    writeFileSync(dll, "");
  }
  const run = (explicit = "", envValue = "") =>
    execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
      "-File", finder, "-ExplicitLibrary", explicit,
      "-NativeDepsRoot", managed, "-PythonRoot", pythonRoot], {
      encoding: "utf8",
      env: { ...process.env, VOLLEY_OPENVINO_C_LIB: envValue },
    }).trim();
  try {
    assert.equal(run(), managedDll);
    assert.equal(run(externalDll), externalDll);
    assert.equal(run("", externalDll), externalDll);
    rmSync(managedDll);
    assert.equal(run(), pythonDll);
    const x86 = join(pythonRoot, "Python314-32", "Lib", "site-packages", "openvino", "libs", "openvino_c.dll");
    mkdirSync(join(x86, ".."), { recursive: true });
    writeFileSync(x86, "");
    rmSync(pythonDll);
    assert.equal(run(), "");
    const bad = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive",
      "-ExecutionPolicy", "Bypass", "-File", finder,
      "-ExplicitLibrary", join(dir, "missing.dll"), "-NativeDepsRoot", managed,
      "-PythonRoot", pythonRoot], { encoding: "utf8", env: { ...process.env, VOLLEY_OPENVINO_C_LIB: "" } });
    assert.notEqual(bad.status, 0);
    assert.match(bad.stderr, /OpenVINO/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
const launcher = readFileSync("scripts/start-volley-rust-dev.ps1", "utf8");
assert.match(launcher, /find-openvino-runtime\.ps1/);
assert.match(launcher, /VOLLEY_OPENVINO_C_LIB=if\(\$openvino\)/);
console.log("Windows OpenVINO locator: OK");
