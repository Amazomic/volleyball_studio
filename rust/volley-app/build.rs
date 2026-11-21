use std::{env, fs, path::Path, process::Command};

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

fn main() {
    let manifest = std::path::PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let Some(root) = git(&manifest, &["rev-parse", "--show-toplevel"]) else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    if !root.join(".git").exists() {
        return;
    }
    let Some(remote) = git(&manifest, &["remote", "get-url", "origin"]) else {
        return;
    };
    if !remote.contains("github.com/") && !remote.contains("github.com:") {
        return;
    }
    let Some(sha) = git(&manifest, &["rev-parse", "HEAD"]) else {
        return;
    };
    if sha.len() != 40 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return;
    }
    println!("cargo:rustc-env=VOLLEY_BUILD_GIT_SHA={sha}");
    println!("cargo:rustc-env=VOLLEY_BUILD_GIT_ROOT={}", root.display());

    if let Some(git_dir) = git(&manifest, &["rev-parse", "--absolute-git-dir"]) {
        let git_dir = std::path::PathBuf::from(git_dir);
        let head = git_dir.join("HEAD");
        println!("cargo:rerun-if-changed={}", head.display());
        if let Ok(reference) = fs::read_to_string(head) {
            if let Some(reference) = reference.trim().strip_prefix("ref: ") {
                println!(
                    "cargo:rerun-if-changed={}",
                    git_dir.join(reference).display()
                );
            }
        }
        println!(
            "cargo:rerun-if-changed={}",
            git_dir.join("packed-refs").display()
        );
    }
}
