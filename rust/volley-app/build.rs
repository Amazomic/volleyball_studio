use std::{env, fs, path::Path, process::Command};

fn git_output(repo: &Path, args: &[&str], git_dir: Option<(&Path, &Path)>) -> Option<String> {
    let mut command = Command::new("git");
    if let Some((dir, worktree)) = git_dir {
        command
            .arg("--git-dir")
            .arg(dir)
            .arg("--work-tree")
            .arg(worktree);
    }
    let output = command.arg("-C").arg(repo).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    if let Some(result) = git_output(repo, args, None) {
        return Some(result);
    }

    let root = repo
        .ancestors()
        .find(|ancestor| ancestor.join(".git").is_file())?;
    let pointer = fs::read_to_string(root.join(".git")).ok()?;
    let raw_git_dir = pointer.trim().strip_prefix("gitdir: ")?.trim();
    let git_dir = if Path::new(raw_git_dir).is_absolute() {
        std::path::PathBuf::from(raw_git_dir)
    } else if raw_git_dir.len() >= 3
        && raw_git_dir.as_bytes()[0].is_ascii_alphabetic()
        && raw_git_dir.as_bytes()[1] == b':'
        && matches!(raw_git_dir.as_bytes()[2], b'/' | b'\\')
    {
        let output = Command::new("wslpath")
            .args(["-u", raw_git_dir])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        std::path::PathBuf::from(String::from_utf8(output.stdout).ok()?.trim())
    } else {
        root.join(raw_git_dir)
    };
    let git_dir = git_dir.canonicalize().ok()?;
    git_output(repo, args, Some((&git_dir, root)))
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
