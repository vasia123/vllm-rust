use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=scripts/pre-commit");

    install_git_hooks();
}

fn install_git_hooks() {
    let hook_dst = Path::new(".git/hooks/pre-commit");
    let hook_src = Path::new("scripts/pre-commit");

    if !Path::new(".git/hooks").exists() || !hook_src.exists() {
        return;
    }

    // Skip if already installed and up-to-date
    if hook_dst.exists() {
        let src = fs::read(hook_src).unwrap_or_default();
        let dst = fs::read(hook_dst).unwrap_or_default();
        if src == dst {
            return;
        }
    }

    if let Err(e) = fs::copy(hook_src, hook_dst) {
        println!("cargo:warning=Failed to install pre-commit hook: {e}");
        return;
    }

    // chmod +x on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(hook_dst, fs::Permissions::from_mode(0o755));
    }

    println!("cargo:warning=Installed pre-commit hook (fmt + clippy checks)");
}
