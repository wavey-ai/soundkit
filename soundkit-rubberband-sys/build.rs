use std::env;
use std::path::PathBuf;
#[cfg(target_os = "macos")]
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=RUBBERBAND_LIB_DIR");

    if let Ok(lib_dir) = env::var("RUBBERBAND_LIB_DIR") {
        link_search(lib_dir);
        return;
    }

    if pkg_config::Config::new()
        .atleast_version("3.0.0")
        .probe("rubberband")
        .is_ok()
    {
        return;
    }

    #[cfg(target_os = "macos")]
    if let Some(prefix) = brew_prefix("rubberband") {
        link_search(prefix.join("lib"));
        return;
    }

    panic!(
        "Could not find the Rubber Band library. Install it with your system package manager \
         or set RUBBERBAND_LIB_DIR to the directory containing librubberband."
    );
}

fn link_search(path: impl Into<PathBuf>) {
    let path = path.into();
    println!("cargo:rustc-link-search=native={}", path.display());
    println!("cargo:rustc-link-lib=rubberband");
}

#[cfg(target_os = "macos")]
fn brew_prefix(formula: &str) -> Option<PathBuf> {
    let output = Command::new("brew")
        .args(["--prefix", formula])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let prefix = String::from_utf8(output.stdout).ok()?;
    let trimmed = prefix.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}
