fn main() {
    pkg_config::Config::new()
        .probe("opencore-amrnb")
        .expect("opencore-amrnb is required to build soundkit-amr");
}
