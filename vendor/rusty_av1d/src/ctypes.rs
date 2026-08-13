//! C type aliases and `errno` values, without the `libc` dependency.
//!
//! # Why this exists
//!
//! Nothing in this crate is C — it is a pure-Rust port of dav1d, and the `.S`
//! files are feature-gated assembly that is off by default. The only thing tying
//! it to a C library was the `libc` *crate*, which supplies C type aliases and
//! `errno` constants as bindings to the platform C library.
//!
//! Those bindings do not exist on `wasm32-unknown-unknown`, because that target
//! has no libc. That single dependency was the sole thing preventing a
//! WebAssembly build of the decoder:
//!
//! ```text
//! error[E0432]: unresolved import `libc::ptrdiff_t`: no `ptrdiff_t` in the root
//! error[E0425]: cannot find value `ENOENT` in crate `libc`
//! ```
//!
//! # The rule followed here
//!
//! Every definition **mirrors `libc` exactly on every platform this crate
//! already supported**, so native builds are unchanged — this is a dependency
//! removal, not a behaviour change. Where a value genuinely varies by platform
//! it is `cfg`-selected rather than unified, and [`tests`] asserts the whole
//! table against `libc` itself (a dev-dependency), so any platform whose values
//! differ from this table fails CI on that platform instead of shipping quietly.

/// Signed difference between two pointers.
///
/// `libc` defines this as `isize` on every supported target; measured 8 bytes
/// here against `isize`'s 8.
#[allow(non_camel_case_types)]
pub type ptrdiff_t = isize;

/// Signed integer wide enough to hold a pointer. `isize` everywhere.
#[allow(non_camel_case_types)]
pub type intptr_t = isize;

/// Unsigned integer wide enough to hold a pointer. `usize` everywhere.
#[allow(non_camel_case_types)]
pub type uintptr_t = usize;

/// File offset.
///
/// Genuinely platform-varying, and it appears in a `#[repr(C)]` struct
/// ([`Dav1dDataProps`](crate::include::dav1d::common::Dav1dDataProps)), so it is
/// mirrored rather than unified: **measured 4 bytes on Windows MSVC** and 8 on
/// 64-bit Unix. Unifying it would have silently changed that struct's layout on
/// Windows.
#[allow(non_camel_case_types)]
#[cfg(windows)]
pub type off_t = i32;

/// See the Windows variant above.
#[allow(non_camel_case_types)]
#[cfg(all(
    not(windows),
    not(target_family = "wasm"),
    target_pointer_width = "32"
))]
pub type off_t = i32;

/// See the Windows variant above. wasm has no `libc::off_t` to match, so it
/// takes the 64-bit definition — a byte offset into a bitstream wants to be 64
/// bits regardless.
#[allow(non_camel_case_types)]
#[cfg(any(
    target_family = "wasm",
    all(not(windows), target_pointer_width = "64")
))]
pub type off_t = i64;

/// `errno` values used as [`Rav1dError`](crate::error::Rav1dError) discriminants.
///
/// These vary by platform too — `ENOPROTOOPT` measured **123** on Windows and is
/// 92 on Linux, 42 on Apple; `EAGAIN` is 11 on Linux/Windows but 35 on Apple. The
/// enum's numeric value has therefore never been portable, so preserving each
/// platform's existing value keeps this change invisible to existing builds
/// rather than picking one and calling it canonical.
///
/// wasm takes the Linux values, which are the ones `dav1d` upstream documents.
pub mod errno {
    pub const ENOENT: i32 = 2;
    pub const ENOMEM: i32 = 12;
    pub const EINVAL: i32 = 22;
    pub const ERANGE: i32 = 34;

    /// 35 on Apple (where it equals `EWOULDBLOCK`), 11 elsewhere.
    #[cfg(target_vendor = "apple")]
    pub const EAGAIN: i32 = 35;
    /// See the Apple variant above.
    #[cfg(not(target_vendor = "apple"))]
    pub const EAGAIN: i32 = 11;

    #[cfg(windows)]
    pub const ENOPROTOOPT: i32 = 123;
    /// See the Windows variant above.
    #[cfg(target_vendor = "apple")]
    pub const ENOPROTOOPT: i32 = 42;
    /// See the Windows variant above. Also used for wasm.
    #[cfg(not(any(windows, target_vendor = "apple")))]
    pub const ENOPROTOOPT: i32 = 92;
}

/// The table above encodes platform knowledge that can only be *measured* on the
/// platform in question. Rather than trust it, assert it against `libc` itself
/// wherever `libc` exists — so a wrong value fails that platform's CI instead of
/// shipping. On wasm there is no `libc` to disagree with, which is the whole
/// point of the module.
#[cfg(test)]
#[cfg(not(target_family = "wasm"))]
mod tests {
    use super::*;

    #[test]
    fn c_types_match_libc_exactly() {
        use std::mem::size_of;
        assert_eq!(size_of::<ptrdiff_t>(), size_of::<libc::ptrdiff_t>());
        assert_eq!(size_of::<intptr_t>(), size_of::<libc::intptr_t>());
        assert_eq!(size_of::<uintptr_t>(), size_of::<libc::uintptr_t>());
        assert_eq!(
            size_of::<off_t>(),
            size_of::<libc::off_t>(),
            "off_t sits in a #[repr(C)] struct — a size mismatch changes its layout"
        );
    }

    #[test]
    fn errno_values_match_libc_exactly() {
        assert_eq!(errno::ENOENT, libc::ENOENT);
        assert_eq!(errno::EAGAIN, libc::EAGAIN);
        assert_eq!(errno::ENOMEM, libc::ENOMEM);
        assert_eq!(errno::EINVAL, libc::EINVAL);
        assert_eq!(errno::ERANGE, libc::ERANGE);
        assert_eq!(errno::ENOPROTOOPT, libc::ENOPROTOOPT);
    }

    /// The discriminants are cast `as u8`, so any value above 255 would silently
    /// truncate and collide with another error.
    #[test]
    fn errno_values_survive_the_u8_cast() {
        for (name, v) in [
            ("ENOENT", errno::ENOENT),
            ("EAGAIN", errno::EAGAIN),
            ("ENOMEM", errno::ENOMEM),
            ("EINVAL", errno::EINVAL),
            ("ERANGE", errno::ERANGE),
            ("ENOPROTOOPT", errno::ENOPROTOOPT),
        ] {
            assert!(
                (0..=255).contains(&v),
                "{name} = {v} does not fit in the u8 discriminant"
            );
        }
    }
}
