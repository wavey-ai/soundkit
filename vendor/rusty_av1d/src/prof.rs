// rav1d analyzer spine — feature-gated stage profiler (decoder side).
//
// Port of the rusty_av1e encoder profiler (src/prof.rs) to rav1d's decode
// pipeline, per the `analyzer` skill.
//
//  * OFF (default): `scope()` returns a zero-sized guard with no `Drop`, so every
//    call site is elided — the shipped decoder is byte-identical to stock rav1d
//    (decode md5 unchanged). `profile` is an opt-in cargo feature.
//  * ON (`--features profile`): each `scope(Stage)` times `rdtsc()..drop` into a
//    per-stage atomic cycle bucket (+ a call count). `dump()` recovers ms from a
//    wall/cycle ratio captured between `reset()` and the dump, so the *percentage*
//    breakdown is calibration-free and the *absolute* ms is wall-anchored.
//
// rav1d IS a `#![deny(unsafe_op_in_unsafe_fn)]`-style crate but ships asm and
// uses `unsafe` widely, so the `rdtsc` intrinsic needs no special gating.
//
// IMPORTANT: run the breakdown SINGLE-THREADED (`--threads 1`). rav1d's frame/tile
// task threads would sum per-worker cycles past wall time and scramble the residue.
// The decode is deterministic (1T md5 == MT md5), so 1-thread loses no fidelity.
//
// Nesting (single-pass inline decode, n_threads=1):
//   Total (rav1d_decode_frame_main)
//     ├─ TileSbrow  (rav1d_decode_tile_sbrow): mode symbols + coeff + pred + itx
//     │    ├─ ReconIntra (recon_b_intra): coeff decode + ipred(asm) + itx(asm)
//     │    ├─ ReconInter (recon_b_inter): coeff decode + mc(asm)    + itx(asm)
//     │    │    └─ CoeffDecode (decode_coefs)  [info — the pure-Rust entropy leaf]
//     │    │         └─ MsacSymbol (msac)      [info]
//     │    └─ glue = mode-symbol decode + tile setup
//     └─ FilterSbrow (filter_sbrow): Deblock + Cdef + LoopRestoration  [all asm, info]

/// A decode pipeline stage. Discriminants are declaration-ordered `0..COUNT` so
/// `stage as usize` indexes the bucket arrays directly.
macro_rules! stages {
  ($($variant:ident => $name:literal),+ $(,)?) => {
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[repr(usize)]
    pub enum Stage { $($variant),+ }

    impl Stage {
      pub const ALL: &'static [Stage] = &[$(Stage::$variant),+];
      pub const COUNT: usize = Stage::ALL.len();
      pub const fn name(self) -> &'static str {
        match self { $(Stage::$variant => $name),+ }
      }
      #[inline(always)]
      pub const fn idx(self) -> usize { self as usize }
    }
  };
}

// `Total` MUST be first (denominator + residue base). `TileSbrow` is the
// decompose parent (the analog of the encoder's `PartitionRdo`).
stages! {
  Total            => "TOTAL (decode_frame)",
  TileSbrow        => "tile decode+recon (sym+coeff+pred+itx)",
  FilterSbrow      => "in-loop filters (deblock+cdef+lr)",
  // --- TileSbrow children (nested; inclusive leaf timings) ---
  ReconIntra       => "recon intra (coeff+ipred+itx)",
  ReconInter       => "recon inter (coeff+mc+itx)",
  // --- info tier: nested inside the stages above; displayed, never summed ---
  CoeffDecode      => "coeff decode (PURE RUST) [info]",
  MsacSymbol       => "msac symbol decode [info]",
  Deblock          => "deblock (asm) [info]",
  Cdef             => "cdef (asm) [info]",
  LoopRestoration  => "loop restoration (asm) [info]",
  // --- decode_coefs internal audit (nested inside CoeffDecode) ---
  CoefClass        => "coef: token loop (class+fill) [info]",
  CoefLevelsFill   => "coef:   levels memset [info]",
  CoefDequant      => "coef: dequant+sign loop [info]",
  // per-coefficient differential: true get_lo_ctx = CoefLoCtx − CoefScopeRef
  CoefLoCtx        => "coef:   get_lo_ctx (raw) [info]",
  CoefScopeRef     => "coef:   scope-overhead ref [info]",
  // --- mode-symbol path audit (nested inside TileSbrow, ~17% of decode) ---
  MvRefsFind       => "mode: refmvs_find (MV pred) [info]",
  VartxTree        => "mode: read_vartx_tree [info]",
  ResetCtx         => "mode: reset_context fills [info]",
}

impl Stage {
  /// Info stages nest inside OTHER scoped stages, so they are displayed for
  /// audits but excluded from every sum (including them would double-count).
  pub const fn is_info(self) -> bool {
    matches!(
      self,
      Stage::CoeffDecode
        | Stage::MsacSymbol
        | Stage::Deblock
        | Stage::Cdef
        | Stage::LoopRestoration
        | Stage::CoefClass
        | Stage::CoefLevelsFill
        | Stage::CoefDequant
        | Stage::CoefLoCtx
        | Stage::CoefScopeRef
        | Stage::MvRefsFind
        | Stage::VartxTree
        | Stage::ResetCtx
    )
  }

  /// Child stages nest inside `TileSbrow`; reported as a sub-breakdown of it,
  /// not as top-level siblings (so the top-level residue stays honest).
  pub const fn is_tile_child(self) -> bool {
    matches!(self, Stage::ReconIntra | Stage::ReconInter)
  }
}

#[cfg(feature = "profile")]
pub use imp::*;

#[cfg(not(feature = "profile"))]
pub use noop::*;

// ---------------------------------------------------------------------------
// Real implementation (feature = "profile")
// ---------------------------------------------------------------------------
#[cfg(feature = "profile")]
mod imp {
  use super::Stage;
  use std::sync::atomic::{AtomicU64, Ordering};
  use std::sync::Mutex;
  use std::time::Instant;

  static CYCLES: [AtomicU64; Stage::COUNT] = [const { AtomicU64::new(0) }; Stage::COUNT];
  static CALLS: [AtomicU64; Stage::COUNT] = [const { AtomicU64::new(0) }; Stage::COUNT];

  // Wall/cycle calibration anchor, captured at `reset()` (lazily on first scope
  // if reset was never called). Cold path only — never contends the hot scope.
  static ANCHOR: Mutex<Option<(Instant, u64)>> = Mutex::new(None);

  #[inline(always)]
  fn rdtsc() -> u64 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
      #[cfg(target_arch = "x86_64")]
      // SAFETY: _rdtsc is always available on x86_64.
      unsafe { core::arch::x86_64::_rdtsc() }
      #[cfg(target_arch = "x86")]
      // SAFETY: _rdtsc is always available on x86.
      unsafe { core::arch::x86::_rdtsc() }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
      static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
      BASE.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
  }

  fn ensure_anchor() {
    let mut a = ANCHOR.lock().unwrap();
    if a.is_none() {
      *a = Some((Instant::now(), rdtsc()));
    }
  }

  #[must_use = "the scope guard must be held for the duration being timed"]
  pub struct Guard {
    idx: usize,
    start: u64,
  }

  impl Drop for Guard {
    #[inline(always)]
    fn drop(&mut self) {
      let elapsed = rdtsc().wrapping_sub(self.start);
      CYCLES[self.idx].fetch_add(elapsed, Ordering::Relaxed);
      CALLS[self.idx].fetch_add(1, Ordering::Relaxed);
    }
  }

  #[inline(always)]
  pub fn scope(stage: Stage) -> Guard {
    // First scope with no reset arms the calibration clock.
    if stage.idx() == Stage::Total.idx() {
      ensure_anchor();
    }
    Guard { idx: stage.idx(), start: rdtsc() }
  }

  pub fn reset() {
    for b in CYCLES.iter() { b.store(0, Ordering::Relaxed); }
    for c in CALLS.iter() { c.store(0, Ordering::Relaxed); }
    *ANCHOR.lock().unwrap() = Some((Instant::now(), rdtsc()));
  }

  fn ns_per_cycle() -> f64 {
    match *ANCHOR.lock().unwrap() {
      Some((inst, tsc0)) => {
        let wall_ns = inst.elapsed().as_nanos() as f64;
        let cyc = rdtsc().wrapping_sub(tsc0) as f64;
        if cyc > 0.0 { wall_ns / cyc } else { 0.0 }
      }
      None => 0.0,
    }
  }

  pub fn snapshot() -> Vec<(Stage, f64, u64)> {
    let npc = ns_per_cycle();
    Stage::ALL
      .iter()
      .map(|&s| {
        let cyc = CYCLES[s.idx()].load(Ordering::Relaxed) as f64;
        let calls = CALLS[s.idx()].load(Ordering::Relaxed);
        (s, cyc * npc / 1.0e6, calls)
      })
      .collect()
  }

  /// Print the top-down breakdown. Top-level rows (TileSbrow, FilterSbrow) sum to
  /// ~Total (residue = refmvs + setup); TileSbrow is decomposed into its recon
  /// children; the info tier attributes deeper (coeff/msac inside recon, and the
  /// asm filter kernels inside FilterSbrow).
  pub fn dump(label: &str) {
    let snap = snapshot();
    let total_ms = snap[Stage::Total.idx()].1;
    let tile_ms = snap[Stage::TileSbrow.idx()].1;
    let pct = |ms: f64, d: f64| if d > 0.0 { 100.0 * ms / d } else { 0.0 };

    let mut top: Vec<(Stage, f64, u64)> = snap
      .iter()
      .copied()
      .filter(|(s, ..)| *s != Stage::Total && !s.is_tile_child() && !s.is_info())
      .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let sum_top: f64 = top.iter().map(|(_, ms, _)| ms).sum();
    let residue_ms = total_ms - sum_top;

    eprintln!("\n=== prof::dump [{label}] ===");
    eprintln!("  {:<40} {:>10} {:>8}  {:>12}", "stage", "ms", "% tot", "calls");
    eprintln!("  {}", "-".repeat(74));
    for (s, ms, calls) in &top {
      eprintln!("  {:<40} {:>10.3} {:>7.1}% {:>12}", s.name(), ms, pct(*ms, total_ms), calls);
    }
    eprintln!("  {}", "-".repeat(74));
    eprintln!("  {:<40} {:>10.3} {:>7.1}%", "RESIDUE (refmvs+setup+obu)", residue_ms, pct(residue_ms, total_ms));
    eprintln!("  {:<40} {:>10.3} {:>7.1}%", "TOTAL", total_ms, pct(total_ms, total_ms));

    let mut kids: Vec<(Stage, f64, u64)> =
      snap.iter().copied().filter(|(s, ..)| s.is_tile_child()).collect();
    if kids.iter().any(|(_, ms, c)| *ms > 0.0 || *c > 0) {
      kids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
      let sum_kids: f64 = kids.iter().map(|(_, ms, _)| ms).sum();
      let glue_ms = tile_ms - sum_kids;
      eprintln!("\n  -- decompose 'tile decode+recon' ({tile_ms:.1} ms, % of it) --");
      for (s, ms, calls) in &kids {
        eprintln!("  {:<40} {:>10.3} {:>7.1}% {:>12}", s.name(), ms, pct(*ms, tile_ms), calls);
      }
      eprintln!("  {:<40} {:>10.3} {:>7.1}%", "mode-symbol decode + setup (rust)", glue_ms, pct(glue_ms, tile_ms));
    }

    let infos: Vec<(Stage, f64, u64)> =
      snap.iter().copied().filter(|(s, _, c)| s.is_info() && *c > 0).collect();
    if !infos.is_empty() {
      eprintln!("\n  -- info tier (nested inside stages above; NOT summed; % of Total) --");
      eprintln!("     (CoeffDecode/MsacSymbol nest in recon; Deblock/Cdef/LR decompose FilterSbrow)");
      for (s, ms, calls) in &infos {
        eprintln!("  {:<40} {:>10.3} {:>7.1}% {:>12}", s.name(), ms, pct(*ms, total_ms), calls);
      }
    }
    eprintln!("  (top residue = Total − Σ top-level; tile glue = TileSbrow − Σ children)\n");
  }
}

// ---------------------------------------------------------------------------
// No-op implementation (feature off) — every symbol elides to nothing.
// ---------------------------------------------------------------------------
#[cfg(not(feature = "profile"))]
mod noop {
  use super::Stage;

  /// Zero-sized, no `Drop` — the optimizer removes it entirely.
  pub struct Guard;

  #[inline(always)]
  pub fn scope(_stage: Stage) -> Guard { Guard }
  #[inline(always)]
  pub fn reset() {}
  #[inline(always)]
  pub fn snapshot() -> Vec<(Stage, f64, u64)> { Vec::new() }
  #[inline(always)]
  pub fn dump(_label: &str) {}
}
