//! Opus Custom golden vectors made with libopus, checked without libopus.
//!
//! `tests/data/opus_custom/vectors/<case>.pkt` holds the packets that
//! `opus_custom_encode` (libopus with custom modes, complexity 10) made from
//! the synthetic signal below, and `<case>.txt` holds FNV-1a hashes of what
//! libopus decoded:
//!
//! - `libopus_decode_*`: libopus's decode of its own packets, in each PCM
//!   format. The Rust decoder must give the same samples.
//! - `libopus_plc_s16`: libopus's decode of those packets with a fixed set
//!   lost. The Rust concealment must give the same samples.
//! - `rust_portable_packets`, `libopus_decode_rust_s16`: the Rust encoder's
//!   packets with the portable PVQ search, and libopus's decode of them. The
//!   Rust encoder must reproduce the packets and the Rust decoder libopus's
//!   decode.
//! - `rcp_fingerprint`, `native_identical`: on an x86-64 CPU whose `rcpps`
//!   estimates match the generating machine's, the Rust encoder's default
//!   packets equal libopus's when `native_identical` is 1.
//!
//! `regenerate_vectors` (ignored; needs `OPUS_CUSTOM_INTEROP_C`, the tool
//! built from `tools/opus_custom_interop.c`) rewrites the files, and
//! `live_libopus_comparison` (ignored, same variable) compares against
//! libopus directly.

use soundkit_opus::{Decoder, Encoder};
use std::path::PathBuf;
use std::process::Command;

struct Case {
    name: &'static str,
    frame: usize,
    channels: usize,
    bitrate: i32,
    vbr: bool,
    signalling: bool,
}

const CASES: [Case; 4] = [
    Case {
        name: "f48-stereo-256k-cbr",
        frame: 48,
        channels: 2,
        bitrate: 256_000,
        vbr: false,
        signalling: true,
    },
    Case {
        name: "f64-mono-192k-cvbr",
        frame: 64,
        channels: 1,
        bitrate: 192_000,
        vbr: true,
        signalling: true,
    },
    Case {
        name: "f72-stereo-320k-cbr-nosig",
        frame: 72,
        channels: 2,
        bitrate: 320_000,
        vbr: false,
        signalling: false,
    },
    Case {
        name: "f120-stereo-256k-cvbr",
        frame: 120,
        channels: 2,
        bitrate: 256_000,
        vbr: true,
        signalling: true,
    },
];

const SECONDS_X10: usize = 5;

/// A deterministic test signal in integer arithmetic, so it is the same on
/// every target: two triangle waves, filtered noise bursts, clicks, a quiet
/// stretch and 20 ms of digital silence at the start.
fn signal(channels: usize) -> Vec<i16> {
    let frames = 48_000 * SECONDS_X10 / 10;
    let mut out = Vec::with_capacity(frames * channels);
    let mut seed = 0x1234_5678u32;
    let mut lowpass = [0i32; 2];
    for i in 0..frames {
        for c in 0..channels {
            if i < 960 {
                out.push(0);
                continue;
            }
            let triangle = |period: usize| {
                let phase = (i % period) as i32 * 4 * 8192 / period as i32;
                if phase < 2 * 8192 {
                    phase - 8192
                } else {
                    3 * 8192 - phase
                }
            };
            let mut v = triangle(218 + 40 * c) / 2 + triangle(15 + c) / 5;
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((seed >> 16) as i32 & 0x3fff) - 0x2000;
            lowpass[c] += (noise - lowpass[c]) / 4;
            if (i / 4800) % 3 == 1 {
                v += lowpass[c];
            }
            if i % 4800 == 0 {
                v += 12_000;
            }
            if (i / 12_000) % 2 == 1 {
                v /= 40;
            }
            out.push(v.clamp(-32_768, 32_767) as i16);
        }
    }
    out
}

fn lost(index: usize) -> bool {
    index % 17 == 5 || (100..103).contains(&index) || (200..250).contains(&index)
}

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash = (*hash ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
    }
}

const FNV_START: u64 = 0xcbf2_9ce4_8422_2325;

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/opus_custom/vectors")
}

fn read_packets(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut pos = 0;
    while pos + 2 <= bytes.len() {
        let n = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        packets.push(bytes[pos + 2..pos + 2 + n].to_vec());
        pos += 2 + n;
    }
    packets
}

fn write_packets(packets: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for packet in packets {
        out.extend_from_slice(&(packet.len() as u16).to_le_bytes());
        out.extend_from_slice(packet);
    }
    out
}

fn encoder(case: &Case, portable: bool) -> Encoder {
    let mut encoder = Encoder::new_custom(48_000, case.channels, case.frame).unwrap();
    encoder.set_custom_signalling(case.signalling).unwrap();
    encoder.set_bitrate(case.bitrate).unwrap();
    encoder.set_vbr(case.vbr).unwrap();
    encoder.set_portable_pvq(portable).unwrap();
    encoder
}

fn encode(case: &Case, portable: bool) -> Vec<Vec<u8>> {
    let mut encoder = encoder(case, portable);
    signal(case.channels)
        .chunks_exact(case.frame * case.channels)
        .map(|frame| encoder.encode_i16_vec(frame, case.frame).unwrap())
        .collect()
}

/// Hashes the Rust decoder's output for `packets` in `format`.
fn decode_hash(case: &Case, packets: &[Vec<u8>], format: &str) -> u64 {
    let mut decoder = Decoder::new_custom(48_000, case.channels, case.frame).unwrap();
    decoder.set_custom_signalling(case.signalling).unwrap();
    let mut hash = FNV_START;
    for packet in packets {
        match format {
            "s16" => decoder
                .decode_i16_vec(packet, false)
                .unwrap()
                .iter()
                .for_each(|v| fnv(&mut hash, &v.to_le_bytes())),
            "s24" => decoder
                .decode_i24_vec(packet, false)
                .unwrap()
                .iter()
                .for_each(|v| fnv(&mut hash, &v.to_le_bytes())),
            _ => decoder
                .decode_f32_vec(packet, false)
                .unwrap()
                .iter()
                .for_each(|v| fnv(&mut hash, &v.to_le_bytes())),
        }
    }
    hash
}

fn with_losses(packets: &[Vec<u8>]) -> Vec<Vec<u8>> {
    packets
        .iter()
        .enumerate()
        .map(|(i, p)| if lost(i) { Vec::new() } else { p.clone() })
        .collect()
}

fn manifest(case: &Case) -> std::collections::BTreeMap<String, String> {
    let text = std::fs::read_to_string(vectors_dir().join(format!("{}.txt", case.name)))
        .expect("vector manifest");
    text.lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

fn hex(value: u64) -> String {
    format!("{value:016x}")
}

#[test]
fn rust_decoder_matches_libopus_on_libopus_packets() {
    for case in &CASES {
        let m = manifest(case);
        let packets =
            read_packets(&std::fs::read(vectors_dir().join(format!("{}.pkt", case.name))).unwrap());
        for format in ["s16", "s24", "f32"] {
            assert_eq!(
                hex(decode_hash(case, &packets, format)),
                m[&format!("libopus_decode_{format}")],
                "{} {format}",
                case.name
            );
        }
        assert_eq!(
            hex(decode_hash(case, &with_losses(&packets), "s16")),
            m["libopus_plc_s16"],
            "{} concealment",
            case.name
        );
    }
}

#[test]
fn rust_packets_decode_as_libopus_decodes_them() {
    for case in &CASES {
        let m = manifest(case);
        let packets = encode(case, true);
        let mut hash = FNV_START;
        fnv(&mut hash, &write_packets(&packets));
        assert_eq!(hex(hash), m["rust_portable_packets"], "{}", case.name);
        assert_eq!(
            hex(decode_hash(case, &packets, "s16")),
            m["libopus_decode_rust_s16"],
            "{}",
            case.name
        );
    }
}

#[test]
fn rust_encoder_matches_opus_custom_encode_on_the_same_cpu_family() {
    let fingerprint = soundkit_opus::celt::refkern::rcp_estimate_fingerprint();
    for case in &CASES {
        let m = manifest(case);
        if fingerprint.map(hex).as_deref() != Some(m["rcp_fingerprint"].as_str()) {
            eprintln!(
                "{}: this CPU's rcpps estimates differ from the vectors' machine; skipped",
                case.name
            );
            continue;
        }
        let expected = std::fs::read(vectors_dir().join(format!("{}.pkt", case.name))).unwrap();
        let packets = encode(case, false);
        let identical = write_packets(&packets) == expected;
        assert_eq!(identical, m["native_identical"] == "1", "{}", case.name);
        assert!(
            identical,
            "{}: Rust packets differ from opus_custom_encode",
            case.name
        );
    }
}

fn c_tool() -> Option<PathBuf> {
    std::env::var_os("OPUS_CUSTOM_INTEROP_C").map(PathBuf::from)
}

fn run_c(tool: &PathBuf, args: &[&str]) {
    let status = Command::new(tool)
        .args(args)
        .status()
        .expect("run the C tool");
    assert!(status.success(), "C tool failed: {args:?}");
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        // Unique per call: tests run in parallel and share case names.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "soundkit-opus-{tag}-{}-{serial}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn file_hash(path: &str) -> u64 {
    let mut hash = FNV_START;
    fnv(&mut hash, &std::fs::read(path).unwrap());
    hash
}

fn c_args(case: &Case) -> Vec<String> {
    vec![
        "--frame".into(),
        case.frame.to_string(),
        "--channels".into(),
        case.channels.to_string(),
        "--signalling".into(),
        u8::from(case.signalling).to_string(),
        "--pcm".into(),
        "s16".into(),
    ]
}

/// libopus's packets and decodes for one case, in `scratch`.
fn libopus_outputs(tool: &PathBuf, case: &Case, scratch: &Scratch) -> Vec<u8> {
    let pcm: Vec<u8> = signal(case.channels)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(scratch.path("in.s16"), pcm).unwrap();
    let common = c_args(case);
    let mut enc: Vec<String> = vec!["encode".into()];
    enc.extend(common.iter().cloned());
    enc.extend([
        "--bitrate".into(),
        case.bitrate.to_string(),
        "--vbr".into(),
        u8::from(case.vbr).to_string(),
        "--complexity".into(),
        "10".into(),
        scratch.path("in.s16"),
        scratch.path("lib.pkt"),
    ]);
    run_c(tool, &enc.iter().map(String::as_str).collect::<Vec<_>>());
    std::fs::read(scratch.path("lib.pkt")).unwrap()
}

fn c_decode_hash(
    tool: &PathBuf,
    case: &Case,
    packets: &[u8],
    format: &str,
    scratch: &Scratch,
) -> u64 {
    std::fs::write(scratch.path("dec.pkt"), packets).unwrap();
    let mut args: Vec<String> = vec!["decode".into()];
    let mut common = c_args(case);
    common[7] = format.into();
    args.extend(common);
    args.extend([scratch.path("dec.pkt"), scratch.path("dec.pcm")]);
    run_c(tool, &args.iter().map(String::as_str).collect::<Vec<_>>());
    file_hash(&scratch.path("dec.pcm"))
}

#[test]
#[ignore = "needs OPUS_CUSTOM_INTEROP_C; rewrites tests/data/opus_custom/vectors"]
fn regenerate_vectors() {
    let tool = c_tool().expect("set OPUS_CUSTOM_INTEROP_C to the libopus tool");
    std::fs::create_dir_all(vectors_dir()).unwrap();
    for case in &CASES {
        let scratch = Scratch::new(case.name);
        let lib = libopus_outputs(&tool, case, &scratch);
        let lib_packets = read_packets(&lib);
        let mut lines = vec![
            format!("frame = {}", case.frame),
            format!("channels = {}", case.channels),
            format!("bitrate = {}", case.bitrate),
            format!("vbr = {}", u8::from(case.vbr)),
            format!("signalling = {}", u8::from(case.signalling)),
        ];
        for format in ["s16", "s24", "f32"] {
            let hash = c_decode_hash(&tool, case, &lib, format, &scratch);
            lines.push(format!("libopus_decode_{format} = {}", hex(hash)));
        }
        let lossy = write_packets(&with_losses(&lib_packets));
        lines.push(format!(
            "libopus_plc_s16 = {}",
            hex(c_decode_hash(&tool, case, &lossy, "s16", &scratch))
        ));
        let rust = write_packets(&encode(case, true));
        let mut hash = FNV_START;
        fnv(&mut hash, &rust);
        lines.push(format!("rust_portable_packets = {}", hex(hash)));
        lines.push(format!(
            "libopus_decode_rust_s16 = {}",
            hex(c_decode_hash(&tool, case, &rust, "s16", &scratch))
        ));
        let fingerprint = soundkit_opus::celt::refkern::rcp_estimate_fingerprint();
        lines.push(format!(
            "rcp_fingerprint = {}",
            fingerprint.map(hex).unwrap_or_else(|| "none".into())
        ));
        let native = write_packets(&encode(case, false));
        lines.push(format!("native_identical = {}", u8::from(native == lib)));
        std::fs::write(vectors_dir().join(format!("{}.pkt", case.name)), &lib).unwrap();
        std::fs::write(
            vectors_dir().join(format!("{}.txt", case.name)),
            lines.join("\n") + "\n",
        )
        .unwrap();
    }
}

#[test]
#[ignore = "needs OPUS_CUSTOM_INTEROP_C (tools/opus_custom_interop.c built with libopus)"]
fn live_libopus_comparison() {
    let tool = c_tool().expect("set OPUS_CUSTOM_INTEROP_C to the libopus tool");
    for case in &CASES {
        let scratch = Scratch::new(case.name);
        let lib = libopus_outputs(&tool, case, &scratch);
        let lib_packets = read_packets(&lib);
        for format in ["s16", "s24", "f32"] {
            assert_eq!(
                c_decode_hash(&tool, case, &lib, format, &scratch),
                decode_hash(case, &lib_packets, format),
                "{} libopus packets, {format}",
                case.name
            );
        }
        let lossy = with_losses(&lib_packets);
        assert_eq!(
            c_decode_hash(&tool, case, &write_packets(&lossy), "s16", &scratch),
            decode_hash(case, &lossy, "s16"),
            "{} concealment",
            case.name
        );
        for portable in [false, true] {
            let rust = encode(case, portable);
            assert_eq!(
                c_decode_hash(&tool, case, &write_packets(&rust), "s16", &scratch),
                decode_hash(case, &rust, "s16"),
                "{} Rust packets",
                case.name
            );
        }
        let native = write_packets(&encode(case, false));
        let same = read_packets(&native)
            .iter()
            .zip(&lib_packets)
            .filter(|(a, b)| a == b)
            .count();
        eprintln!(
            "{}: {same}/{} packets identical to opus_custom_encode",
            case.name,
            lib_packets.len()
        );
    }
}

// Channel switches and damaged packets.
//
// `<name>.pkt` interleaves libopus's mono and stereo packets of the same
// signal and frame size (signalling on): runs of 37 packets in the first
// half, then alternate packets. libopus decodes such a stream with each
// output channel's own synthesis and post-filter state, downmixes stereo
// packets in a mono decoder, and decodes packets whose range coding is
// damaged. `<name>.txt` holds the hashes of libopus's decodes with a stereo
// and a mono decoder: clean, with losses, and with damage (some payloads
// bit-flipped, some packets with an invalid signalling byte, which libopus
// rejects after taking their end band, followed by a loss).

struct SwitchCase {
    name: &'static str,
    frame: usize,
}

const SWITCH_CASES: [SwitchCase; 2] = [
    SwitchCase {
        name: "f48-switch",
        frame: 48,
    },
    SwitchCase {
        name: "f120-switch",
        frame: 120,
    },
];

fn switch_stereo_at(index: usize, packets: usize) -> bool {
    if index < packets / 2 {
        (index / 37) % 2 == 1
    } else {
        index % 2 == 1
    }
}

/// Deterministic damage: flipped payload bits, and packets whose signalling
/// byte names an impossible frame size (or, in the standard mode, no CELT
/// configuration), each followed by a lost packet, the first of them at the
/// start of the stream.
fn damaged(case: &SwitchCase, packets: &[Vec<u8>]) -> Vec<Vec<u8>> {
    packets
        .iter()
        .enumerate()
        .map(|(i, packet)| {
            let mut packet = packet.clone();
            // At the start the decoder has not yet decoded two packets in a
            // row, so the loss after the rejected packet is concealed with
            // noise, which uses the end band the rejected packet set.
            if i % 29 == 8 || i == 1 {
                packet.clear();
            } else if (i % 29 == 7 || i == 0) && !packet.is_empty() {
                packet[0] = if case.frame == 120 {
                    0x10
                } else {
                    (packet[0] & 0x07) | 0x40 | (3 << 3)
                };
            } else if i % 11 == 3 && packet.len() > 3 {
                let middle = packet.len() / 2;
                packet[middle] ^= 0x5a;
                let last = packet.len() - 1;
                packet[last] ^= 0x81;
            }
            packet
        })
        .collect()
}

/// Hashes the Rust decoder's output and counts the packets it rejects; a
/// rejected packet adds no output, as with the C tool's `--errors skip`.
fn decode_stream_hash(
    frame: usize,
    channels: usize,
    packets: &[Vec<u8>],
    format: &str,
) -> (u64, usize) {
    let mut decoder = Decoder::new_custom(48_000, channels, frame).unwrap();
    let mut hash = FNV_START;
    let mut errors = 0;
    for packet in packets {
        let ok = match format {
            "s16" => decoder
                .decode_i16_vec(packet, false)
                .map(|pcm| pcm.iter().for_each(|v| fnv(&mut hash, &v.to_le_bytes()))),
            _ => decoder
                .decode_f32_vec(packet, false)
                .map(|pcm| pcm.iter().for_each(|v| fnv(&mut hash, &v.to_le_bytes()))),
        };
        errors += usize::from(ok.is_err());
    }
    (hash, errors)
}

fn switch_manifest(case: &SwitchCase) -> std::collections::BTreeMap<String, String> {
    let text = std::fs::read_to_string(vectors_dir().join(format!("{}.txt", case.name)))
        .expect("vector manifest");
    text.lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

/// The streams each switch case decodes, by manifest key prefix.
fn switch_streams(case: &SwitchCase, packets: &[Vec<u8>]) -> [(&'static str, Vec<Vec<u8>>); 3] {
    [
        ("clean", packets.to_vec()),
        ("plc", with_losses(packets)),
        ("damaged", damaged(case, packets)),
    ]
}

#[test]
fn rust_decoder_matches_libopus_across_channel_switches_and_damage() {
    for case in &SWITCH_CASES {
        let m = switch_manifest(case);
        let packets =
            read_packets(&std::fs::read(vectors_dir().join(format!("{}.pkt", case.name))).unwrap());
        for (stream, input) in switch_streams(case, &packets) {
            for channels in [2, 1] {
                for format in ["s16", "f32"] {
                    let key = format!("libopus_{stream}_c{channels}_{format}");
                    let (hash, errors) = decode_stream_hash(case.frame, channels, &input, format);
                    assert_eq!(hex(hash), m[&key], "{} {key}", case.name);
                    assert_eq!(
                        errors.to_string(),
                        m[&format!("libopus_{stream}_rejected")],
                        "{} {key} rejected packets",
                        case.name
                    );
                }
            }
        }
    }
}

/// The 120-sample custom packets carry libopus's TOC form of the signalling
/// byte, so they are also CELT-only Opus packets. A standard decoder with
/// libopus arithmetic must decode them as `opus_custom_decode` does,
/// including stereo packets in a mono decoder.
#[test]
fn standard_decoder_matches_libopus_across_channel_switches() {
    let case = &SWITCH_CASES[1];
    assert_eq!(case.frame, 120);
    let m = switch_manifest(case);
    let packets =
        read_packets(&std::fs::read(vectors_dir().join(format!("{}.pkt", case.name))).unwrap());
    for (stream, input) in switch_streams(case, &packets).into_iter().take(2) {
        for channels in [2, 1] {
            let mut decoder = Decoder::new(48_000, channels).unwrap();
            decoder.set_libopus_arithmetic(true).unwrap();
            let mut hash = FNV_START;
            for packet in &input {
                for v in decoder.decode_i16_vec(packet, false).unwrap() {
                    fnv(&mut hash, &v.to_le_bytes());
                }
            }
            let key = format!("libopus_{stream}_c{channels}_s16");
            assert_eq!(hex(hash), m[&key], "standard decoder {key}");
        }
    }
}

/// Runs the C tool's decoder with `--errors skip` and returns the output
/// hash and the number of packets it rejected.
fn c_stream_hash(
    tool: &PathBuf,
    frame: usize,
    channels: usize,
    packets: &[Vec<u8>],
    format: &str,
    scratch: &Scratch,
) -> (u64, usize) {
    std::fs::write(scratch.path("switch.pkt"), write_packets(packets)).unwrap();
    let frame = frame.to_string();
    let channels = channels.to_string();
    let output = Command::new(tool)
        .args([
            "decode",
            "--frame",
            &frame,
            "--channels",
            &channels,
            "--errors",
            "skip",
            "--pcm",
            format,
            &scratch.path("switch.pkt"),
            &scratch.path("switch.pcm"),
        ])
        .output()
        .expect("run the C tool");
    assert!(output.status.success(), "C tool failed");
    let errors = String::from_utf8_lossy(&output.stderr)
        .lines()
        .filter(|line| line.contains("decode error"))
        .count();
    (file_hash(&scratch.path("switch.pcm")), errors)
}

#[test]
#[ignore = "needs OPUS_CUSTOM_INTEROP_C; rewrites the channel-switch vectors"]
fn regenerate_switch_vectors() {
    let tool = c_tool().expect("set OPUS_CUSTOM_INTEROP_C to the libopus tool");
    for case in &SWITCH_CASES {
        let scratch = Scratch::new(case.name);
        let mut streams = Vec::new();
        for (channels, bitrate) in [(1usize, "128000"), (2, "256000")] {
            let pcm: Vec<u8> = signal(channels)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            std::fs::write(scratch.path("in.s16"), pcm).unwrap();
            let frame = case.frame.to_string();
            let channels = channels.to_string();
            run_c(
                &tool,
                &[
                    "encode",
                    "--frame",
                    &frame,
                    "--channels",
                    &channels,
                    "--bitrate",
                    bitrate,
                    "--vbr",
                    "1",
                    "--pcm",
                    "s16",
                    &scratch.path("in.s16"),
                    &scratch.path("enc.pkt"),
                ],
            );
            streams.push(read_packets(
                &std::fs::read(scratch.path("enc.pkt")).unwrap(),
            ));
        }
        let count = streams[0].len();
        let packets: Vec<Vec<u8>> = (0..count)
            .map(|i| streams[usize::from(switch_stereo_at(i, count))][i].clone())
            .collect();
        let mut lines = vec![format!("frame = {}", case.frame)];
        for (stream, input) in switch_streams(case, &packets) {
            let mut rejected = None;
            for channels in [2, 1] {
                for format in ["s16", "f32"] {
                    let (hash, errors) =
                        c_stream_hash(&tool, case.frame, channels, &input, format, &scratch);
                    lines.push(format!(
                        "libopus_{stream}_c{channels}_{format} = {}",
                        hex(hash)
                    ));
                    assert_eq!(*rejected.get_or_insert(errors), errors);
                }
            }
            lines.push(format!("libopus_{stream}_rejected = {}", rejected.unwrap()));
        }
        std::fs::write(
            vectors_dir().join(format!("{}.pkt", case.name)),
            write_packets(&packets),
        )
        .unwrap();
        std::fs::write(
            vectors_dir().join(format!("{}.txt", case.name)),
            lines.join("\n") + "\n",
        )
        .unwrap();
    }
}
