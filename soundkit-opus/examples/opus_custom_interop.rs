//! Rust side of the Opus Custom interop harness.
//!
//! Takes the same arguments as `tools/opus_custom_interop.c` (see that file
//! for the PCM and packet file formats):
//!
//! ```text
//! opus_custom_interop encode --frame N --channels C [--bitrate B|max] [--vbr 0|1]
//!     [--cvbr 0|1] [--signalling 0|1] [--max-bytes N] --pcm s16|s24|f32 IN OUT
//! opus_custom_interop decode --frame N --channels C [--signalling 0|1]
//!     --pcm s16|s24|f32 IN OUT
//! ```
//!
//! Rust-only options: `--decisions libopus|soundkit` picks the encoder's
//! decision constants (default `libopus` for custom modes), `--arith
//! libopus|soundkit` picks the arithmetic (default `libopus`), `--pvq
//! native|portable` picks the PVQ search (default `native`: libopus's SSE2
//! search on x86-64), and `--complexity` is accepted and ignored (the Rust
//! encoder always runs the complexity-10 analysis).

use soundkit_opus::{Decoder, Encoder, BITRATE_MAX};
use std::fs;
use std::process::exit;

struct Args {
    command: String,
    frame: usize,
    channels: usize,
    bitrate: i32,
    vbr: bool,
    cvbr: bool,
    signalling: bool,
    max_bytes: usize,
    pcm: String,
    libopus_decisions: bool,
    libopus_arith: bool,
    portable_pvq: bool,
    input: String,
    output: String,
}

fn usage() -> ! {
    eprintln!(
        "usage: opus_custom_interop encode|decode --frame N [--channels C] [--bitrate B|max] \
         [--vbr 0|1] [--cvbr 0|1] [--signalling 0|1] [--max-bytes N] [--pcm s16|s24|f32] \
         [--decisions libopus|soundkit] [--arith libopus|soundkit] [--pvq native|portable] \
         IN OUT"
    );
    exit(2);
}

fn parse() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        usage();
    }
    let mut args = Args {
        command: argv[0].clone(),
        frame: 0,
        channels: 2,
        bitrate: BITRATE_MAX,
        vbr: false,
        cvbr: true,
        signalling: true,
        max_bytes: 1275,
        pcm: "s16".into(),
        libopus_decisions: true,
        libopus_arith: true,
        portable_pvq: false,
        input: String::new(),
        output: String::new(),
    };
    let mut positional = Vec::new();
    let mut i = 1;
    while i < argv.len() {
        let key = argv[i].as_str();
        if let Some(name) = key.strip_prefix("--") {
            let value = argv.get(i + 1).unwrap_or_else(|| usage()).as_str();
            let flag = || value == "1";
            match name {
                "frame" => args.frame = value.parse().unwrap_or_else(|_| usage()),
                "channels" => args.channels = value.parse().unwrap_or_else(|_| usage()),
                "bitrate" => {
                    args.bitrate = if value == "max" {
                        BITRATE_MAX
                    } else {
                        value.parse().unwrap_or_else(|_| usage())
                    }
                }
                "vbr" => args.vbr = flag(),
                "cvbr" => args.cvbr = flag(),
                "signalling" => args.signalling = flag(),
                "max-bytes" => args.max_bytes = value.parse().unwrap_or_else(|_| usage()),
                "pcm" => args.pcm = value.to_string(),
                "complexity" => {}
                "decisions" => args.libopus_decisions = value == "libopus",
                "arith" => args.libopus_arith = value == "libopus",
                "pvq" => args.portable_pvq = value == "portable",
                _ => usage(),
            }
            i += 2;
        } else {
            positional.push(key.to_string());
            i += 1;
        }
    }
    if positional.len() != 2 || args.frame == 0 {
        usage();
    }
    args.input = positional[0].clone();
    args.output = positional[1].clone();
    args
}

fn fail(message: String) -> ! {
    eprintln!("opus_custom_interop: {message}");
    exit(1);
}

fn read_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn read_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn read_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn encode(args: &Args) {
    let raw = fs::read(&args.input).unwrap_or_else(|e| fail(format!("{}: {e}", args.input)));
    let mut encoder = Encoder::new_custom(48_000, args.channels, args.frame)
        .unwrap_or_else(|e| fail(format!("encoder: {e}")));
    encoder
        .set_custom_signalling(args.signalling)
        .unwrap_or_else(|e| fail(format!("signalling: {e}")));
    encoder
        .set_bitrate(args.bitrate)
        .unwrap_or_else(|e| fail(format!("bitrate: {e}")));
    encoder
        .set_vbr(args.vbr)
        .unwrap_or_else(|e| fail(format!("vbr: {e}")));
    encoder
        .set_vbr_constraint(args.cvbr)
        .unwrap_or_else(|e| fail(format!("cvbr: {e}")));
    encoder
        .set_max_packet_bytes(args.max_bytes)
        .unwrap_or_else(|e| fail(format!("max bytes: {e}")));
    encoder.set_libopus_decisions(args.libopus_decisions);
    encoder
        .set_libopus_arithmetic(args.libopus_arith)
        .unwrap_or_else(|e| fail(format!("arith: {e}")));
    encoder
        .set_portable_pvq(args.portable_pvq)
        .unwrap_or_else(|e| fail(format!("pvq: {e}")));
    let samples_per_frame = args.frame * args.channels;
    let mut out = Vec::new();
    let mut packet = Vec::new();
    let emit = |packet: &[u8], out: &mut Vec<u8>| {
        out.extend_from_slice(&(packet.len() as u16).to_le_bytes());
        out.extend_from_slice(packet);
    };
    match args.pcm.as_str() {
        "s16" => {
            for frame in read_i16(&raw).chunks_exact(samples_per_frame) {
                encoder
                    .encode_i16_into(frame, args.frame, &mut packet)
                    .unwrap_or_else(|e| fail(format!("encode: {e}")));
                emit(&packet, &mut out);
            }
        }
        "s24" => {
            for frame in read_i32(&raw).chunks_exact(samples_per_frame) {
                encoder
                    .encode_i24_into(frame, args.frame, &mut packet)
                    .unwrap_or_else(|e| fail(format!("encode: {e}")));
                emit(&packet, &mut out);
            }
        }
        "f32" => {
            for frame in read_f32(&raw).chunks_exact(samples_per_frame) {
                encoder
                    .encode_f32_into(frame, args.frame, &mut packet)
                    .unwrap_or_else(|e| fail(format!("encode: {e}")));
                emit(&packet, &mut out);
            }
        }
        _ => usage(),
    }
    fs::write(&args.output, out).unwrap_or_else(|e| fail(format!("{}: {e}", args.output)));
}

fn decode(args: &Args) {
    let data = fs::read(&args.input).unwrap_or_else(|e| fail(format!("{}: {e}", args.input)));
    let mut decoder = Decoder::new_custom(48_000, args.channels, args.frame)
        .unwrap_or_else(|e| fail(format!("decoder: {e}")));
    decoder
        .set_custom_signalling(args.signalling)
        .unwrap_or_else(|e| fail(format!("signalling: {e}")));
    decoder
        .set_libopus_arithmetic(args.libopus_arith)
        .unwrap_or_else(|e| fail(format!("arith: {e}")));
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut index = 0usize;
    let mut s16 = Vec::new();
    let mut s32 = Vec::new();
    let mut f32s = Vec::new();
    while pos + 2 <= data.len() {
        let len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + len > data.len() {
            fail("truncated packet file".into());
        }
        let packet = &data[pos..pos + len];
        let result = match args.pcm.as_str() {
            "s16" => decoder.decode_i16_into(packet, false, &mut s16).map(|_| {
                s16.iter()
                    .for_each(|v| out.extend_from_slice(&v.to_le_bytes()))
            }),
            "s24" => decoder.decode_i24_into(packet, false, &mut s32).map(|_| {
                s32.iter()
                    .for_each(|v| out.extend_from_slice(&v.to_le_bytes()))
            }),
            "f32" => decoder.decode_f32_into(packet, false, &mut f32s).map(|_| {
                f32s.iter()
                    .for_each(|v| out.extend_from_slice(&v.to_le_bytes()))
            }),
            _ => usage(),
        };
        if let Err(error) = result {
            fail(format!("packet {index}: decode error {error}"));
        }
        pos += len;
        index += 1;
    }
    fs::write(&args.output, out).unwrap_or_else(|e| fail(format!("{}: {e}", args.output)));
}

fn main() {
    let args = parse();
    match args.command.as_str() {
        "encode" => encode(&args),
        "decode" => decode(&args),
        _ => usage(),
    }
}
