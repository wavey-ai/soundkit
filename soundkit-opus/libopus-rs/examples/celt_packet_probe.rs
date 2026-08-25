use libopus_rs::celt::codec::{decode_spectral_frame, CeltFrameConfig};
use libopus_rs::celt::modes::CeltMode;
use std::collections::HashMap;
use std::env;
use std::io::{self, Read};

struct StreamState {
    old_band_e: Vec<f32>,
    seed: u32,
}

fn usage() -> ! {
    eprintln!(
        "usage: celt_packet_probe <frame_size> <bitrate> [max_frame]\n\
         Reads raw_celt_bench --dump-packets TSV from stdin."
    );
    std::process::exit(2);
}

fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    for i in (0..hex.len()).step_by(2) {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    Some(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 3 || args.len() > 4 {
        usage();
    }
    let frame_size = args[1].parse::<usize>().unwrap_or_else(|_| usage());
    let bitrate = args[2].parse::<i32>().unwrap_or_else(|_| usage());
    let max_frame = args
        .get(3)
        .map(|arg| arg.parse::<usize>().unwrap_or_else(|_| usage()));

    let mode = CeltMode::standard_48k();
    let lm = match frame_size {
        120 => 0,
        240 => 1,
        480 => 2,
        960 => 3,
        _ => usage(),
    };

    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    println!(
        "impl\tmode\tframe_size\tbitrate\tframe\tlen\ttransient\tprefilter\tpitch\tqgain\ttapset\tspread\ttrim\tcoded\tintensity\tdual\tbalance\tebits\tpulses\tcollapse"
    );
    let mut states = HashMap::<String, StreamState>::new();
    for line in input.lines() {
        let cols = line.split('\t').collect::<Vec<_>>();
        if cols.len() != 8 || cols[0] == "impl" {
            continue;
        }
        if cols[2].parse::<usize>().ok() != Some(frame_size)
            || cols[4].parse::<i32>().ok() != Some(bitrate)
        {
            continue;
        }
        let frame = match cols[5].parse::<usize>() {
            Ok(frame) => frame,
            Err(_) => continue,
        };
        if max_frame.is_some_and(|max_frame| frame > max_frame) {
            continue;
        }
        let Some(packet) = decode_hex(cols[7]) else {
            continue;
        };
        if packet.is_empty() {
            continue;
        }
        let channels = if packet[0] & 0x04 != 0 { 2 } else { 1 };
        let config = CeltFrameConfig::new(&mode, lm, channels, packet.len() - 1)?;
        let key = format!("{}\t{}\t{}\t{}", cols[0], cols[1], cols[2], cols[4]);
        let state = states.entry(key).or_insert_with(|| StreamState {
            old_band_e: vec![0.0; channels * mode.nb_ebands],
            seed: 0,
        });
        let decoded = decode_spectral_frame(
            &mode,
            &config,
            &packet[1..],
            &mut state.old_band_e,
            &mut state.seed,
        )?;
        let ebits = decoded.allocation.ebits[..decoded.allocation.coded_bands]
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let pulses = decoded.allocation.pulses[..decoded.allocation.coded_bands]
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let collapse = decoded.collapse_masks[..channels * decoded.allocation.coded_bands]
            .iter()
            .map(|mask| format!("{mask:02x}"))
            .collect::<Vec<_>>()
            .join(",");
        let (pitch, qgain, tapset) = decoded
            .prefilter
            .map(|prefilter| (prefilter.pitch, prefilter.qgain, prefilter.tapset))
            .unwrap_or((0, 0, 0));
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            cols[0],
            cols[1],
            frame_size,
            bitrate,
            frame,
            packet.len(),
            i32::from(decoded.is_transient),
            i32::from(decoded.prefilter.is_some()),
            pitch,
            qgain,
            tapset,
            decoded.spread,
            decoded.alloc_trim,
            decoded.allocation.coded_bands,
            decoded.allocation.intensity,
            i32::from(decoded.allocation.dual_stereo),
            decoded.allocation.balance,
            ebits,
            pulses,
            collapse
        );
    }
    Ok(())
}
