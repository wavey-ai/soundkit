use soundkit_alac::AlacPacketDecoder;
use soundkit_audio_demux::Mp4MediaIndex;
use std::env;
use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

struct AlacInput {
    cookie: Vec<u8>,
    packets: Vec<Vec<u8>>,
}

fn load_input(bytes: &[u8]) -> AlacInput {
    let index = Mp4MediaIndex::from_file(bytes).expect("index M4A");
    let track = index
        .tracks
        .iter()
        .find(|track| track.codec == "alac")
        .expect("ALAC track");
    let packets = index
        .samples
        .iter()
        .enumerate()
        .filter(|(_, sample)| sample.track_id == track.track_id)
        .map(|(sample_index, sample)| {
            let start = usize::try_from(sample.absolute_offset).expect("packet offset");
            let end = start
                .checked_add(sample.size as usize)
                .expect("packet range");
            index
                .packet_from_sample_bytes(sample_index, &bytes[start..end])
                .expect("extract ALAC packet")
                .data
        })
        .collect();
    AlacInput {
        cookie: track.codec_private.clone(),
        packets,
    }
}

fn checksum_pcm(pcm: &[u8], bit_depth: u8) -> i64 {
    match bit_depth {
        16 => pcm
            .chunks_exact(2)
            .map(|sample| i64::from(i16::from_le_bytes([sample[0], sample[1]])))
            .sum(),
        24 => pcm
            .chunks_exact(3)
            .map(|sample| {
                let value = i32::from_le_bytes([sample[0], sample[1], sample[2], 0]);
                i64::from((value << 8) >> 8)
            })
            .sum(),
        32 => pcm
            .chunks_exact(4)
            .map(|sample| {
                i64::from(i32::from_le_bytes([
                    sample[0], sample[1], sample[2], sample[3],
                ]))
            })
            .sum(),
        _ => panic!("unsupported ALAC PCM depth {bit_depth}"),
    }
}

fn decode_once(
    decoder: &mut AlacPacketDecoder,
    packets: &[Vec<u8>],
    pcm: &mut Vec<u8>,
    calculate_checksum: bool,
    mut output: Option<&mut fs::File>,
) -> (usize, i64) {
    let mut samples = 0usize;
    let mut checksum = 0i64;
    let bytes_per_sample = usize::from(decoder.bit_depth().div_ceil(8));
    for packet in packets {
        let decoded = decoder
            .decode_packet_into(black_box(packet), pcm)
            .expect("decode ALAC");
        let decoded = black_box(decoded);
        samples = samples.wrapping_add(decoded.len() / bytes_per_sample);
        if calculate_checksum {
            checksum = checksum.wrapping_add(checksum_pcm(decoded, decoder.bit_depth()));
        }
        if let Some(file) = output.as_deref_mut() {
            file.write_all(decoded).expect("write PCM output");
        }
    }
    (samples, checksum)
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .expect("usage: decode_bench <input.m4a> [iterations] [output.pcm]");
    let iterations = args
        .next()
        .map(|value| value.parse::<usize>().expect("integer iterations"))
        .unwrap_or(50);
    let output_path = args.next();
    let bytes = fs::read(&path).expect("read ALAC input");
    let input = load_input(&bytes);
    let mut decoder = AlacPacketDecoder::new(&input.cookie).expect("initialize ALAC");
    let mut pcm = Vec::new();

    for _ in 0..3 {
        black_box(decode_once(
            &mut decoder,
            &input.packets,
            &mut pcm,
            false,
            None,
        ));
    }

    let started = Instant::now();
    let mut total_samples = 0usize;
    for _ in 0..iterations {
        total_samples = total_samples.wrapping_add(
            black_box(decode_once(
                &mut decoder,
                &input.packets,
                &mut pcm,
                false,
                None,
            ))
            .0,
        );
    }
    let elapsed = started.elapsed();
    let checksum = decode_once(&mut decoder, &input.packets, &mut pcm, true, None).1;

    if let Some(output_path) = output_path {
        let mut output = fs::File::create(output_path).expect("create PCM output");
        decode_once(
            &mut decoder,
            &input.packets,
            &mut pcm,
            false,
            Some(&mut output),
        );
    }

    println!(
        "implementation=soundkit-alac codec=alac operation=decode input_bytes={} packets={} iterations={} samples={} sample_rate={} channels={} bit_depth={} elapsed_ns={} checksum={}",
        bytes.len(),
        input.packets.len(),
        iterations,
        total_samples,
        decoder.sample_rate(),
        decoder.channels(),
        decoder.bit_depth(),
        elapsed.as_nanos(),
        checksum,
    );
}
