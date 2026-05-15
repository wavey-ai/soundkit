use libopus_rs::{Application, Decoder, Encoder, Error};

#[test]
fn encode_and_decode_report_unimplemented_until_signal_path_is_ported() {
    let mut encoder = Encoder::new(48_000, 2, Application::Audio).unwrap();
    let pcm = vec![0i16; 960 * 2];
    assert_eq!(encoder.encode_i16(&pcm, 960), Err(Error::Unimplemented));

    let mut decoder = Decoder::new(48_000, 2).unwrap();
    let packet = [0x04u8];
    assert_eq!(
        decoder.decode_i16(&packet, false),
        Err(Error::Unimplemented)
    );
}
