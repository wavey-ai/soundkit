use soundkit_opus::*;

#[test]
fn packet_parse_code0_code1_and_code2_edges() {
    let mut packet = [0u8; 1276 * 3];

    for i in 0..64 {
        packet[0] = i << 2;
        let parsed = parse_packet(&packet[..4]).unwrap();
        assert_eq!(parsed.frame_count(), 1, "code 0 toc {i}");
        assert_eq!(parsed.toc >> 2, i);
        assert_eq!(parsed.frames()[0].offset, 1);
        assert_eq!(parsed.frames()[0].data.len(), 3);
    }

    for i in 0..64 {
        packet[0] = (i << 2) + 1;
        for len in [1, 3, 2551] {
            let parsed = parse_packet(&packet[..len]).unwrap();
            assert_eq!(parsed.frame_count(), 2, "code 1 toc {i} len {len}");
            assert_eq!(parsed.frames()[0].data.len(), parsed.frames()[1].data.len());
            assert_eq!(parsed.frames()[0].data.len(), (len - 1) / 2);
        }
        for len in [0, 2, 2552, 2553] {
            assert_eq!(
                parse_packet(&packet[..len]).unwrap_err(),
                Error::InvalidPacket,
                "code 1 toc {i} len {len}"
            );
        }
    }

    for i in 0..64 {
        packet[0] = (i << 2) + 2;
        assert_eq!(
            parse_packet(&packet[..1]).unwrap_err(),
            Error::InvalidPacket
        );

        for frame0 in [0, 1, 251, 252, 700, 1274] {
            if frame0 < 252 {
                packet[1] = frame0 as u8;
                packet[2] = 0;
            } else {
                packet[1] = (252 + (frame0 & 3)) as u8;
                packet[2] = ((frame0 - 252) >> 2) as u8;
            }
            let header = if frame0 < 252 { 2 } else { 3 };
            let len = frame0 + header + 17;
            let parsed = parse_packet(&packet[..len]).unwrap();
            assert_eq!(parsed.frame_count(), 2, "code 2 toc {i} frame0 {frame0}");
            assert_eq!(parsed.frames()[0].data.len(), frame0);
            assert_eq!(parsed.frames()[1].data.len(), len - header - frame0);
            assert_eq!(
                parsed.frames()[1].offset,
                parsed.frames()[0].offset + parsed.frames()[0].data.len()
            );
        }
    }
}

#[test]
fn packet_parse_code3_padding_edges() {
    let mut packet = [0u8; 70_000];

    for i in 0..64 {
        packet[0] = (i << 2) + 3;
        assert_eq!(
            parse_packet(&packet[..1]).unwrap_err(),
            Error::InvalidPacket
        );

        packet[1] = 1;
        let parsed = parse_packet(&packet[..2 + 1275]).unwrap();
        assert_eq!(parsed.frame_count(), 1);
        assert_eq!(parsed.frames()[0].data.len(), 1275);

        packet[1] = 0x80 | 1 | 0x40;
        packet[2] = 72;
        let parsed = parse_packet(&packet[..2 + 1 + 72 + 33]).unwrap();
        assert_eq!(parsed.frame_count(), 1, "padding toc {i}");
        assert_eq!(parsed.frames()[0].data.len(), 33);
        assert_eq!(parsed.toc >> 2, i);
    }

    packet[0] = 3;
    packet[1] = 0x80 | 1 | 0x40;
    for b in &mut packet[2..127] {
        *b = 255;
    }
    assert_eq!(
        parse_packet(&packet[..127]).unwrap_err(),
        Error::InvalidPacket
    );
}

#[test]
fn packet_duration_helpers_match_upstream_api_expectations() {
    let mut packet = [0u8; 4];

    packet[0] = 0;
    assert_eq!(sample_count(&packet[..1], 48_000).unwrap(), 480);
    assert_eq!(sample_count(&packet[..1], 96_000).unwrap(), 960);
    assert_eq!(sample_count(&packet[..1], 32_000).unwrap(), 320);
    assert_eq!(sample_count(&packet[..1], 8_000).unwrap(), 80);

    packet[0] = 3;
    assert_eq!(
        sample_count(&packet[..1], 24_000).unwrap_err(),
        Error::InvalidPacket
    );

    packet[0] = (63 << 2) | 3;
    packet[1] = 63;
    assert_eq!(
        sample_count(&packet[..2], 48_000).unwrap_err(),
        Error::InvalidPacket
    );

    assert_eq!(frame_count(&[]).unwrap_err(), Error::InvalidPacket);
    assert_eq!(bandwidth(&[]).unwrap_err(), Error::InvalidPacket);
    assert_eq!(
        samples_per_frame(&[], 48_000).unwrap_err(),
        Error::InvalidPacket
    );
}

#[test]
fn repacketizer_pad_unpad_round_trip() {
    let mut packet = [0u8; 2048];
    packet[0] = 0x08 | 0x04;
    for (i, byte) in packet[1..101].iter_mut().enumerate() {
        *byte = (i & 0xff) as u8;
    }

    let original = &packet[..101];
    let padded = packet_pad(original, 357).unwrap();
    assert_eq!(padded.len(), 357);
    let unpadded = packet_unpad(&padded).unwrap();
    assert_eq!(unpadded, original);

    let mut rp = Repacketizer::new();
    rp.cat(original).unwrap();
    assert_eq!(rp.frame_count(), 1);
    assert_eq!(rp.out().unwrap(), original);
}

#[test]
fn packet_parser_rejects_padding_overflow_regression_shape() {
    const PACKET_SIZE: usize = 16_909_318;
    let mut input = vec![0xffu8; PACKET_SIZE];
    input[1] = 0x41;
    input[PACKET_SIZE - 1] = 0x0b;

    assert_eq!(parse_packet(&input).unwrap_err(), Error::InvalidPacket);
}
