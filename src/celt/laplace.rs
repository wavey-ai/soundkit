use crate::celt::entropy::{RangeDecoder, RangeEncoder};

const LAPLACE_LOG_MINP: i32 = 0;
const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;
const LAPLACE_NMIN: u32 = 16;

fn get_freq1(fs0: u32, decay: i32) -> u32 {
    let ft = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    ((ft as i32 * (16384 - decay)) >> 15) as u32
}

pub fn get_start_freq(decay: i32) -> u32 {
    let ft = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN + 1);
    let fs = (ft as i32 * (16384 - decay)) / (16384 + decay);
    (fs as u32) + LAPLACE_MINP
}

pub fn encode_laplace(enc: &mut RangeEncoder, value: i32, mut fs: u32, decay: i32) -> i32 {
    let mut fl = 0u32;
    let mut represented = value;
    let mut val = value;
    if val != 0 {
        let s = -i32::from(val < 0);
        val = (val + s) ^ s;
        fl = fs;
        fs = get_freq1(fs, decay);
        let mut i = 1;

        while fs > 0 && i < val {
            fs *= 2;
            fl += fs + 2 * LAPLACE_MINP;
            fs = ((fs as i32 * decay) >> 15) as u32;
            i += 1;
        }

        if fs == 0 {
            let ndi_max = (32768 - fl + LAPLACE_MINP - 1) >> LAPLACE_LOG_MINP;
            let ndi_max = (ndi_max as i32 - s) >> 1;
            let di = (val - i).min(ndi_max - 1);
            fl += (2 * di + 1 + s) as u32 * LAPLACE_MINP;
            fs = LAPLACE_MINP.min(32768 - fl);
            represented = (i + di + s) ^ s;
        } else {
            fs += LAPLACE_MINP;
            fl += fs & !(s as u32);
        }
    }
    enc.encode_bin(fl, fl + fs, 15);
    represented
}

pub fn decode_laplace(dec: &mut RangeDecoder, mut fs: u32, decay: i32) -> i32 {
    let mut val = 0i32;
    let fm = dec.decode_bin(15);
    let mut fl = 0u32;
    if fm >= fs {
        val += 1;
        fl = fs;
        fs = get_freq1(fs, decay) + LAPLACE_MINP;

        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = (((fs - 2 * LAPLACE_MINP) as i32 * decay) >> 15) as u32;
            fs += LAPLACE_MINP;
            val += 1;
        }

        if fs <= LAPLACE_MINP {
            let di = ((fm - fl) >> (LAPLACE_LOG_MINP + 1)) as i32;
            val += di;
            fl += 2 * di as u32 * LAPLACE_MINP;
        }
        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }
    dec.update(fl, (fl + fs).min(32768), 32768);
    val
}

pub fn encode_laplace_p0(enc: &mut RangeEncoder, value: i32, p0: u16, decay: u16) {
    let sign_icdf = [32768u16 - p0, (32768u16 - p0) / 2, 0];
    let s = if value == 0 {
        0
    } else if value > 0 {
        1
    } else {
        2
    };
    enc.encode_icdf16(s, &sign_icdf, 15);

    let mut value = value.abs();
    if value != 0 {
        let mut icdf = [0u16; 8];
        icdf[0] = 7.max(decay);
        for i in 1..7 {
            icdf[i] = (7 - i as u16).max(((icdf[i - 1] as u32 * decay as u32) >> 15) as u16);
        }
        icdf[7] = 0;
        value -= 1;
        loop {
            enc.encode_icdf16(value.min(7) as usize, &icdf, 15);
            value -= 7;
            if value < 0 {
                break;
            }
        }
    }
}

pub fn decode_laplace_p0(dec: &mut RangeDecoder, p0: u16, decay: u16) -> i32 {
    let sign_icdf = [32768u16 - p0, (32768u16 - p0) / 2, 0];
    let mut s = dec.decode_icdf16(&sign_icdf, 15) as i32;
    if s == 2 {
        s = -1;
    }
    if s == 0 {
        return 0;
    }

    let mut icdf = [0u16; 8];
    icdf[0] = 7.max(decay);
    for i in 1..7 {
        icdf[i] = (7 - i as u16).max(((icdf[i - 1] as u32 * decay as u32) >> 15) as u16);
    }
    icdf[7] = 0;

    let mut value = 1i32;
    loop {
        let v = dec.decode_icdf16(&icdf, 15) as i32;
        value += v;
        if v != 7 {
            break;
        }
    }
    s * value
}
