const EC_WINDOW_SIZE: u32 = 32;
const EC_UINT_BITS: i32 = 8;
const BITRES: i32 = 3;
const EC_SYM_BITS: i32 = 8;
const EC_CODE_BITS: i32 = 32;
const EC_SYM_MAX: u32 = (1 << EC_SYM_BITS) - 1;
const EC_CODE_SHIFT: i32 = EC_CODE_BITS - EC_SYM_BITS - 1;
const EC_CODE_TOP: u32 = 1 << (EC_CODE_BITS - 1);
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
const EC_CODE_EXTRA: i32 = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1;

#[inline]
pub fn ec_ilog(v: u32) -> i32 {
    if v == 0 {
        0
    } else {
        (u32::BITS - v.leading_zeros()) as i32
    }
}

#[inline]
fn imul32(a: u32, b: u32) -> u32 {
    a.wrapping_mul(b)
}

#[inline]
fn celt_udiv(n: u32, d: u32) -> u32 {
    n / d
}

#[inline]
fn mini(a: u32, b: u32) -> u32 {
    a.min(b)
}

#[derive(Clone, Debug)]
pub struct RangeEncoder {
    buf: Vec<u8>,
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: i32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: i32,
    error: i32,
}

impl RangeEncoder {
    pub fn new(size: usize) -> Self {
        Self {
            buf: vec![0; size],
            storage: size as u32,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: EC_CODE_BITS + 1,
            offs: 0,
            rng: EC_CODE_TOP,
            val: 0,
            ext: 0,
            rem: -1,
            error: 0,
        }
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buf[..self.storage as usize]
    }

    pub fn range_data(&self) -> &[u8] {
        self.buffer()
    }

    pub fn range_bytes(&self) -> u32 {
        self.offs
    }

    pub fn error(&self) -> i32 {
        self.error
    }

    pub fn storage_bytes(&self) -> usize {
        self.storage as usize
    }

    pub fn tell(&self) -> i32 {
        self.nbits_total - ec_ilog(self.rng)
    }

    pub fn tell_frac(&self) -> u32 {
        tell_frac(self.nbits_total, self.rng)
    }

    fn write_byte(&mut self, value: u32) -> i32 {
        if self.offs + self.end_offs >= self.storage {
            -1
        } else {
            self.buf[self.offs as usize] = value as u8;
            self.offs += 1;
            0
        }
    }

    fn write_byte_at_end(&mut self, value: u32) -> i32 {
        if self.offs + self.end_offs >= self.storage {
            -1
        } else {
            self.end_offs += 1;
            self.buf[(self.storage - self.end_offs) as usize] = value as u8;
            0
        }
    }

    fn carry_out(&mut self, c: i32) {
        if c != EC_SYM_MAX as i32 {
            let carry = c >> EC_SYM_BITS;
            if self.rem >= 0 {
                self.error |= self.write_byte((self.rem + carry) as u32);
            }
            if self.ext > 0 {
                let sym = (EC_SYM_MAX as i32 + carry) as u32 & EC_SYM_MAX;
                loop {
                    self.error |= self.write_byte(sym);
                    self.ext -= 1;
                    if self.ext == 0 {
                        break;
                    }
                }
            }
            self.rem = c & EC_SYM_MAX as i32;
        } else {
            self.ext = self.ext.wrapping_add(1);
        }
    }

    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.carry_out((self.val >> EC_CODE_SHIFT) as i32);
            self.val = (self.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.rng <<= EC_SYM_BITS;
            self.nbits_total += EC_SYM_BITS;
        }
    }

    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = celt_udiv(self.rng, ft);
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(imul32(r, ft - fl)));
            self.rng = imul32(r, fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(imul32(r, ft - fh));
        }
        self.normalize();
    }

    pub fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
        let r = self.rng >> bits;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(imul32(r, (1u32 << bits) - fl)));
            self.rng = imul32(r, fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(imul32(r, (1u32 << bits) - fh));
        }
        self.normalize();
    }

    pub fn encode_bit_logp(&mut self, val: bool, logp: u32) {
        let r = self.rng;
        let l = self.val;
        let s = r >> logp;
        let r_minus_s = r - s;
        if val {
            self.val = l + r_minus_s;
        }
        self.rng = if val { s } else { r_minus_s };
        self.normalize();
    }

    pub fn encode_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            let prev = icdf[s - 1] as u32;
            let cur = icdf[s] as u32;
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(imul32(r, prev)));
            self.rng = imul32(r, prev - cur);
        } else {
            self.rng = self.rng.wrapping_sub(imul32(r, icdf[0] as u32));
        }
        self.normalize();
    }

    pub fn encode_icdf16(&mut self, s: usize, icdf: &[u16], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            let prev = icdf[s - 1] as u32;
            let cur = icdf[s] as u32;
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(imul32(r, prev)));
            self.rng = imul32(r, prev - cur);
        } else {
            self.rng = self.rng.wrapping_sub(imul32(r, icdf[0] as u32));
        }
        self.normalize();
    }

    pub fn encode_uint(&mut self, fl: u32, mut ft: u32) {
        ft -= 1;
        let mut ftb = ec_ilog(ft);
        if ftb > EC_UINT_BITS {
            ftb -= EC_UINT_BITS;
            let ft_top = (ft >> ftb) + 1;
            let fl_top = fl >> ftb;
            self.encode(fl_top, fl_top + 1, ft_top);
            self.encode_bits(fl & ((1u32 << ftb) - 1), ftb as u32);
        } else {
            self.encode(fl, fl + 1, ft + 1);
        }
    }

    pub fn encode_bits(&mut self, fl: u32, bits: u32) {
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if used as u32 + bits > EC_WINDOW_SIZE {
            loop {
                self.error |= self.write_byte_at_end(window & EC_SYM_MAX);
                window >>= EC_SYM_BITS;
                used -= EC_SYM_BITS;
                if used < EC_SYM_BITS {
                    break;
                }
            }
        }
        window |= fl << used;
        used += bits as i32;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    pub fn patch_initial_bits(&mut self, val: u32, nbits: u32) {
        let shift = EC_SYM_BITS as u32 - nbits;
        let mask = ((1u32 << nbits) - 1) << shift;
        if self.offs > 0 {
            self.buf[0] = ((self.buf[0] as u32 & !mask) | (val << shift)) as u8;
        } else if self.rem >= 0 {
            self.rem = ((self.rem as u32 & !mask) | (val << shift)) as i32;
        } else if self.rng <= (EC_CODE_TOP >> nbits) {
            self.val =
                (self.val & !(mask << EC_CODE_SHIFT)) | (val << (EC_CODE_SHIFT as u32 + shift));
        } else {
            self.error = -1;
        }
    }

    pub fn shrink(&mut self, size: usize) {
        let end_offs = self.end_offs as usize;
        if size > self.buf.len() {
            self.error = -1;
            return;
        }
        if end_offs > 0 {
            let src_start = self.storage as usize - end_offs;
            let dst_start = size - end_offs;
            self.buf
                .copy_within(src_start..src_start + end_offs, dst_start);
        }
        self.storage = size as u32;
    }

    pub fn finish(&mut self) {
        let mut l = EC_CODE_BITS - ec_ilog(self.rng);
        let mut msk = (EC_CODE_TOP - 1) >> l;
        let mut end = self.val.wrapping_add(msk) & !msk;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            msk >>= 1;
            end = self.val.wrapping_add(msk) & !msk;
        }
        while l > 0 {
            self.carry_out((end >> EC_CODE_SHIFT) as i32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l -= EC_SYM_BITS;
        }
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }

        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= EC_SYM_BITS {
            self.error |= self.write_byte_at_end(window & EC_SYM_MAX);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS;
        }
        if self.error == 0 {
            let start = self.offs as usize;
            let end_clear = (self.storage - self.end_offs) as usize;
            if end_clear > start {
                self.buf[start..end_clear].fill(0);
            }
            if used > 0 {
                if self.end_offs >= self.storage {
                    self.error = -1;
                } else {
                    let neg_l = -l;
                    if self.offs + self.end_offs >= self.storage && neg_l < used {
                        window = if neg_l > 0 {
                            window & ((1u32 << neg_l as u32) - 1)
                        } else {
                            0
                        };
                        self.error = -1;
                    }
                    let byte = (self.storage - self.end_offs - 1) as usize;
                    self.buf[byte] |= window as u8;
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct RangeDecoder {
    buf: Vec<u8>,
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: i32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: i32,
    error: i32,
}

impl RangeDecoder {
    pub fn new(buf: &[u8]) -> Self {
        let mut dec = Self {
            buf: buf.to_vec(),
            storage: buf.len() as u32,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: EC_CODE_BITS + 1
                - ((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) * EC_SYM_BITS,
            offs: 0,
            rng: 1u32 << EC_CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: 0,
        };
        dec.rem = dec.read_byte();
        dec.val = dec.rng - 1 - (dec.rem as u32 >> (EC_SYM_BITS - EC_CODE_EXTRA));
        dec.normalize();
        dec
    }

    pub fn error(&self) -> i32 {
        self.error
    }

    pub fn storage_bytes(&self) -> usize {
        self.storage as usize
    }

    pub fn tell(&self) -> i32 {
        self.nbits_total - ec_ilog(self.rng)
    }

    pub fn tell_frac(&self) -> u32 {
        tell_frac(self.nbits_total, self.rng)
    }

    fn read_byte(&mut self) -> i32 {
        if self.offs < self.storage {
            let value = self.buf[self.offs as usize];
            self.offs += 1;
            value as i32
        } else {
            0
        }
    }

    fn read_byte_from_end(&mut self) -> i32 {
        if self.end_offs < self.storage {
            self.end_offs += 1;
            self.buf[(self.storage - self.end_offs) as usize] as i32
        } else {
            0
        }
    }

    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.nbits_total += EC_SYM_BITS;
            self.rng <<= EC_SYM_BITS;
            let mut sym = self.rem;
            self.rem = self.read_byte();
            sym = (sym << EC_SYM_BITS | self.rem) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            self.val =
                ((self.val << EC_SYM_BITS) + (EC_SYM_MAX & !(sym as u32))) & (EC_CODE_TOP - 1);
        }
    }

    pub fn decode(&mut self, ft: u32) -> u32 {
        self.ext = celt_udiv(self.rng, ft);
        let s = self.val / self.ext;
        ft - mini(s + 1, ft)
    }

    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1u32 << bits) - mini(s + 1, 1u32 << bits)
    }

    pub fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = imul32(self.ext, ft - fh);
        self.val -= s;
        self.rng = if fl > 0 {
            imul32(self.ext, fh - fl)
        } else {
            self.rng - s
        };
        self.normalize();
    }

    pub fn decode_bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if !ret {
            self.val = d - s;
        }
        self.rng = if ret { s } else { r - s };
        self.normalize();
        ret
    }

    pub fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        let r = self.rng >> ftb;
        let d = self.val;
        let mut s = self.rng;
        let mut ret = 0usize;
        let mut t;
        loop {
            t = s;
            s = imul32(r, icdf[ret] as u32);
            if d >= s {
                break;
            }
            ret += 1;
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret
    }

    pub fn decode_icdf16(&mut self, icdf: &[u16], ftb: u32) -> usize {
        let r = self.rng >> ftb;
        let d = self.val;
        let mut s = self.rng;
        let mut ret = 0usize;
        let mut t;
        loop {
            t = s;
            s = imul32(r, icdf[ret] as u32);
            if d >= s {
                break;
            }
            ret += 1;
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret
    }

    pub fn decode_uint(&mut self, mut ft: u32) -> u32 {
        ft -= 1;
        let mut ftb = ec_ilog(ft);
        if ftb > EC_UINT_BITS {
            ftb -= EC_UINT_BITS;
            let ft_top = (ft >> ftb) + 1;
            let s = self.decode(ft_top);
            self.update(s, s + 1, ft_top);
            let t = (s << ftb) | self.decode_bits(ftb as u32);
            if t <= ft {
                t
            } else {
                self.error = 1;
                ft
            }
        } else {
            ft += 1;
            let s = self.decode(ft);
            self.update(s, s + 1, ft);
            s
        }
    }

    pub fn decode_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if (available as u32) < bits {
            loop {
                window |= (self.read_byte_from_end() as u32) << available;
                available += EC_SYM_BITS;
                if available as u32 > EC_WINDOW_SIZE - EC_SYM_BITS as u32 {
                    break;
                }
            }
        }
        let ret = window & ((1u32 << bits) - 1);
        window >>= bits;
        available -= bits as i32;
        self.end_window = window;
        self.nend_bits = available;
        self.nbits_total += bits as i32;
        ret
    }
}

fn tell_frac(nbits_total: i32, rng: u32) -> u32 {
    const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
    let nbits = (nbits_total as u32) << BITRES;
    let mut l = ec_ilog(rng);
    let r = rng >> (l - 16);
    let mut b = ((r >> 12) - 8) as usize;
    b += usize::from(r > CORRECTION[b]);
    l = (l << 3) + b as i32;
    nbits - l as u32
}
