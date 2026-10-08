// The develop plan on the GPU. Each function mirrors the one of the same name
// in src/develop.rs; parameter indices are those of src/gpu.rs (module p).

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> tables: array<f32>;
@group(0) @binding(2) var<storage, read> source: array<u32>;
@group(0) @binding(3) var<storage, read_write> planes: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
struct Pass { op: u32, a: u32, b: u32, c: u32 }
@group(0) @binding(6) var<uniform> pass_args: Pass;

const WIDTH = 0u; const HEIGHT = 1u;
const RX = 2u; const RY = 3u; const RW = 4u; const RH = 5u;
const WX0 = 6u; const WY0 = 7u; const WX1 = 8u; const WY1 = 9u;
const FW = 10u; const FH = 11u; const FORMAT = 12u; const HAS_ALPHA = 13u; const EXACT = 14u;
const XS = 15u; const YS = 16u; const M = 17u; const EXPOSURE = 26u;
const UNTOUCHED_ON = 27u; const UNTOUCHED = 28u;
const BIT_DEPTH = 37u; const MAX = 38u; const CLIPPING = 39u;
const HAZE_ON = 40u; const HAZE_AMOUNT = 41u; const AIR = 42u; const HAZE_GW = 45u; const HAZE_GH = 46u; const HAZE_STEP = 47u;
const BASE_ON = 48u; const BASE_GW = 49u; const BASE_GH = 50u; const BASE_STEP = 51u;
const CLAR_ON = 52u; const CLAR_GW = 53u; const CLAR_GH = 54u; const CLAR_STEP = 55u;
const SHADOWS = 56u; const HIGHLIGHTS = 57u; const CLARITY = 58u; const TEXTURE = 59u; const SHARPEN = 60u; const SHARPEN_MASKING = 61u;
const NOISE_ON = 62u; const NOISE_K = 63u; const CHROMA_ON = 64u; const CHROMA_AMOUNT = 65u;
const TONE_SCENE = 66u; const TONE_SCALE = 67u; const CLIP_AT = 68u; const BLACK = 69u; const WHITE = 70u;
const CURVES_ON = 71u;
const COLOUR_ON = 72u; const MONO = 73u; const MIXER_ACTIVE = 74u; const SATURATION = 75u; const VIBRANCE = 76u; const GRADING = 77u; const BALANCE = 78u;
const MIXER = 79u; const TINT = 103u; const LIFT = 111u;
const VIG_AMOUNT = 115u; const VIG_MID = 116u; const VIG_FEATHER = 117u; const GRAIN_AMOUNT = 118u; const GRAIN_CELL = 119u;
const BAND_HUES = 120u;
const TEXTURE_ON = 128u; const SHARP_ON = 129u;
const OFF_DECODE = 130u; const OFF_ENCODE = 131u; const OFF_TONE = 132u; const OFF_CURVES = 133u;
const OFF_SHADOW_W = 134u; const OFF_HIGHLIGHT_W = 135u; const OFF_HAZE = 136u;
const OFF_BASE_A = 137u; const OFF_BASE_B = 138u; const OFF_CLAR_A = 139u; const OFF_CLAR_B = 140u;
const SRC_TABLE = 141u; const SRC_ALPHA = 142u; const PLANE_SIZE = 143u; const NEED_CHROMA = 144u;

const PL_LUM = 0u; const PL_CHROMA_R = 1u; const PL_CHROMA_B = 2u; const PL_TEXTURE = 3u; const PL_SHARP = 4u;
const PL_NOISE_A = 5u; const PL_NOISE_B = 6u; const PL_T2 = 9u; const PL_T3 = 10u;

const TABLE = 4096u;
const SCENE_TABLE = 4096u;
const SCENE_LO = -16.0; const SCENE_HI = 8.0;
const ZONE_STEPS = 1024u;
const GREY = 0.18;
const PI = 3.14159265358979;

fn P(i: u32) -> u32 { return params[i]; }
fn F(i: u32) -> f32 { return bitcast<f32>(params[i]); }

fn ease(a: f32, b: f32, x: f32) -> f32 {
    var d = b - a; if (d == 0.0) { d = 1e-6; }
    let t = clamp((x - a) / d, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
fn round_up(x: f32) -> u32 { return u32(floor(x + 0.5)); }
fn work_luma(c: vec3f) -> f32 { return 0.2627216817 * c.x + 0.6779893173 * c.y + 0.0592890010 * c.z; }
fn work_to_srgb(c: vec3f) -> vec3f {
    return vec3f(1.66036293778 * c.x - 0.587540226938 * c.y - 0.0728227108432 * c.z,
        -0.124563751098 * c.x + 1.1329115641 * c.y - 0.00834781299972 * c.z,
        -0.0181565968008 * c.x - 0.100601710304 * c.y + 1.1187583071 * c.z);
}
fn luma(c: vec3f) -> f32 { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }
fn lookup(off: u32, p: f32) -> f32 {
    let k = min(u32(p), TABLE - 1u);
    return tables[off + k] + (tables[off + k + 1u] - tables[off + k]) * (p - f32(k));
}
fn decode(v: f32) -> f32 { return lookup(P(OFF_DECODE), clamp(v, 0.0, 1.0) * f32(TABLE)); }
fn encode(v: f32) -> f32 { return lookup(P(OFF_ENCODE), sqrt(clamp(v, 0.0, 1.0)) * f32(TABLE)); }

// ---------------------------------------------------------------- sampling

fn texel(i: u32) -> f32 {
    let format = P(FORMAT);
    if (format == 0u) { return bitcast<f32>(source[i]); }
    if (format == 1u) { return f32((source[i / 2u] >> (16u * (i % 2u))) & 0xffffu); }
    if (format == 2u) { return bitcast<f32>(source[P(SRC_TABLE) + ((source[i / 4u] >> (8u * (i % 4u))) & 0xffu)]); }
    return bitcast<f32>(source[P(SRC_TABLE) + ((source[i / 2u] >> (16u * (i % 2u))) & 0xffffu)]);
}
fn opacity(pixel: u32) -> f32 {
    return f32((source[P(SRC_ALPHA) + pixel / 4u] >> (8u * (pixel % 4u))) & 0xffu);
}
fn pixel_rgb(pixel: u32) -> vec3f { return vec3f(texel(pixel * 3u), texel(pixel * 3u + 1u), texel(pixel * 3u + 2u)); }

// The working-space colour of output pixel (x, y) through matrix `m` (a
// parameter index), times `gain`, with its opacity.
fn read_pixel(x: u32, y: u32, m: u32, gain: f32) -> vec4f {
    let fw = P(FW); let fh = P(FH);
    var c: vec3f; var alpha = 1.0;
    if (P(EXACT) == 1u) {
        let p = y * fw + x;
        c = pixel_rgb(p);
        if (P(HAS_ALPHA) == 1u) { alpha = opacity(p) / 255.0; }
    } else {
        let sx = clamp((f32(x) + 0.5) * F(XS) - 0.5, 0.0, f32(fw - 1u));
        let sy = clamp((f32(y) + 0.5) * F(YS) - 0.5, 0.0, f32(fh - 1u));
        let x0 = u32(sx); let y0 = u32(sy);
        let x1 = min(x0 + 1u, fw - 1u); let y1 = min(y0 + 1u, fh - 1u);
        let dx = sx - f32(x0); let dy = sy - f32(y0);
        let w00 = (1.0 - dx) * (1.0 - dy); let w10 = dx * (1.0 - dy); let w01 = (1.0 - dx) * dy; let w11 = dx * dy;
        let p00 = y0 * fw + x0; let p10 = y0 * fw + x1; let p01 = y1 * fw + x0; let p11 = y1 * fw + x1;
        c = pixel_rgb(p00) * w00 + pixel_rgb(p10) * w10 + pixel_rgb(p01) * w01 + pixel_rgb(p11) * w11;
        if (P(HAS_ALPHA) == 1u) { alpha = (opacity(p00) * w00 + opacity(p10) * w10 + opacity(p01) * w01 + opacity(p11) * w11) / 255.0; }
    }
    let r = (F(m) * c.x + F(m + 1u) * c.y + F(m + 2u) * c.z) * gain;
    let g = (F(m + 3u) * c.x + F(m + 4u) * c.y + F(m + 5u) * c.z) * gain;
    let b = (F(m + 6u) * c.x + F(m + 7u) * c.y + F(m + 8u) * c.z) * gain;
    return vec4f(max(r, 0.0), max(g, 0.0), max(b, 0.0), alpha);
}

fn grid_at(off: u32, gw: u32, gh: u32, step: f32, x: u32, y: u32) -> f32 {
    let gx = clamp((f32(x) + 0.5) / step - 0.5, 0.0, f32(gw - 1u));
    let gy = clamp((f32(y) + 0.5) / step - 0.5, 0.0, f32(gh - 1u));
    let x0 = u32(gx); let y0 = u32(gy);
    let x1 = min(x0 + 1u, gw - 1u); let y1 = min(y0 + 1u, gh - 1u);
    let dx = gx - f32(x0); let dy = gy - f32(y0);
    return (tables[off + y0 * gw + x0] * (1.0 - dx) + tables[off + y0 * gw + x1] * dx) * (1.0 - dy)
        + (tables[off + y1 * gw + x0] * (1.0 - dx) + tables[off + y1 * gw + x1] * dx) * dy;
}

fn apply_haze(c: vec3f, x: u32, y: u32, light: f32) -> vec3f {
    let air = vec3f(F(AIR), F(AIR + 1u), F(AIR + 2u)) * light;
    let amount = F(HAZE_AMOUNT);
    if (amount > 0.0) {
        let t = max(1.0 - (1.0 - grid_at(P(OFF_HAZE), P(HAZE_GW), P(HAZE_GH), F(HAZE_STEP), x, y)) * amount, 0.1);
        return max((c - air) / t + air, vec3f(0.0));
    }
    let k = -amount * 0.6;
    return c * (1.0 - k) + air * k;
}

fn exposed(x: u32, y: u32) -> vec4f {
    var s = read_pixel(x, y, M, F(EXPOSURE));
    if (P(HAZE_ON) == 1u) { s = vec4f(apply_haze(s.xyz, x, y, F(EXPOSURE)), s.w); }
    return s;
}

fn log_luminance(c: vec3f) -> f32 { return log2(max(work_luma(c), 1e-6) / GREY); }

// ------------------------------------------------------------ window passes

fn window_size() -> vec2u { return vec2u(P(WX1) - P(WX0), P(WY1) - P(WY0)); }
fn plane(index: u32) -> u32 { return index * P(PLANE_SIZE); }

@compute @workgroup_size(8, 8)
fn prepare(@builtin(global_invocation_id) id: vec3u) {
    let size = window_size();
    if (id.x >= size.x || id.y >= size.y) { return; }
    let x = P(WX0) + id.x; let y = P(WY0) + id.y;
    let c = exposed(x, y).xyz;
    let i = id.y * size.x + id.x;
    planes[plane(PL_LUM) + i] = log_luminance(c);
    if (P(NEED_CHROMA) == 1u) {
        let l = work_luma(c) + 1e-6;
        planes[plane(PL_CHROMA_R) + i] = c.x / l;
        planes[plane(PL_CHROMA_B) + i] = c.z / l;
    }
}

@compute @workgroup_size(8, 8)
fn box_rows(@builtin(global_invocation_id) id: vec3u) {
    let size = window_size();
    if (id.x >= size.x || id.y >= size.y) { return; }
    let src = plane(pass_args.a); let dst = plane(pass_args.b); let r = i32(pass_args.c);
    let row = id.y * size.x;
    var sum = 0.0;
    for (var k = -r; k <= r; k++) { sum += planes[src + row + u32(clamp(i32(id.x) + k, 0, i32(size.x) - 1))]; }
    planes[dst + row + id.x] = sum / f32(2 * r + 1);
}

@compute @workgroup_size(8, 8)
fn box_columns(@builtin(global_invocation_id) id: vec3u) {
    let size = window_size();
    if (id.x >= size.x || id.y >= size.y) { return; }
    let src = plane(pass_args.a); let dst = plane(pass_args.b); let r = i32(pass_args.c);
    var sum = 0.0;
    for (var k = -r; k <= r; k++) { sum += planes[src + u32(clamp(i32(id.y) + k, 0, i32(size.y) - 1)) * size.x + id.x]; }
    planes[dst + id.y * size.x + id.x] = sum / f32(2 * r + 1);
}

@compute @workgroup_size(8, 8)
fn square(@builtin(global_invocation_id) gid: vec3u) {
    let size = window_size();
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let id = vec3u(gid.y * size.x + gid.x, 0u, 0u);
    let v = planes[plane(pass_args.a) + id.x];
    planes[plane(pass_args.b) + id.x] = v * v;
}

@compute @workgroup_size(8, 8)
fn copy_plane(@builtin(global_invocation_id) gid: vec3u) {
    let size = window_size();
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let id = vec3u(gid.y * size.x + gid.x, 0u, 0u);
    planes[plane(pass_args.b) + id.x] = planes[plane(pass_args.a) + id.x];
}

@compute @workgroup_size(8, 8)
fn guided(@builtin(global_invocation_id) gid: vec3u) {
    let size = window_size();
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let id = vec3u(gid.y * size.x + gid.x, 0u, 0u);
    let mean = planes[plane(pass_args.a) + id.x];
    let correlation = planes[plane(pass_args.b) + id.x];
    let eps = bitcast<f32>(pass_args.c);
    let variance = max(correlation - mean * mean, 0.0);
    let a = variance / (variance + eps);
    planes[plane(PL_T2) + id.x] = a;
    planes[plane(PL_T3) + id.x] = mean - a * mean;
}

// ---------------------------------------------------------------- colour

fn to_oklab(c: vec3f) -> vec3f {
    let l = 0.4122214708 * c.x + 0.5363325363 * c.y + 0.0514459929 * c.z;
    let m = 0.2119034982 * c.x + 0.6806995451 * c.y + 0.1073969566 * c.z;
    let s = 0.0883024619 * c.x + 0.2817188376 * c.y + 0.6299787005 * c.z;
    let l3 = sign(l) * pow(abs(l), 1.0 / 3.0); let m3 = sign(m) * pow(abs(m), 1.0 / 3.0); let s3 = sign(s) * pow(abs(s), 1.0 / 3.0);
    return vec3f(0.2104542553 * l3 + 0.7936177850 * m3 - 0.0040720468 * s3,
        1.9779984951 * l3 - 2.4285922050 * m3 + 0.4505937099 * s3,
        0.0259040371 * l3 + 0.7827717662 * m3 - 0.8086757660 * s3);
}
fn from_oklab(lab: vec3f) -> vec3f {
    let l0 = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m0 = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s0 = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    let l = l0 * l0 * l0; let m = m0 * m0 * m0; let s = s0 * s0 * s0;
    return vec3f(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}
fn in_gamut(c: vec3f) -> bool { return all(c >= vec3f(-1e-6)) && all(c <= vec3f(1.0 + 1e-6)); }
fn fit_gamut(lab_in: vec3f) -> vec3f {
    let l = clamp(lab_in.x, 0.0, 1.0);
    let first = from_oklab(vec3f(l, lab_in.yz));
    if (in_gamut(first)) { return first; }
    var lo = 0.0; var hi = 1.0;
    for (var i = 0; i < 12; i++) {
        let k = (lo + hi) * 0.5;
        if (in_gamut(from_oklab(vec3f(l, lab_in.yz * k)))) { lo = k; } else { hi = k; }
    }
    return clamp(from_oklab(vec3f(l, lab_in.yz * lo)), vec3f(0.0), vec3f(1.0));
}
fn hue_of(a: f32, b: f32) -> f32 {
    if (a == 0.0 && b == 0.0) { return 0.0; }
    var d = degrees(atan2(b, a));
    if (d < 0.0) { d += 360.0; }
    return d;
}
fn outside(s: vec3f) -> bool { return s.x < -1e-5 || s.y < -1e-5 || s.z < -1e-5; }
fn roll_off(s: vec3f, white: f32) -> vec3f {
    let peak = max(s.x, max(s.y, s.z));
    if (peak > white) {
        let k = clamp((peak - white) / peak, 0.0, 1.0);
        let lab = to_oklab(s / peak);
        return fit_gamut(vec3f(lab.x + (1.0 - lab.x) * k, lab.yz * (1.0 - k))) * white;
    }
    return fit_gamut(to_oklab(s / white)) * white;
}

fn band_weight(hue: f32, band: u32) -> f32 {
    var h = hue;
    if (h < F(BAND_HUES)) { h += 360.0; }
    for (var i = 0u; i < 8u; i++) {
        let start = F(BAND_HUES + i);
        var to = F(BAND_HUES) + 360.0;
        if (i + 1u < 8u) { to = F(BAND_HUES + i + 1u); }
        if (h >= start && h < to) {
            let t = (h - start) / (to - start);
            let near = 0.5 + 0.5 * cos(PI * t);
            if (band == i) { return near; }
            if (band == (i + 1u) % 8u) { return 1.0 - near; }
            return 0.0;
        }
    }
    return 0.0;
}

fn colour_stage(out_in: vec3f) -> vec3f {
    let lab = to_oklab(vec3f(decode(out_in.x), decode(out_in.y), decode(out_in.z)));
    var ll = lab.x; var a = lab.y; var b = lab.z;
    let chroma = sqrt(a * a + b * b);
    let mono = P(MONO) == 1u; let mixer = P(MIXER_ACTIVE) == 1u; let vibrance = F(VIBRANCE);
    var hue = 0.0;
    if (mixer || mono || vibrance != 0.0) { hue = hue_of(a, b); }
    let colourful = ease(0.0, 0.06, chroma);
    var hue_shift = 0.0; var sat_shift = 0.0; var lum_shift = 0.0;
    if (mixer || mono) {
        for (var k = 0u; k < 8u; k++) {
            let w = band_weight(hue, k);
            if (w == 0.0) { continue; }
            hue_shift += w * F(MIXER + k * 3u);
            sat_shift += w * F(MIXER + k * 3u + 1u);
            lum_shift += w * F(MIXER + k * 3u + 2u);
        }
    }
    if (mono) {
        ll *= exp2(lum_shift * colourful * 0.8);
        a = 0.0; b = 0.0;
    } else {
        var factor = max(1.0 + F(SATURATION), 0.0) * max(1.0 + sat_shift, 0.0);
        if (vibrance != 0.0) {
            let skin = ease(30.0, 45.0, hue) * (1.0 - ease(75.0, 90.0, hue)) * ease(0.02, 0.05, chroma) * (1.0 - ease(0.18, 0.26, chroma));
            if (vibrance > 0.0) { factor *= 1.0 + vibrance * (1.0 - ease(0.0, 0.25, chroma)) * (1.0 - 0.8 * skin); } else { factor *= 1.0 + vibrance; }
        }
        if (lum_shift != 0.0) { ll *= exp2(lum_shift * colourful * 0.6); }
        if (hue_shift != 0.0) {
            let turn = radians(hue_shift * colourful);
            let s = sin(turn); let c = cos(turn);
            let turned = a * c - b * s; b = a * s + b * c; a = turned;
        }
        a *= factor; b *= factor;
    }
    if (P(GRADING) == 1u) {
        let k = round_up(clamp(ll - F(BALANCE), 0.0, 1.0) * f32(ZONE_STEPS));
        let ws = tables[P(OFF_SHADOW_W) + k]; let wh = tables[P(OFF_HIGHLIGHT_W) + k];
        let wm = max(1.0 - ws - wh, 0.0);
        a += F(TINT) * ws + F(TINT + 2u) * wm + F(TINT + 4u) * wh + F(TINT + 6u);
        b += F(TINT + 1u) * ws + F(TINT + 3u) * wm + F(TINT + 5u) * wh + F(TINT + 7u);
        ll += F(LIFT) * ws + F(LIFT + 1u) * wm + F(LIFT + 2u) * wh + F(LIFT + 3u);
    }
    let linear = fit_gamut(vec3f(ll, a, b));
    return vec3f(encode(linear.x), encode(linear.y), encode(linear.z));
}

fn hash(x: u32, y: u32) -> f32 {
    var h = x * 374761393u + y * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return f32(h ^ (h >> 16u)) / 4294967295.0 * 2.0 - 1.0;
}
fn grain_at(x: u32, y: u32, cell: f32) -> f32 {
    let gx = f32(x) / cell; let gy = f32(y) / cell;
    let fx = floor(gx); let fy = floor(gy);
    let dx = gx - fx; let dy = gy - fy;
    let sx = dx * dx * (3.0 - 2.0 * dx); let sy = dy * dy * (3.0 - 2.0 * dy);
    let x0 = u32(fx); let y0 = u32(fy);
    let top = hash(x0, y0) * (1.0 - sx) + hash(x0 + 1u, y0) * sx;
    let bottom = hash(x0, y0 + 1u) * (1.0 - sx) + hash(x0 + 1u, y0 + 1u) * sx;
    return top * (1.0 - sy) + bottom * sy;
}

fn curve_read(channel: u32, v: f32) -> f32 {
    let off = P(OFF_CURVES) + channel * (TABLE + 1u);
    return lookup(off, clamp(v, 0.0, 1.0) * f32(TABLE));
}

struct Shaded { out: vec3f, hi: bool, lo: bool }

fn shade(rgb: vec3f, x: u32, y: u32, w: u32) -> Shaded {
    var c = rgb;
    let l = log_luminance(c);
    var delta = 0.0;
    if (P(BASE_ON) == 1u) {
        let b = grid_at(P(OFF_BASE_A), P(BASE_GW), P(BASE_GH), F(BASE_STEP), x, y) * l + grid_at(P(OFF_BASE_B), P(BASE_GW), P(BASE_GH), F(BASE_STEP), x, y);
        let ws = (1.0 - ease(-5.0, 0.0, b)) * ease(-12.0, -8.0, b);
        let wh = ease(-0.5, 3.0, b);
        delta += F(SHADOWS) * 1.5 * ws + F(HIGHLIGHTS) * 1.5 * wh;
    }
    if (P(CLAR_ON) == 1u) {
        let b = grid_at(P(OFF_CLAR_A), P(CLAR_GW), P(CLAR_GH), F(CLAR_STEP), x, y) * l + grid_at(P(OFF_CLAR_B), P(CLAR_GW), P(CLAR_GH), F(CLAR_STEP), x, y);
        let detail = clamp(l - b, -2.0, 2.0);
        delta += F(CLARITY) * 0.7 * detail * (0.3 + 0.7 * exp(-(b / 3.0) * (b / 3.0)));
    }
    if (P(TEXTURE_ON) == 1u) {
        let detail = l - planes[plane(PL_TEXTURE) + w];
        delta += F(TEXTURE) * clamp(detail, -1.0, 1.0) * (1.0 - 0.5 * ease(0.5, 1.5, abs(detail)));
    }
    if (P(SHARP_ON) == 1u) {
        let detail = l - planes[plane(PL_SHARP) + w];
        var mask = 1.0;
        if (F(SHARPEN_MASKING) != 0.0) { mask = ease(0.0, F(SHARPEN_MASKING) * 0.15, abs(detail)); }
        delta += detail * F(SHARPEN) * mask;
    }
    if (P(NOISE_ON) == 1u) {
        delta += (planes[plane(PL_NOISE_A) + w] * l + planes[plane(PL_NOISE_B) + w] - l) * F(NOISE_K);
    }
    if (delta != 0.0) { c *= exp2(delta); }
    if (P(CHROMA_ON) == 1u) {
        let y0 = work_luma(c);
        let amount = F(CHROMA_AMOUNT);
        let r1 = planes[plane(PL_CHROMA_R) + w] * y0; let b1 = planes[plane(PL_CHROMA_B) + w] * y0;
        c.x += (r1 - c.x) * amount; c.z += (b1 - c.z) * amount;
        c.y = max((y0 - 0.2627216817 * c.x - 0.0592890010 * c.z) / 0.6779893173, 0.0);
    }

    var out: vec3f; var hi: bool; var lo: bool;
    if (P(TONE_SCENE) == 1u) {
        let yl = work_luma(c);
        let stops = log2(max(yl, 1e-9) / GREY);
        let pos = clamp((stops - SCENE_LO) / (SCENE_HI - SCENE_LO), 0.0, 1.0) * f32(SCENE_TABLE);
        let k = min(u32(pos), SCENE_TABLE - 1u);
        let off = P(OFF_TONE);
        let o = tables[off + k] + (tables[off + k + 1u] - tables[off + k]) * (pos - f32(k));
        var ratio = 0.0;
        if (yl > 1e-9) { ratio = o / yl; }
        let s = work_to_srgb(c * ratio);
        hi = max(s.x, max(s.y, s.z)) > 1.0;
        lo = o <= 0.0;
        var v = s;
        if (hi || outside(s)) { v = roll_off(s, 1.0); }
        out = vec3f(encode(v.x), encode(v.y), encode(v.z));
    } else {
        let s = work_to_srgb(c);
        let black = F(BLACK); let white = F(WHITE); let clip_at = F(CLIP_AT);
        var v = max((s + black) * white, vec3f(0.0));
        hi = max(v.x, max(v.y, v.z)) > clip_at;
        lo = v.x <= 0.0 && v.y <= 0.0 && v.z <= 0.0;
        if (hi || outside(s)) { v = roll_off((s + black) * white, clip_at); }
        let off = P(OFF_TONE); let scale = F(TONE_SCALE);
        out = vec3f(tables[off + min(round_up(v.x * scale), 65535u)], tables[off + min(round_up(v.y * scale), 65535u)], tables[off + min(round_up(v.z * scale), 65535u)]);
    }
    if (P(CURVES_ON) == 1u) { out = vec3f(curve_read(0u, out.x), curve_read(1u, out.y), curve_read(2u, out.z)); }
    if (P(COLOUR_ON) == 1u) { out = colour_stage(out); }
    let amount = F(VIG_AMOUNT);
    if (amount != 0.0) {
        let dx = (f32(x) + 0.5) / f32(P(WIDTH)) * 2.0 - 1.0; let dy = (f32(y) + 0.5) / f32(P(HEIGHT)) * 2.0 - 1.0;
        let f = ease(F(VIG_MID) - F(VIG_FEATHER), F(VIG_MID) + F(VIG_FEATHER), sqrt(dx * dx + dy * dy) / 1.41421356 * 1.4);
        if (amount < 0.0) { out *= 1.0 + amount * f; } else { out += (vec3f(1.0) - out) * amount * f; }
    }
    let grain = F(GRAIN_AMOUNT);
    if (grain != 0.0) {
        let n = grain_at(x, y, F(GRAIN_CELL)) * grain;
        let lm = luma(out);
        out = clamp(out + vec3f(n * (0.35 + 2.6 * lm * (1.0 - lm))), vec3f(0.0), vec3f(1.0));
    }
    return Shaded(out, hi, lo);
}

@compute @workgroup_size(8, 8)
fn develop(@builtin(global_invocation_id) id: vec3u) {
    let rw = P(RW); let rh = P(RH);
    if (id.x >= rw || id.y >= rh) { return; }
    let x = P(RX) + id.x; let y = P(RY) + id.y;
    let i = id.y * rw + id.x;
    let ww = P(WX1) - P(WX0);
    let w = (y - P(WY0)) * ww + (x - P(WX0));
    let s = exposed(x, y);
    let shaded = shade(s.xyz, x, y, w);
    let max_value = F(MAX);
    var q = vec3f(f32(round_up(shaded.out.x * max_value)), f32(round_up(shaded.out.y * max_value)), f32(round_up(shaded.out.z * max_value)));
    let to_histogram = 255.0 / max_value;
    atomicAdd(&stats[256u + min(round_up(q.x * to_histogram), 255u)], 1u);
    atomicAdd(&stats[512u + min(round_up(q.y * to_histogram), 255u)], 1u);
    atomicAdd(&stats[768u + min(round_up(q.z * to_histogram), 255u)], 1u);
    atomicAdd(&stats[min(round_up(luma(q) * to_histogram), 255u)], 1u);
    var hi = shaded.hi; var lo = shaded.lo;
    if (hi || lo) {
        if (P(UNTOUCHED_ON) == 0u) { hi = false; lo = false; }
        else {
            let o = read_pixel(x, y, UNTOUCHED, 1.0);
            if (hi && max(o.x, max(o.y, o.z)) >= 1.0 - 1e-6) { hi = false; }
            if (lo && o.x <= 0.0 && o.y <= 0.0 && o.z <= 0.0) { lo = false; }
        }
    }
    if (hi) { atomicAdd(&stats[1024u], 1u); }
    if (lo) { atomicAdd(&stats[1025u], 1u); }
    if (P(CLIPPING) == 1u && hi) { q = vec3f(max_value, 0.0, 0.0); } else if (P(CLIPPING) == 1u && lo) { q = vec3f(0.0, 0.0, max_value); }
    let alpha = round_up(s.w * max_value);
    if (P(BIT_DEPTH) == 8u) {
        output[i] = u32(q.x) | (u32(q.y) << 8u) | (u32(q.z) << 16u) | (alpha << 24u);
    } else {
        output[i * 2u] = u32(q.x) | (u32(q.y) << 16u);
        output[i * 2u + 1u] = u32(q.z) | (alpha << 16u);
    }
}
