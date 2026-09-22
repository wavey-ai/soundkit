#include "libraw/libraw.h"
#include <string>
#include <sstream>
#include <cmath>

static LibRaw *decoder = nullptr;
static libraw_processed_image_t *picture = nullptr;
static std::string error, metadata;
static std::string quote(const char *value) {
    std::string result = "\"";
    for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
        if (*p == '"' || *p == '\\') result += '\\';
        if (*p >= 32) result += *p;
    }
    return result + "\"";
}
static int fail(int code) { error = libraw_strerror(code); return 0; }
extern "C" {
void raw_close() {
    if (picture) LibRaw::dcraw_clear_mem(picture);
    picture = nullptr;
    delete decoder; decoder = nullptr;
}
const char *raw_error() { return error.c_str(); }
int raw_open(void *bytes, unsigned length) {
    raw_close(); error.clear();
    try {
        decoder = new LibRaw();
        int code = decoder->open_buffer(bytes, length);
        if (code) return fail(code);
        const auto &s = decoder->imgdata.sizes;
        if (uint64_t(s.raw_width) * s.raw_height > 80000000) { error = "This RAW exceeds the 80 megapixel browser limit."; return 0; }
        return 1;
    } catch (...) { error = "Not enough memory to open this RAW."; return 0; }
}
const char *raw_metadata() {
    if (!decoder) return "{}";
    const auto &d = decoder->imgdata;
    std::ostringstream out;
    out << "{\"make\":" << quote(d.idata.make) << ",\"model\":" << quote(d.idata.model)
        << ",\"width\":" << d.sizes.width << ",\"height\":" << d.sizes.height
        << ",\"orientation\":" << d.sizes.flip << ",\"iso\":" << d.other.iso_speed
        << ",\"aperture\":" << d.other.aperture << ",\"shutter\":" << d.other.shutter
        << ",\"maximum\":" << d.color.maximum << ",\"wb\":[";
    for (int c = 0; c < 4; c++) { if (c) out << ','; out << (d.color.cam_mul[c] > 0 ? d.color.cam_mul[c] : d.color.pre_mul[c]); }
    out << "],\"matrix\":[";
    for (int row = 0; row < 3; row++) for (int c = 0; c < 3; c++) { if (row || c) out << ','; out << d.color.rgb_cam[row][c]; }
    out << "]}"; metadata = out.str(); return metadata.c_str();
}
int raw_thumbnail() {
    if (!decoder) return 0;
    int code = decoder->unpack_thumb();
    if (code || decoder->imgdata.thumbnail.tformat != LIBRAW_THUMBNAIL_JPEG) return 0;
    return int(reinterpret_cast<uintptr_t>(decoder->imgdata.thumbnail.thumb));
}
unsigned raw_thumbnail_size() { return decoder ? decoder->imgdata.thumbnail.tlength : 0; }
int raw_decode(int half) {
    if (!decoder) { error = "Open a RAW first."; return 0; }
    try {
        auto &p = decoder->imgdata.params;
        p.half_size = half ? 1 : 0;
        p.output_bps = 16;
        p.output_color = 0; // Camera RGB: convert to working RGB in floating point, without clipping.
        p.no_auto_scale = 1; // Preserve sensor levels and apply white balance after demosaicing.
        p.no_auto_bright = 1;
        p.adjust_maximum_thr = 0;
        p.gamm[0] = p.gamm[1] = 1;
        p.user_qual = half ? 0 : 3;
        int code = decoder->unpack();
        if (code) return fail(code);
        code = decoder->dcraw_process();
        if (code) return fail(code);
        picture = decoder->dcraw_make_mem_image(&code);
        if (!picture || code) return fail(code ? code : LIBRAW_UNSPECIFIED_ERROR);
        if (picture->bits != 16 || picture->colors != 3) { error = "This RAW's sensor layout is not supported by the RAW decoder."; return 0; }
        return 1;
    } catch (...) { error = "Not enough memory to develop this RAW. Try a smaller file."; return 0; }
}
void *raw_pixels() { return picture ? picture->data : nullptr; }
unsigned raw_size() { return picture ? picture->data_size : 0; }
unsigned raw_width() { return picture ? picture->width : 0; }
unsigned raw_height() { return picture ? picture->height : 0; }
}
