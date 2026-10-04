#include <libheif/heif.h>
#include <libde265/de265.h>
#include <cstdint>
#include <new>
#include <sstream>
#include <string>
#include <vector>

static heif_image *picture = nullptr;
static const uint8_t *plane = nullptr;
static size_t stride = 0;
static int width = 0, height = 0, depth = 0;
static std::vector<uint8_t> profile;
static std::string error, metadata;
static const uint64_t MAX_PIXELS = 80000000;

static int fail(const char *message) { error = message; return 0; }

extern "C" {
// libheif turns the HEVC deblocking and SAO filters off in an Emscripten
// build. A photograph needs the two filters, so the linker sends the call
// here (--wrap) and the filters stay on.
void __real_de265_set_parameter_bool(de265_decoder_context *, enum de265_param, int);
void __wrap_de265_set_parameter_bool(de265_decoder_context *context, enum de265_param param, int value) {
    if (param == DE265_DECODER_PARAM_DISABLE_DEBLOCKING || param == DE265_DECODER_PARAM_DISABLE_SAO) value = 0;
    __real_de265_set_parameter_bool(context, param, value);
}

void skheif_close() {
    if (picture) heif_image_release(picture);
    picture = nullptr; plane = nullptr; stride = 0; width = height = depth = 0;
    profile.clear(); profile.shrink_to_fit();
}
const char *skheif_error() { return error.c_str(); }
// Decodes the primary image to interleaved RGBA with the file's rotation,
// mirror and crop applied. The input must stay alive until this returns.
int skheif_decode(const uint8_t *bytes, size_t length) {
    skheif_close(); error.clear(); metadata = "{}";
    heif_context *context = nullptr;
    heif_image_handle *handle = nullptr;
    heif_decoding_options *options = nullptr;
    int ok = 0;
    try {
        do {
            context = heif_context_alloc();
            if (!context) { fail("Not enough memory to open this photograph."); break; }
            heif_context_get_security_limits(context)->max_image_size_pixels = MAX_PIXELS;
            heif_error status = heif_context_read_from_memory_without_copy(context, bytes, length, nullptr);
            if (status.code) { fail(status.message); break; }
            status = heif_context_get_primary_image_handle(context, &handle);
            if (status.code) { fail(status.message); break; }
            if (uint64_t(heif_image_handle_get_width(handle)) * uint64_t(heif_image_handle_get_height(handle)) > MAX_PIXELS) {
                fail("This photograph exceeds the 80 megapixel browser limit."); break;
            }
            const bool deep = heif_image_handle_get_luma_bits_per_pixel(handle) > 8;
            options = heif_decoding_options_alloc();
            if (!options) { fail("Not enough memory to decode this photograph."); break; }
            // libheif takes the nearest chroma sample unless bilinear is the only algorithm it can use.
            options->color_conversion_options.preferred_chroma_upsampling_algorithm = heif_chroma_upsampling_bilinear;
            options->color_conversion_options.only_use_preferred_chroma_algorithm = 1;
            status = heif_decode_image(handle, &picture, heif_colorspace_RGB,
                deep ? heif_chroma_interleaved_RRGGBBAA_LE : heif_chroma_interleaved_RGBA, options);
            if (status.code || !picture) { fail(status.code ? status.message : "This photograph could not be decoded."); break; }
            plane = heif_image_get_plane_readonly2(picture, heif_channel_interleaved, &stride);
            width = heif_image_get_width(picture, heif_channel_interleaved);
            height = heif_image_get_height(picture, heif_channel_interleaved);
            depth = heif_image_get_bits_per_pixel_range(picture, heif_channel_interleaved);
            if (!plane || width <= 0 || height <= 0 || depth < 8 || depth > 16) { fail("This photograph could not be decoded."); break; }
            const size_t size = heif_image_handle_get_raw_color_profile_size(handle);
            if (size) {
                profile.resize(size);
                if (heif_image_handle_get_raw_color_profile(handle, profile.data()).code) profile.clear();
            }
            std::ostringstream out;
            out << "{\"hasAlpha\":" << (heif_image_handle_has_alpha_channel(handle) ? "true" : "false")
                << ",\"premultiplied\":" << (heif_image_handle_is_premultiplied_alpha(handle) ? "true" : "false");
            heif_color_profile_nclx *nclx = nullptr;
            if (!heif_image_handle_get_nclx_color_profile(handle, &nclx).code && nclx) {
                out << ",\"nclx\":{\"primaries\":" << int(nclx->color_primaries) << ",\"transfer\":" << int(nclx->transfer_characteristics)
                    << ",\"matrix\":" << int(nclx->matrix_coefficients) << ",\"fullRange\":" << (nclx->full_range_flag ? "true" : "false") << "}";
            } else out << ",\"nclx\":null";
            if (nclx) heif_nclx_color_profile_free(nclx);
            out << "}"; metadata = out.str();
            ok = 1;
        } while (false);
    } catch (const std::bad_alloc &) { fail("Not enough memory to decode this photograph.");
    } catch (...) { fail("This photograph could not be decoded."); }
    if (options) heif_decoding_options_free(options);
    if (handle) heif_image_handle_release(handle);
    if (context) heif_context_free(context);
    if (!ok) skheif_close();
    return ok;
}
const uint8_t *skheif_pixels() { return plane; }
unsigned skheif_stride() { return unsigned(stride); }
int skheif_width() { return width; }
int skheif_height() { return height; }
int skheif_depth() { return depth; }
const char *skheif_metadata() { return metadata.c_str(); }
const uint8_t *skheif_profile() { return profile.empty() ? nullptr : profile.data(); }
unsigned skheif_profile_size() { return unsigned(profile.size()); }
}
