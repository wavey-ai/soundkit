#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 SOURCE_MOV [OUTPUT_DIRECTORY]" >&2
  exit 2
fi

source_media=$1
output_dir=${2:-build/video-compat/never-final}
mkdir -p "$output_dir"

common_video="-ss 30 -t 3 -map 0:v:0 -vf scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2 -r 25"
common_audio="-map 0:a:0 -ar 48000 -ac 2"

# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libx264 -profile:v high -pix_fmt yuv420p $common_audio -c:a aac -b:a 192k -movflags +faststart "$output_dir/h264-high-aac.mp4"
# CMAF-style fragmented MP4 uses empty sample tables in moov and resolves both
# tracks through moof/traf/trun records. Remuxing keeps codec output identical.
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/h264-high-aac.mp4" -map 0 -c copy -movflags +frag_keyframe+empty_moov+default_base_moof -frag_duration 1000000 "$output_dir/h264-aac-fragmented.mp4"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/h264-high-aac.mp4" -map 0 -c copy -movflags +cmaf+frag_keyframe+empty_moov -frag_duration 1000000 "$output_dir/h264-aac-cmaf.mp4"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/h264-high-aac.mp4" -map 0 -c copy -movflags +dash+frag_keyframe+empty_moov -frag_duration 1000000 "$output_dir/h264-aac-dash.mp4"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/h264-high-aac.mp4" -map 0 -c copy -movflags +frag_keyframe+empty_moov+separate_moof -frag_duration 1000000 "$output_dir/h264-aac-separate-moof.mp4"
# A two-rate presentation timeline exercises non-uniform stts/ctts entries.
ffmpeg -hide_banner -loglevel error -y -ss 30 -t 3 -i "$source_media" -map 0:v:0 -map 0:a:0 -vf "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,select='if(lt(t,1.5),not(mod(n,2)),not(mod(n,3)))',setpts='if(lt(N,19),N*2,(38+(N-19)*3))/(25*TB)'" -fps_mode vfr -c:v libx264 -profile:v high -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart "$output_dir/h264-vfr-aac.mp4"
# High 4:2:2 and 4:4:4 are less common delivery formats, but NLEs can export
# them. Keep deterministic fixtures even while their native decoder support is
# an explicit capability gap.
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libx264 -profile:v high422 -pix_fmt yuv422p $common_audio -c:a aac -b:a 192k -movflags +faststart "$output_dir/h264-high422-aac.mp4"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libx264 -profile:v high444 -pix_fmt yuv444p $common_audio -c:a aac -b:a 192k -movflags +faststart "$output_dir/h264-high444-aac.mp4"
# MP4 permits FLAC while QuickTime MOV does not. Force the common professional
# 24-bit depth; ffmpeg otherwise emits experimental 32-bit FLAC from AAC input.
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/h264-high-aac.mp4" -map 0:v:0 -map 0:a:0 -c:v copy -c:a flac -sample_fmt s32 -bits_per_raw_sample 24 -strict experimental "$output_dir/h264-flac.mp4"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libx265 -profile:v main -pix_fmt yuv420p -tag:v hvc1 $common_audio -c:a aac -b:a 192k -movflags +faststart "$output_dir/hevc-main-aac.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libx265 -profile:v main10 -pix_fmt yuv420p10le -tag:v hvc1 $common_audio -c:a pcm_s24le -movflags +faststart "$output_dir/hevc-main10-pcm.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libx265 -profile:v main422-10 -pix_fmt yuv422p10le -tag:v hvc1 $common_audio -c:a aac -b:a 192k -movflags +faststart "$output_dir/hevc-main422-10-aac.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libx265 -profile:v main444-10 -pix_fmt yuv444p10le -tag:v hvc1 $common_audio -c:a aac -b:a 192k -movflags +faststart "$output_dir/hevc-main444-10-aac.mov"
# Exercise every Apple ProRes profile used by professional export tools.
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v prores_ks -profile:v 0 -pix_fmt yuv422p10le $common_audio -c:a pcm_s24le -movflags +faststart "$output_dir/prores-proxy-pcm.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v prores_ks -profile:v 1 -pix_fmt yuv422p10le $common_audio -c:a pcm_s24le -movflags +faststart "$output_dir/prores-lt-pcm.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v prores_ks -profile:v 2 -pix_fmt yuv422p10le $common_audio -c:a pcm_s24le -movflags +faststart "$output_dir/prores-standard-pcm.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le $common_audio -c:a pcm_s24le -movflags +faststart "$output_dir/prores-422-hq-pcm.mov"
# ProRes 4444 keeps a coded alpha plane. A source without alpha receives an
# opaque plane, which still exercises the professional 12-bit 4:4:4 path.
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v prores_ks -profile:v 4 -pix_fmt yuva444p10le -alpha_bits 16 $common_audio -c:a pcm_s24le -movflags +faststart "$output_dir/prores-4444-alpha-pcm.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v prores_ks -profile:v 5 -pix_fmt yuva444p10le -alpha_bits 16 $common_audio -c:a pcm_s24le -movflags +faststart "$output_dir/prores-4444xq-alpha-pcm.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v dnxhd -profile:v dnxhr_hqx -pix_fmt yuv422p10le $common_audio -c:a pcm_s24le "$output_dir/dnxhr-hqx-pcm.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libvpx-vp9 -profile:v 0 -pix_fmt yuv420p -b:v 0 -crf 31 $common_audio -c:a libopus -b:a 160k "$output_dir/vp9-profile0-opus.webm"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libvpx-vp9 -profile:v 2 -pix_fmt yuv420p10le -b:v 0 -crf 31 $common_audio -c:a libopus -b:a 160k "$output_dir/vp9-profile2-10bit-opus.webm"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libsvtav1 -preset 10 -pix_fmt yuv420p -crf 35 $common_audio -c:a libopus -b:a 160k "$output_dir/av1-main-opus.webm"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libsvtav1 -preset 10 -pix_fmt yuv420p10le -crf 35 $common_audio -c:a libopus -b:a 160k "$output_dir/av1-main10-opus.webm"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/h264-high-aac.mp4" -map 0 -c copy "$output_dir/matroska-h264-aac.mkv"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/hevc-main-aac.mov" -map 0 -c copy "$output_dir/matroska-hevc-aac.mkv"

ffmpeg -hide_banner -loglevel error -y -i "$output_dir/h264-high-aac.mp4" -map 0:v:0 -c copy -bsf:v h264_mp4toannexb -f h264 "$output_dir/h264-high.264"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/hevc-main-aac.mov" -map 0:v:0 -c copy -bsf:v hevc_mp4toannexb -f hevc "$output_dir/hevc-main.265"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/hevc-main10-pcm.mov" -map 0:v:0 -c copy -bsf:v hevc_mp4toannexb -f hevc "$output_dir/hevc-main10.265"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/vp9-profile0-opus.webm" -map 0:v:0 -c copy -f ivf "$output_dir/vp9-profile0.ivf"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/av1-main-opus.webm" -map 0:v:0 -c copy -f ivf "$output_dir/av1-main.ivf"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/prores-422-hq-pcm.mov" -map 0:v:0 -c copy -f data "$output_dir/prores-422-hq.bin"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/prores-4444-alpha-pcm.mov" -map 0:v:0 -c copy -f data "$output_dir/prores-4444-alpha.bin"

find "$output_dir" -maxdepth 1 -type f ! -name SHA256SUMS -exec shasum -a 256 {} \; | sort -k2 > "$output_dir/SHA256SUMS"
