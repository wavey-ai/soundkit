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
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libx265 -profile:v main -pix_fmt yuv420p -tag:v hvc1 $common_audio -c:a aac -b:a 192k -movflags +faststart "$output_dir/hevc-main-aac.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libx265 -profile:v main10 -pix_fmt yuv420p10le -tag:v hvc1 $common_audio -c:a pcm_s24le -movflags +faststart "$output_dir/hevc-main10-pcm.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le $common_audio -c:a pcm_s24le -movflags +faststart "$output_dir/prores-422-hq-pcm.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v dnxhd -profile:v dnxhr_hqx -pix_fmt yuv422p10le $common_audio -c:a pcm_s24le "$output_dir/dnxhr-hqx-pcm.mov"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libvpx-vp9 -profile:v 0 -pix_fmt yuv420p -b:v 0 -crf 31 $common_audio -c:a libopus -b:a 160k "$output_dir/vp9-profile0-opus.webm"
# shellcheck disable=SC2086
ffmpeg -hide_banner -loglevel error -y -i "$source_media" $common_video -c:v libsvtav1 -preset 10 -pix_fmt yuv420p -crf 35 $common_audio -c:a libopus -b:a 160k "$output_dir/av1-main-opus.webm"

ffmpeg -hide_banner -loglevel error -y -i "$output_dir/h264-high-aac.mp4" -map 0:v:0 -c copy -bsf:v h264_mp4toannexb -f h264 "$output_dir/h264-high.264"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/hevc-main-aac.mov" -map 0:v:0 -c copy -bsf:v hevc_mp4toannexb -f hevc "$output_dir/hevc-main.265"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/hevc-main10-pcm.mov" -map 0:v:0 -c copy -bsf:v hevc_mp4toannexb -f hevc "$output_dir/hevc-main10.265"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/vp9-profile0-opus.webm" -map 0:v:0 -c copy -f ivf "$output_dir/vp9-profile0.ivf"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/av1-main-opus.webm" -map 0:v:0 -c copy -f ivf "$output_dir/av1-main.ivf"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/prores-422-hq-pcm.mov" -map 0:v:0 -c copy -f data "$output_dir/prores-422-hq.bin"

find "$output_dir" -maxdepth 1 -type f -exec shasum -a 256 {} \; | sort -k2 > "$output_dir/SHA256SUMS"
