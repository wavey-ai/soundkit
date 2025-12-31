#!/bin/bash
# Pure bash benchmark for ffmpeg
# Runs two modes: native (no conversion) and 16kHz mono

benchmark_format() {
    local format=$1
    local mode=$2  # "native" or "16k_mono"
    local testdata_dir="../testdata/$format"

    if [ ! -d "$testdata_dir" ]; then
        return
    fi

    # Collect files into array
    local files=()
    for file in "$testdata_dir"/*; do
        if [ -f "$file" ]; then
            files+=("$file")
        fi
    done

    local file_count=${#files[@]}

    if [ "$file_count" -eq 0 ]; then
        return
    fi

    local start_time=$(date +%s.%N)
    local total_bytes=0
    local successful=0

    # Set ffmpeg args based on mode
    local ffmpeg_args
    if [ "$mode" = "native" ]; then
        ffmpeg_args="-f s16le -acodec pcm_s16le"
    else
        ffmpeg_args="-f s16le -acodec pcm_s16le -ar 16000 -ac 1"
    fi

    # Process each file
    for file in "${files[@]}"; do
        local file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)

        if ffmpeg -i "$file" $ffmpeg_args - >/dev/null 2>&1; then
            total_bytes=$((total_bytes + file_size))
            successful=$((successful + 1))
        fi
    done

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)
    local files_per_sec=$(echo "scale=2; $successful / $elapsed" | bc)
    local mb_total=$(echo "scale=2; $total_bytes / 1048576" | bc)
    local mb_per_sec=$(echo "scale=2; $mb_total / $elapsed" | bc)

    printf "  %-10s %3d files  %5.1fs  %5.1f files/s  %4.2f MB/s\n" \
        "$format" "$successful" "$elapsed" "$files_per_sec" "$mb_per_sec"
}

FORMATS="mp3 flac ogg_opus mac_aac"

echo "============================================================"
echo "FFmpeg Audio Decoding Benchmark"
echo "============================================================"

echo ""
echo "=== Native (no sample rate/channel conversion) ==="
for fmt in $FORMATS; do
    benchmark_format "$fmt" "native"
done

echo ""
echo "=== 16kHz Mono (resampled + downmixed) ==="
for fmt in $FORMATS; do
    benchmark_format "$fmt" "16k_mono"
done

echo ""
echo "Done!"
