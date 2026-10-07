#!/bin/sh
# Home page video: an endless loop with no jump and no dissolve, backdrop toned down to the tile grey, no sound.
# usage: sh tools/hero-video.sh assets/hero/hero-source.mp4 assets/hero
# The clip plays forward and then backward, so its last frame leads straight back into its first one.
# FROM / TO are the turnaround frames: pick moments where the model barely moves, so the reversal does not show
# (in the owner's video the camera drifts a little, so it never returns to its first frame by itself).
# Full source size (1920 px) and a high bitrate: the source is already compressed, a second pass must not soften it.
# Then the sneakers are cut out of one frame (tools/hero-shoes.py): that cut-out steps over the plate's bottom edge.
set -e
SRC=$1; OUT=$2; FROM=${FROM:-12}; TO=${TO:-230}; K=${TONE:-0.93}   # TONE: backdrop ~248 * 0.93 = the tile grey 231
N=$((TO - FROM + 1))
# toning in YUV is the same as multiplying R, G and B by K, without a round trip through RGB
TONE_F="lutyuv=y=16+(val-16)*$K:u=128+(val-128)*$K:v=128+(val-128)*$K"
ffmpeg -v error -y -i "$SRC" -an -filter_complex "
  [0:v]trim=start_frame=$FROM:end_frame=$((TO + 1)),setpts=PTS-STARTPTS,$TONE_F,split[f][b];
  [b]reverse,trim=start_frame=1:end_frame=$((N - 1)),setpts=PTS-STARTPTS[r];
  [f][r]concat=n=2:v=1:a=0,format=yuv420p[v]" \
  -map "[v]" -r 24 -c:v libx264 -preset slow -crf 19 -tune film -profile:v high -movflags +faststart "$OUT/hero-loop.mp4"
ffmpeg -v error -y -i "$OUT/hero-loop.mp4" -frames:v 1 -q:v 3 "$OUT/hero-loop.jpg"
# WebM copy for browsers without H.264 (Chromium builds, some Linux Firefox)
ffmpeg -v error -y -i "$OUT/hero-loop.mp4" -c:v libvpx-vp9 -b:v 0 -crf 30 -row-mt 1 -deadline good -cpu-used 1 "$OUT/hero-loop.webm"
# the sneakers, from a frame in the middle of the loop
ffmpeg -v error -y -i "$SRC" -vf "select='eq(n\,120)',$TONE_F" -frames:v 1 "$OUT/frame.png"
python3 "$(dirname "$0")/hero-shoes.py" "$OUT/frame.png" "$OUT/hero-shoes.webp"
rm "$OUT/frame.png"
