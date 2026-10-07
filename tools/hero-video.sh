#!/bin/sh
# Home page video: an endless loop with no jump and no dissolve, backdrop toned down to the tile grey, no sound.
# usage: sh tools/hero-video.sh assets/hero/hero-source.mp4 assets/hero
# The clip plays forward and then backward, so its last frame leads straight back into its first one.
# FROM / TO are the turnaround frames: pick moments where the model barely moves, so the reversal does not show
# (in the owner's video the camera drifts a little, so it never returns to its first frame by itself).
set -e
SRC=$1; OUT=$2; FROM=${FROM:-12}; TO=${TO:-230}; TONE=${TONE:-0.93}   # TONE: backdrop ~248 * 0.93 = the tile grey 231
N=$((TO - FROM + 1))
ffmpeg -v error -y -i "$SRC" -an -filter_complex "
  [0:v]trim=start_frame=$FROM:end_frame=$((TO + 1)),setpts=PTS-STARTPTS,lutrgb=r=val*$TONE:g=val*$TONE:b=val*$TONE,scale=1600:-2,split[f][b];
  [b]reverse,trim=start_frame=1:end_frame=$((N - 1)),setpts=PTS-STARTPTS[r];
  [f][r]concat=n=2:v=1:a=0,format=yuv420p[v]" \
  -map "[v]" -r 24 -c:v libx264 -preset slow -crf 26 -profile:v high -movflags +faststart "$OUT/hero.mp4"
ffmpeg -v error -y -i "$OUT/hero.mp4" -frames:v 1 -q:v 4 "$OUT/hero.jpg"
# WebM copy for browsers without H.264 (Chromium builds, some Linux Firefox)
ffmpeg -v error -y -i "$OUT/hero.mp4" -c:v libvpx-vp9 -b:v 0 -crf 38 -row-mt 1 -deadline good -cpu-used 2 "$OUT/hero.webm"
